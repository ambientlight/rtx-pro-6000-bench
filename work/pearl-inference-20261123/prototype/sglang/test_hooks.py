"""CPU-only tests of the staged hooks and the real local broker protocol."""
import ast
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from broker import Fleet
import pearl_runtime as pearl
from stage import stage

SOURCE = Path(os.environ.get("PEARL_SGL_SOURCE", HERE.parents[4] / "sglang-dsv41-production-overlay"))


def method(path, cls, name, scope):
    mod = ast.parse(path.read_text())
    owner = next(node for node in mod.body if isinstance(node, ast.ClassDef) and node.name == cls)
    func = next(node for node in owner.body if getattr(node, "name", "") == name)
    future = ast.parse("from __future__ import annotations").body[0]
    scope = dict(scope)
    exec(compile(ast.Module([future, func], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class Ingress(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fleet = Fleet(("a", "b"), timeout=2)
        self.path = self.root / "broker"
        self.server = await asyncio.start_unix_server(self.fleet.client, self.path)
        self.ranks = [pearl.Rank(i, str(self.path)) for i in range(2)]
        for rank in self.ranks:
            await asyncio.to_thread(rank.idle, lambda: None)
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        for rank in self.ranks:
            rank.close()
        self.server.close()
        await self.server.wait_closed()

    async def eventually(self, f):
        async with asyncio.timeout(2):
            while not f():
                await asyncio.sleep(.001)

    def drains(self):
        for worker in self.fleet.workers:
            worker.held = True
        self.fleet.notify()

    async def test_parent_and_two_children_then_rank_fences_and_new_arrival(self):
        a = pearl.Admission(self.path)
        await a.start()
        await self.eventually(lambda: a.token in self.fleet.claims)
        self.assertTrue(self.fleet.serving)
        items = [NS(), NS()]
        sends = []
        def dispatch(payload):
            self.assertIs(payload, items)
            self.assertTrue(all(item.pearl_dispatch_id in self.fleet.claims for item in items))
            sends.append(payload)
        t = asyncio.create_task(a.dispatch(dispatch, items, items))
        await asyncio.sleep(.01)
        self.assertFalse(t.done())
        self.drains()
        await t
        self.assertEqual(len(sends), 1)
        await a.close()
        self.assertEqual(len(self.fleet.claims), 2)
        for rank in self.ranks:
            await asyncio.to_thread(rank.receive, items)
        self.assertFalse(self.fleet.claims)
        fences = []
        await asyncio.to_thread(self.ranks[0].idle, lambda: fences.append(0))
        self.assertTrue(self.fleet.serving)
        b = pearl.Admission(self.path)
        await b.start()
        await b.wait()
        await asyncio.to_thread(self.ranks[1].idle, lambda: fences.append(1))
        self.assertTrue(self.fleet.serving)
        await asyncio.to_thread(self.ranks[1].idle, lambda: fences.append(1))
        self.assertEqual(fences, [0, 1])
        await b.close()
        self.assertFalse(self.fleet.serving)

    async def test_tokenizer_death_while_draining(self):
        a = pearl.Admission(self.path)
        await a.start()
        await self.eventually(lambda: self.fleet.serving)
        await a.close()
        await self.eventually(lambda: not self.fleet.serving)

    async def test_ambiguous_dispatch_failure_keeps_child_and_rank_loss_parks(self):
        self.drains()
        a = pearl.Admission(self.path)
        await a.start()
        item = NS()
        def fail(_):
            raise OSError("IPC acknowledgement lost")
        with self.assertRaises(OSError):
            await a.dispatch(fail, item, [item])
        await a.close()
        self.assertEqual(set(self.fleet.claims), {item.pearl_dispatch_id})
        self.ranks[0].close()
        await self.eventually(lambda: not self.fleet.idle[0])

    async def test_real_tokenizer_generator_release_independent_of_client_response(self):
        staged = self.root / "sgl"
        await asyncio.to_thread(stage, SOURCE, staged)
        tokenizer_file = staged / "python/sglang/srt/managers/tokenizer_manager.py"
        class Req:
            def normalize_batch_and_arguments(self): pass
            def contains_mm_input(self): return False
            is_single, return_prompt_token_ids = True, False
            rid, max_thinking_tokens, routed_dp_rank, lora_path = "request", None, None, None

        @asynccontextmanager
        async def allow(): yield
        events = []
        wire = NS()
        a = self
        class Manager:
            is_pause_cond = asyncio.Condition()
            is_pause = False
            mm_processor = tokenizer = None
            model_update_lock = NS(reader_lock=allow())
            request_logger = NS(log_received_request=lambda *_: None)
            rid_to_state = {"request": NS()}
            auto_create_handle_loop = lambda self: None
            _set_default_priority = lambda *a: None
            _init_req_state = lambda *a: None
            _release_req_states_on_failure = lambda *a: None
            async def _validate_and_resolve_lora(self, obj): pass
            async def _tokenize_one_request(self, obj):
                await a.eventually(lambda: a.fleet.serving)
                events.append("cpu_tokenizing_while_miner_drains")
                return wire
            async def _send_one_request(self, obj, lease):
                await lease.dispatch(lambda _: events.append("sent"), obj, [obj])
            async def _wait_one_response(self, obj, request):
                events.append("waiting_for_client")
                yield "streaming"
        real_generate = method(tokenizer_file, "TokenizerManager", "generate_request",
                               dict(pearl=pearl, GenerateReqInput=Req,
                                    get_disagg=lambda: NS(language_only=False, language_model_only=False)))
        with patch.object(pearl, "SOCKET", self.path):
            responses = real_generate(Manager(), Req())
            response = asyncio.create_task(anext(responses))
            await self.eventually(lambda: bool(events))
            self.assertEqual(events, ["cpu_tokenizing_while_miner_drains"])
            self.drains()
            self.assertEqual(await response, "streaming")
            self.assertEqual(set(self.fleet.claims), {wire.pearl_dispatch_id})
            for rank in self.ranks:
                await asyncio.to_thread(rank.receive, [wire])
                await asyncio.to_thread(rank.idle, lambda: None)
            self.assertFalse(self.fleet.serving)
            await responses.aclose()


class Scheduler(unittest.TestCase):
    def test_fence_only_at_fully_idle_edge(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / "staged"
            stage(SOURCE, out)
            path = out / "python/sglang/srt/managers/scheduler.py"
            scheduler = MagicMock()
            scheduler._last_stall_publish_ts = 100
            scheduler.enable_unified_memory = False
            scheduler.enable_hisparse = True
            scheduler.disaggregation_mode = "local"
            scheduler.ps.tp_rank, scheduler.ps.gpu_id = 2, 3
            events = []
            rank = pearl.Rank(2, "fixture")
            rank.active = True
            rank._command = lambda cmd: events.append(cmd)
            scheduler.maybe_sleep_on_idle.side_effect = lambda: events.append("SLEEP")
            runtime = NS(SOCKET="fixture", OBSERVE=False, scheduler_rank=lambda i: rank)
            scope = dict(pearl=runtime, time=NS(monotonic=lambda:100), LOAD_STALL_REFRESH_S=1,
                         torch=NS(cuda=NS(synchronize=lambda i: events.append(f"FENCE {i}"))),
                         DisaggregationMode=NS(DECODE="decode"))
            idle = method(path, "Scheduler", "on_idle", scope)
            scheduler.is_fully_idle.return_value = False
            idle(scheduler)
            self.assertFalse(events)
            scheduler.is_fully_idle.return_value = True
            idle(scheduler)
            idle(scheduler)
            self.assertEqual(events, ["FENCE 3", "IDLE 2", "SLEEP", "SLEEP"])

    def test_configuration_rejects_graph_operator_and_allows_shape_trace(self):
        args = NS(cuda_graph_config=NS(prefill=NS(backend="disabled"), decode=NS(backend="full")),
                  enable_torch_compile=False)
        args.dp_size, args.pp_size = 2, 1
        with patch.object(pearl, "SOCKET", "/broker.sock"):
            with self.assertRaisesRegex(ValueError, "DP1"):
                pearl.validate_config(args)
        args.dp_size = 1
        with patch.object(pearl, "OPERATOR", object()):
            with self.assertRaisesRegex(ValueError, "eager"):
                pearl.validate_config(args)
            args.cuda_graph_config.decode.backend = "disabled"
            pearl.validate_config(args)
            args.enable_torch_compile = True
            with self.assertRaisesRegex(ValueError, "torch.compile"):
                pearl.validate_config(args)
        with patch.object(pearl, "OPERATOR", None):
            pearl.validate_config(args)


class Shapes(unittest.TestCase):
    def test_capture_warmup_padding_graph_and_b1(self):
        make = lambda mode, **kw: NS(forward_mode=NS(name=mode), **kw)
        ctx = pearl.Rank(0, None)
        layer = NS(quant_method=NS(apply=lambda *a, **k: "stock"))
        op = NS(project=lambda *a: "mined")
        one = make("DECODE", batch_size=1)
        x = NS(shape=(1,2048))
        with patch.object(pearl, "_rank", ctx), patch.object(pearl, "OPERATOR", op), patch.object(pearl, "SHAPE_DIR", "enabled"):
            self.assertEqual(pearl.project(layer, x, None, one), "stock")
            ctx.active = True
            with patch.object(pearl, "_capturing", return_value=True):
                self.assertEqual(pearl.project(layer, x, None, one), "stock")
            with patch.object(pearl, "_capturing", return_value=False):
                self.assertEqual(pearl.project(layer, x, None, one), "mined")
                self.assertEqual(pearl.project(layer, NS(shape=(4,2048)), None, one), "stock")
                self.assertEqual(pearl.project(layer, x, None, make("DRAFT_EXTEND")), "stock")
            runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=False, bs=4, captured_req_width=6)
            with self.assertRaisesRegex(RuntimeError, "eager"):
                pearl.before_replay(runner, one)
            uniform = make("TARGET_VERIFY", batch_size=1, spec_info=NS(draft_token_num=6))
            with patch.object(pearl, "OPERATOR", None):
                pearl.before_replay(runner, uniform)
                self.assertEqual(ctx.counts["TARGET_VERIFY", "target_graph", 6, 24], 1)
                runner.model_runner.is_draft_worker = True
                pearl.before_replay(runner, uniform)
                self.assertEqual(ctx.counts["TARGET_VERIFY", "target_graph", 6, 24], 1)
        self.assertIsNone(pearl.real_rows(make("TARGET_VERIFY", batch_size=1,
            spec_info=NS(ragged_verify_layout=NS(total_verify_tokens=None)))))
        self.assertEqual(pearl.real_rows(make("EXTEND", extend_num_tokens=16, num_token_non_padded_cpu=11)), 11)


if __name__ == "__main__":
    unittest.main()
