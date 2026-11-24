"""Offline workload tests. No service, model, pool or GPU is required."""
import io
import json
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import workload as w


def job(revision=1):
    return dict(id=f"public-1-{revision}", generation=1, target=f"{1 << 208:064x}",
                share_bits=0x1b010000, header_sha256="c" * 64, upstream_job_sha256="d" * 64)


def peer():
    return dict(run_id="a" * 32, connected=True, authorized_clients=4, accepted=0,
                submitted=0, pending=0, armed=False, job=job())


def ranks(n):
    shapes = {r:[dict(mode="TARGET_VERIFY", path=k, real_rows=2, executed_rows=2, calls=n)
                  for k in ("target_graph", "target_graph_mined")] +
                [dict(mode="EXTEND", path="eager_wo_b", real_rows=2048, executed_rows=2048, calls=n*4)]
              for r in w.RANKS}
    device = dict(verify=dict(verified_steps=n, tickets_scanned=n*320),
                  eager_verify=dict(prefill=dict(verified_steps=n*4, tickets_scanned=n*8192*40)),
                  eager=dict(prefill=dict(mined_rows=n*8192, mined_calls=n*4)))
    return dict(shapes=shapes, counters={r:{"cuda":device} for r in w.RANKS})


class WorkloadTest(unittest.TestCase):
    def test_input_recipe_is_reproducible_and_every_chunk_changes(self):
        def cpu_tokenize(base, model, text):
            return list(text.encode())
        with patch.object(w, "tokenize", cpu_tokenize):
            a, _ = w.public_input(None, None, "ab", 0, 8192)
            b, _ = w.public_input(None, None, "ab", 1, 8192)
            c, _ = w.public_input(None, None, "ab", 0, 8192)
        self.assertEqual(len(a), 8192)
        self.assertEqual(a, c)
        self.assertEqual(len(set(w.chunks(a) + w.chunks(b))), 8)
        self.assertIn("degrees Celsius", bytes(a[-80:]).decode())
        self.assertIn("Final calibration record:", w.public_text("ab", 0, 4))

    def test_exact_model_response(self):
        events = []
        for i in range(1, 4):
            events.append(dict(output_ids=list(range(i)), meta_info=dict(prompt_tokens=7,
                           cached_tokens=0, completion_tokens=i)))
        body = b"".join(b"data: " + json.dumps(e).encode() + b"\n" for e in events)
        with patch.object(w, "request", return_value=io.BytesIO(body)):
            result = w.public_response("local", [9]*7, 3, "s", time.monotonic() + 1)
        self.assertEqual(result["output_ids"], [0,1,2])
        self.assertEqual(result["output_sha256"], w.graph.token_hash([0,1,2]))

    def test_share_odds_and_job_rotation(self):
        a = peer()
        a["armed"] = True
        self.assertEqual(w.expected_shares(a["job"], 1 << 48), 1)
        self.assertEqual(w.stable_job(a, a), a["job"])
        b = dict(a, job=job(2))
        self.assertIsNone(w.stable_job(a, b))
        self.assertIsNone(w.stable_job(a, dict(a, armed=False)))

    def test_local_watch_disarms_after_remote_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm, receipt = (Path(tmp)/name for name in ("arm","receipt"))
            state = peer()
            receipt.write_text(json.dumps(state))
            with w.ArmWindow(arm, receipt, state["run_id"], 3) as window:
                self.assertTrue(arm.is_file())
                w.write(receipt, dict(state, accepted=1))
                self.assertTrue(window.done.wait(1))
            self.assertEqual(window.reason, "pool_accepted")
            self.assertFalse(arm.exists())

    def test_guard_waits_through_warm_readings_and_stale_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"temperature.json"
            def sample(temp, age_s=0):
                return dict(timestamp_ns=time.time_ns()-int(age_s*1e9),
                            temperatures_c=[temp,75,82,70], max_c=temp)
            path.write_text(json.dumps(sample(91, age_s=6)))
            self.assertIsNone(w.temperature(path))
            sequence = iter([sample(89), None, sample(87), sample(84)])
            window = Namespace(done=threading.Event())
            with patch.object(w, "temperature", side_effect=lambda _:next(sequence)), \
                 patch.object(window.done, "wait", return_value=False):
                cooling = w.wait_for_cooling(path, window)
            self.assertEqual(cooling["peak_c"], 89)
            self.assertGreater(cooling["paused_s"], 0)

    def test_verified_wide_prefill_when_some_chunks_bypass_mining(self):
        before, after = ranks(0), ranks(1)
        for r in w.RANKS:
            device = after["counters"][r]["cuda"]
            device["eager"]["prefill"] = dict(mined_calls=2, mined_rows=4096, backpressure_calls=2, backpressure_rows=4096)
            device["eager_verify"]["prefill"] = dict(verified_steps=2, tickets_scanned=40*4096)
        graphs = {r:{"decode_cuda_graph":1} for r in w.RANKS}
        work = w.scan_summary(before, after, graphs)
        self.assertEqual(work["reconciled_mac_equivalent"]["prefill"], 4 * 4096 * 40 * 262_144)
        self.assertEqual(work["ticket_issues"]["prefill"], [])
        with patch.object(w.graph, "drained_projection", return_value=before), \
             patch.object(w.graph, "projection_snapshot", return_value=after) as read:
            self.assertIs(w.request_projection(None, before, graphs, 8192), after)
            read.assert_called_once_with(None)

    def test_complete_offline_run_and_transition_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = Path(tmp)
            args = Namespace(port=12345, key_file=paths/"key", gateway_receipt=paths/"gw.json",
                             receipt=paths/"run.json", arm_file=paths/"armed", shape_dir=paths,
                             seconds=60, temperature_file=None, max_requests=4, input_tokens=8192, output_tokens=8, no_scan_seconds=120)
            state, done = peer(), 0
            def state_now(path, run_id=None):
                return dict(state, armed=args.arm_file.exists(),job=job(1 if done < 2 else 2))
            def snapshot(*args):
                return dict(total=done, active=0, graphs={r:{"decode_cuda_graph":done} for r in w.RANKS})
            def complete(base, ids, output, salt, deadline):
                nonlocal done
                done += 1
                return dict(completed=True, input_tokens=len(ids), output_tokens=output, output_ids=[7]*output)
            profile = dict(model_path=str(w.CHECKPOINT), chunked_prefill_size=w.CHUNK, served_model_name="test")
            with patch.object(w, "gateway", state_now), patch.object(w.graph, "profile", return_value=profile), \
                 patch.object(w.graph, "snapshot", snapshot), patch.object(w.graph, "completed_http_snapshot", snapshot), \
                 patch.object(w.graph, "drained_projection", side_effect=lambda directory:ranks(done)), \
                 patch.object(w, "tokenize", side_effect=lambda base,model,text:list(text.encode())), \
                 patch.object(w, "public_response", complete):
                result = w.run(args)
            self.assertEqual(result["stop_reason"], "request_limit")
            self.assertEqual(result["completed_input_tokens"], 32768)
            self.assertEqual(result["model_issued_tokens"], 32)
            self.assertTrue(all(result["positive_scan_on_all_ranks"].values()))
            self.assertTrue(result["decode_graph_on_all_ranks"])
            self.assertFalse(args.arm_file.exists())
            exposure = result["stable_job_exposure"]
            self.assertEqual(list(exposure), ["public-1-2"])
            self.assertEqual(exposure["public-1-2"]["requests"], 1)
            self.assertEqual(result["unattributed_mac_equivalent"], exposure["public-1-2"]["mac_equivalent"] * 3)
            events = list(map(json.loads, (paths/"run.requests.jsonl").read_text().splitlines()))
            self.assertEqual(len(events), 8)
            self.assertEqual(len({e["input_sha256"] for e in events[::2]}), 4)


if __name__ == "__main__":
    unittest.main()
