"""Offline checks for the interactive entrance, time boundaries and normal/thermal restore paths."""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import hour_trial as trial
from hour_ingress import Relay


class RelayTest(unittest.IsolatedAsyncioTestCase):
    async def test_drain_finishes_current_response_and_blocks_new_requests(self):
        got_request=asyncio.Event()
        allow_response=asyncio.Event()
        done=asyncio.Event()
        observed=[]
        async def backend(reader,writer):
            try:
                observed.append(await reader.readexactly(4))
                got_request.set()
                await allow_response.wait()
                writer.write(b'finished')
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            finally:
                done.set()
        server=await asyncio.start_server(backend,'127.0.0.1',0)
        relay=Relay(server.sockets[0].getsockname()[1])
        relay.server=await asyncio.start_server(relay.client,'127.0.0.1',0)
        port=relay.server.sockets[0].getsockname()[1]
        async with server, relay.server:
            reader,writer=await asyncio.open_connection('127.0.0.1',port)
            try:
                writer.write(b'ping')
                await writer.drain()
                await asyncio.wait_for(got_request.wait(),2)
                relay.drain()
                with self.assertRaises(OSError):
                    await asyncio.open_connection('127.0.0.1',port)
                writer.write(b'late')
                await writer.drain()
                allow_response.set()
                self.assertEqual(await asyncio.wait_for(reader.readexactly(8),2),b'finished')
                self.assertEqual(observed,[b'ping'])
                await asyncio.wait_for(done.wait(),2)
            finally:
                writer.close()
                await writer.wait_closed()


class TrialTest(unittest.TestCase):
    def test_restore_disarms_before_drain_and_preserves_thermal_override(self):
        for hot in (False,True):
            with self.subTest(hot=hot), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                run=Path(directory)
                state=run/'serving'
                state.mkdir()
                (state/'cutover.json').touch()
                (run/'armed').touch()
                if hot:
                    (state/'guard.json').touch()
                (state/'window.json').write_text(json.dumps({'end_monotonic_ns':time.monotonic_ns()}))
                stack.enter_context(patch.object(trial.m,'STATE',state))
                stack.enter_context(patch.object(trial.m,'RUN',run))
                stack.enter_context(patch.object(trial.m,'SOCKET',run/'unavailable'))
                stack.enter_context(patch.object(trial.m,'active',return_value=False))
                drain=stack.enter_context(patch.object(trial,'close_ingress',side_effect=lambda:self.assertFalse((run/'armed').exists())))
                recover=stack.enter_context(patch.object(trial.m,'restore'))
                trial.restore()
                self.assertFalse((run/'armed').exists())
                self.assertEqual(drain.call_count,0 if hot else 1)
                recover.assert_called_once()

    def test_open_uses_pinned_diagnostic_smokes_and_original_model_name(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            run=Path(directory)
            state=run/'serving'
            state.mkdir()
            (run/'metrics').mkdir()
            (run/'metrics/ready.json').touch()
            (state/'trial.json').write_text('{"gateway_run_id":"live"}')
            for name in trial.m.SMOKES:
                (state/name).write_text('{"status":"passed"}')
            stack.enter_context(patch.object(trial.m,'STATE',state))
            stack.enter_context(patch.object(trial.m,'RUN',run))
            stack.enter_context(patch.object(trial.m,'active',return_value=True))
            stack.enter_context(patch.object(trial,'gateway',side_effect=lambda:dict(
                run_id='live',connected=True,armed=(run/'armed').exists(),authorized_clients=4,
                job={'id':'armed' if (run/'armed').exists() else 'old'})))
            info=dict(tp_size=4,ep_size=4,speculative_algorithm='DSPARK',served_model_name='deepseek-v4-flash',
                      cuda_graph_config={'decode':{'backend':'full'},'prefill':{'backend':'disabled'}})
            stack.enter_context(patch.object(trial,'public_get',side_effect=lambda port,path:
                json.dumps(info if path=='/server_info' else {'data':[{'id':'deepseek-v4-flash'}]}).encode()))
            launch=stack.enter_context(patch.object(trial.m,'unit'))
            timer=stack.enter_context(patch.object(trial,'recovery_timer'))
            trial.open_hour()
            window=trial.read_json(state/'window.json')
            self.assertEqual(window['end_monotonic_ns']-window['start_monotonic_ns'],3_600_000_000_000)
            self.assertEqual(window['active_workers'],4)
            self.assertEqual(launch.call_args.args[0],trial.PUBLIC)
            timer.assert_called_once_with(trial.END,3600,trial.m.PYTHON,trial.HERE,'restore')


if __name__=='__main__':
    unittest.main()
