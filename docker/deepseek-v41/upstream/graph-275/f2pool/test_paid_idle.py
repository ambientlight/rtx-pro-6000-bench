"""F2Pool worker attribution and thermal holds with CPU-only local sockets."""
import asyncio
from contextlib import suppress
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paid_idle as idle


class IdleTests(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_job_rewrites_secret_and_accepts_matching_reply_only(self):
        journal=io.StringIO()
        relay=idle.Relay(2,'privateaccount',journal)
        relay.pid=os.getpid()
        raw=dict(job_id='privateaccount-job',header='01'*76,target=f'{2**204-1:064x}',cert_version=3)
        async def mock_pool(r,w):
            auth=json.loads(await r.readline())
            self.assertEqual(auth['params']['wallet'],'privateaccount')
            self.assertEqual(auth['params']['worker'],'joinedidle2')
            w.write(idle.frame(dict(method='mining.notify',params=raw)))
            w.write(idle.frame(dict(id=1,result=True,error=None,type='v2')))
            await w.drain()
            req=json.loads(await r.readline())
            self.assertEqual(req['params']['job_id'],raw['job_id'])
            self.assertEqual(req['params']['plain_proof'],'mock-public-synthetic')
            for key in (44,req['id']):
                w.write(idle.frame(dict(id=key,result=True,error=None)))
            await w.drain()
            await r.read()
            w.close()
        async with await asyncio.start_server(mock_pool,'127.0.0.1',0) as upstream:
            port=upstream.sockets[0].getsockname()[1]
            with patch.object(idle,'HOST','127.0.0.1'),patch.object(idle,'PORT',port):
                async with await asyncio.start_server(relay.accept,'127.0.0.1',0) as srv:
                    rd,wr=await asyncio.open_connection('127.0.0.1',srv.sockets[0].getsockname()[1])
                    wr.write(idle.frame(dict(method='mining.authorize',id=1,params=dict(worker='joinedidle2'))))
                    await wr.drain()
                    notify=json.loads(await asyncio.wait_for(rd.readline(),3))
                    self.assertNotIn('privateaccount',str(notify))
                    self.assertEqual(json.loads(await rd.readline())['result'],True)
                    alias=notify['params']['job_id']
                    wr.write(idle.frame(dict(method='mining.submit',id=2,params=dict(job_id=alias,plain_proof='mock-public-synthetic'))))
                    await wr.drain()
                    reply=json.loads(await asyncio.wait_for(rd.readline(),3))
                    self.assertTrue(reply['result'])
                    self.assertEqual(relay.state['accepted'],1)
                    self.assertEqual(relay.state['submitted'],1)
                    self.assertEqual(relay.state['accepted_work_macs'],2**52)
                    wr.close()
                    await wr.wait_closed()
                    await asyncio.sleep(.02)
        self.assertNotIn('privateaccount',journal.getvalue())
        self.assertNotIn('mock-public-synthetic',journal.getvalue())

    async def test_thermal_interlock_independent_across_admissions(self):
        # Model the worker's reference-counted HOLD sockets. This tests that the
        # cooler owns exactly one such stream through hysteresis and releases only its own.
        holds=set()
        async def worker(rd,wr):
            self.assertEqual(await rd.readline(),b'HOLD\n')
            holds.add(wr)
            wr.write(b'DRAINED\n')
            await wr.drain()
            self.assertEqual(await rd.readline(),b'RELEASE\n')
            holds.remove(wr)
            wr.write(b'RELEASED\n')
            await wr.drain()
            wr.close()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'worker'
            async with await asyncio.start_unix_server(worker,path=p):
                rd,wr=await asyncio.open_unix_connection(p)
                wr.write(b'HOLD\n')
                await wr.drain()
                self.assertEqual(await rd.readline(),b'DRAINED\n')
                cooler=idle.Cooler(p)
                rank={}
                for t in (88,87,85):
                    await cooler.update(t,rank)
                    self.assertTrue(rank['cooling'])
                    self.assertEqual(len(holds),2)
                await cooler.update(84,rank)
                self.assertFalse(rank['cooling'])
                self.assertEqual(len(holds),1)
                wr.write(b'RELEASE\n')
                await wr.drain()
                self.assertEqual(await rd.readline(),b'RELEASED\n')
                self.assertFalse(holds)
                wr.close()
                await wr.wait_closed()


if __name__=='__main__':
    unittest.main()
