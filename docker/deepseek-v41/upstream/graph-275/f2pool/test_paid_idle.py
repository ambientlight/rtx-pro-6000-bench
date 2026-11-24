"""Offline checks for F2Pool worker attribution and prewarm admission."""
import asyncio
import io
import json
import os
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

    def test_prewarm_accepts_warm_cards_until_any_card_reaches_92(self):
        cards = [[str(i), f'GPU-{i}', '850', '275', str(88+i)] for i in range(4)]
        with patch.object(idle, 'gpu_info', return_value=cards):
            self.assertEqual(idle.prewarm_cards(), cards)
            cards[2][4] = '92'
            with self.assertRaisesRegex(RuntimeError, '92 C cutoff'):
                idle.prewarm_cards()


if __name__=='__main__':
    unittest.main()
