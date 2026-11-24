"""Checks for passive live trial accounting, with no production endpoint or GPU."""
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import hour_metrics as m
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/"work/pearl-inference-20261123/prototype"))
from broker import Fleet


def prometheus(n):
    # Non-streaming TTFT and TP replica counters must not inflate streamed timings.
    rows = [f'sglang:http_requests_active{{endpoint="/v1/responses",method="POST"}} 0',
            f'sglang:http_requests_total{{endpoint="/v1/responses",method="POST"}} {n}']
    metrics = {"num_requests_total":1,"prompt_tokens_total":300,"cached_tokens_total":270,
               "generation_tokens_total":17,"uncached_prompt_tokens_histogram_sum":30,
               "time_to_first_token_seconds_count":1,"time_to_first_token_seconds_sum":.125,
               "request_time_per_output_token_seconds_count":1,"request_time_per_output_token_seconds_sum":.01,
               "per_stage_req_latency_seconds_count":1,"per_stage_req_latency_seconds_sum":.3}
    for k,v in metrics.items():
        rows.append(f'sglang:{k}{{stage="prefill_forward",is_streaming="true",account="sensitive"}} {n*v}')
    rows.append(f'sglang:time_to_first_token_seconds_sum{{is_streaming="false"}} {n*99}')
    rows.append(f'sglang:per_stage_req_latency_seconds_sum{{stage="prefill_forward",tp_rank="2"}} {n*88}')
    rows.append(f'sglang:time_to_first_token_seconds_bucket{{is_streaming="true",le="0.5"}} {n}')
    rows.append(f'sglang:cuda_graph_passes_total{{tp_rank="2",mode="decode_cuda_graph"}} {n}')
    return "\n".join(rows)


class Receipts(unittest.TestCase):
    def test_counter_and_histogram_semantics(self):
        before,after=(m.parse_metrics(prometheus(n)) for n in (4,6))
        result=m.serving_diff(before,after)
        self.assertEqual((result["completed_requests"],result["fresh_prompt_tokens"],result["cached_prompt_tokens"],result["output_tokens"]),(2,60,540,34))
        self.assertEqual(result["histograms"]["time_to_first_token_seconds"]["mean_s"],.125)
        self.assertEqual(result["histograms"]["time_to_first_token_seconds"]["p95_bucket_upper_s"],.5)
        self.assertAlmostEqual(result["reciprocal_mean_streaming_request_tpot_tps"],100)
        self.assertAlmostEqual(result["completed_request_prefill_tokens_per_s"],100)
        self.assertNotIn("sensitive",json.dumps(after))
        self.assertNotIn("account",json.dumps(after))

    def test_counter_reset_rejected(self):
        with self.assertRaises(ValueError):
            m.serving_diff(m.parse_metrics(prometheus(2)),m.parse_metrics(prometheus(1)))

    def test_timeline_excludes_identifiers(self):
        output=StringIO()
        fleet=Fleet([],timeline=output,inference_only_ranks=4)
        fleet.idle=[True]*4
        fleet.notify()
        fleet.claims["not-public"]=None
        fleet.notify()
        fleet.notify()
        rows=[json.loads(l) for l in output.getvalue().splitlines()]
        self.assertEqual([r["serving"] for r in rows],[True,False,True])
        self.assertNotIn("not-public",output.getvalue())

    def test_handoff_measurement_brackets(self):
        event=lambda t,s:dict(event="state",monotonic_ns=t,serving=s,watched=4,held=4 if s else 0)
        data=[event(0,True),event(40_000_000,False),event(80_000_000,True),
              dict(event="admitted",monotonic_ns=83_000_000,elapsed_ns=2_000_000,required_drain=True)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"timeline"
            path.write_text("".join(json.dumps(row)+"\n" for row in data))
            work=dict(release_ns=41_000_000,first_gpu_seen_ns=45_000_000)
            summary=m.handoffs(path,20_000_000,100_000_000,[dict(idle=dict(workers=[work]*4))])
        self.assertEqual(summary["broker_idle_permitted_s"],.04)
        self.assertEqual(summary["demand_to_all_worker_drain_ms"]["median_ms"],2)
        self.assertEqual(summary["idle_to_all_gpu_reentry_ms"]["median_ms"],5)

    def test_pool_ledger_excludes_prestart_and_postcutoff(self):
        def event(kind,t):
            return dict(monotonic_ns=t,rank=2,generation=1,rpc_id=8,event=kind,target=f"{(1<<204)-1:064x}")
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"events.jsonl"
            path.write_text("".join(json.dumps(x)+"\n" for x in (event("submitted",20),event("accepted",21),event("rejected",22))))
            result=m.direct_ledger(path,21,22)
        self.assertEqual(result["submitted"],0)
        self.assertEqual(result["accepted"],1)
        self.assertEqual(result["rejected"],0)
        self.assertEqual(result["accepted_work_macs"],1<<52)

    def test_complete_offline_reporting_and_layout(self):
        root=Path(__file__).resolve().parents[1]
        # Published counters from an actual four-rank joined graph microbenchmark.
        fixture=root/"measurements/joined-equal-target-fixed-period.json"
        trace=m.read(fixture)
        result=m.scan_summary(trace["projection_before"],trace["projection_after"],m.graph.graph_deltas(trace["http_before"],trace["http_after"]))
        self.assertTrue(all(result["tickets"]["decode"].values()))
        self.assertTrue(all(result["tickets"]["prefill"].values()))

    def test_energy_integration_and_thermal_units(self):
        rows=[]
        for sec,w in ((1,100),(3,200),(20,250)):
            gpus=[[i,71,w,275,2200,13000,sec*1e6,0] for i in range(4)]
            rows.append(dict(end_ns=sec*10**9,gpu=gpus))
        result=m.thermal(rows)[0]
        self.assertEqual(result["mean_power_w"],150)
        self.assertEqual(result["energy_sample_span_s"],2)
        self.assertEqual(result["sw_thermal_slowdown_s"],19)


if __name__=="__main__":
    unittest.main()
