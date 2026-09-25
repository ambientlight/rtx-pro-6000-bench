import json
import unittest

from dsv41_long_tps import measurements, parse_frame, parse_native_record, validate_events


class LongTPSMetricsTest(unittest.TestCase):
    def fixture(self, status="completed"):
        response = {
            "id": "resp_test",
            "status": status,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "usage": {"input_tokens": 1000, "output_tokens": 60},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "first last"}]}],
        }
        timeline = [
            {
                "seconds": 0.1,
                "event": {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_test"}},
            },
            {"seconds": 2, "event": {"type": "response.output_text.delta", "sequence_number": 1, "delta": "first "}},
            {"seconds": 4, "event": {"type": "response.output_text.delta", "sequence_number": 2, "delta": "last"}},
            {"seconds": 4.1, "event": {"type": "response." + status, "sequence_number": 3, "response": response}},
        ]
        native = {
            "out": {
                "output_ids": list(range(60)),
                "meta_info": {
                    "id": "resp_test",
                    "completion_tokens": 60,
                    "prompt_tokens": 1000,
                    "cached_tokens": 100,
                    "e2e_latency": 4,
                    "decode_throughput": 29.5,
                },
            }
        }
        return timeline, native

    def test_multiline_sse_comments_and_done(self):
        self.assertEqual(parse_frame([": ping"]), None)
        self.assertEqual(parse_frame(["data: [DONE]"]), None)
        self.assertEqual(parse_frame(['data: {"n":', "data: 5}"]), {"n": 5})

    def test_actual_tokens_not_bursts_and_created_is_not_ttft(self):
        timeline, native = self.fixture()
        result, text = measurements(timeline, 4.2, native)
        self.assertEqual(result["output_tokens"], 60)
        self.assertEqual(result["output_bursts"], 2)
        self.assertEqual(result["ttft_client_s"], 2)
        self.assertEqual(result["decode_server_tps"], 29.5)
        self.assertEqual(result["decode_server_s"], 2)
        self.assertEqual(result["ttft_server_s"], 2)
        self.assertEqual(result["effective_prefill_tps"], 450)
        self.assertEqual(text, "first last")

    def test_budget_termination_is_valid(self):
        timeline, native = self.fixture("incomplete")
        self.assertEqual(measurements(timeline, 4.2, native)[0]["status"], "incomplete")

    def test_other_incomplete_is_rejected(self):
        timeline, _ = self.fixture("incomplete")
        timeline[-1]["event"]["response"]["incomplete_details"]["reason"] = "content_filter"
        with self.assertRaises(ValueError):
            validate_events(timeline)

    def test_missing_or_error_terminal_is_rejected(self):
        timeline, _ = self.fixture()
        with self.assertRaises(ValueError):
            validate_events(timeline[:-1])
        timeline[-1]["event"]["type"] = "response.failed"
        with self.assertRaises(ValueError):
            validate_events(timeline)

    def test_sequence_and_assembled_output_must_match(self):
        timeline, _ = self.fixture()
        timeline[1]["event"]["sequence_number"] = 77
        with self.assertRaises(ValueError):
            validate_events(timeline)
        timeline, _ = self.fixture()
        timeline[1]["event"]["delta"] = "corrupted"
        with self.assertRaises(ValueError):
            validate_events(timeline)

    def test_usage_must_match_generated_token_ids(self):
        timeline, native = self.fixture()
        native["out"]["output_ids"].pop()
        with self.assertRaises(ValueError):
            measurements(timeline, 4.2, native)

    def test_native_request_identity_must_match(self):
        timeline, native = self.fixture()
        native["out"]["meta_info"]["id"] = "another_request"
        with self.assertRaises(ValueError):
            measurements(timeline, 4.2, native)

    def test_large_docker_record_inline_timestamps(self):
        timestamp = "2026-09-15T22:16:59.113448770Z "
        record = {"timestamp": "2026-09-15T22:16:59", "out": {"output_ids": [1234, 5678], "text": "abcdef"}}
        raw = json.dumps(record)
        raw = raw.replace("1234", "12" + timestamp + "34").replace("abcdef", "abc" + timestamp + "def")
        self.assertEqual(parse_native_record(timestamp + "[worker] " + raw), record)

    def test_valid_application_timestamp_is_preserved(self):
        record = {"timestamp": "fixture", "text": "2026-09-15T22:16:59.113448770Z content"}
        self.assertEqual(parse_native_record(json.dumps(record)), record)

    def test_total_time_ends_at_terminal_event_not_eof(self):
        timeline, native = self.fixture()
        result, _ = measurements(timeline, 9.0, native)
        self.assertEqual(result["e2e_client_s"], 4.1)
        self.assertEqual(result["stream_eof_client_s"], 9.0)
        self.assertEqual(result["e2e_output_tps"], 60 / 4.1)


if __name__ == "__main__":
    unittest.main()
