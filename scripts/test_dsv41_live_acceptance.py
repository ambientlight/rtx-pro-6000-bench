import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from dsv41_live_acceptance import (
    completed_messages,
    completed_responses,
    decode_sse,
    long_checks,
    metric_total,
    parse_args,
)


class CacheMetricTest(unittest.TestCase):
    def test_cache_suite_is_bounded_and_opt_in(self):
        self.assertEqual(parse_args(["--suite", "cache"]).suite, "cache")
        self.assertEqual(parse_args([]).suite, "api")

    def test_filters_exact_counter_and_rank(self):
        raw = "\n".join(
            [
                "# TYPE count counter",
                'count{mode="hit",tp_rank="0"} 1.5e3',
                'count{mode="hit",tp_rank="1"} 8000',
                'count_created{mode="hit",tp_rank="0"} 999999',
                'count{mode="input",tp_rank="0"} 42',
            ]
        )
        self.assertEqual(metric_total(raw, "count", {"mode": "hit", "tp_rank": "0"}), 1500)
        self.assertEqual(metric_total(raw, "count", {"mode": "missing"}), 0)

    def test_sums_series_and_unlabelled_samples(self):
        self.assertEqual(metric_total('backup{pool="full"} 8\nbackup{pool="swa"} 16', "backup"), 24)
        self.assertEqual(metric_total("backup 24", "backup"), 24)

    def test_nonfinite_or_negative_values_fail(self):
        for value in ("NaN", "+Inf", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metric_total("backup " + value, "backup")


class LongContextBoundsTest(unittest.TestCase):
    def test_400k_target_fits_v1_production_context(self):
        args = parse_args(["--suite", "long", "--long-contexts", "400000"])
        self.assertEqual(args.long_contexts, [400000])
        self.assertEqual(args.context_limit, 524288)

    def test_520k_target_fits_half_native_context(self):
        args = parse_args(["--suite", "long", "--long-contexts", "520000"])
        self.assertEqual(args.long_contexts, [520000])
        self.assertEqual(args.context_limit, 524288)

    def test_explicit_rollback_context_keeps_400k_bound(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--suite", "long", "--long-contexts", "520000", "--context-limit", "409600"])
        args = parse_args(["--suite", "long", "--long-contexts", "400000", "--context-limit", "409600"])
        self.assertEqual(args.context_limit, 409600)

    def test_default_three_targets_are_unchanged(self):
        self.assertEqual(parse_args([]).long_contexts, [32768, 131072, 200000])

    def test_target_must_leave_output_budget_and_margin(self):
        for target in (524224, 524288, 1023):
            with self.subTest(target=target), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(["--long-contexts", str(target)])
                self.assertEqual(error.exception.code, 2)

    def test_at_most_three_requests(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--long-contexts", "32000", "64000", "128000", "200000"])

    def test_measured_overflow_never_sends_inference(self):
        client = SimpleNamespace(
            long_contexts=[520000],
            context_limit=524288,
            model="deepseek-v4-flash",
            request=mock.Mock(return_value={"count": 524240}),
        )
        with self.assertRaises(AssertionError):
            long_checks(client)
        self.assertEqual(client.request.call_count, 3)
        self.assertTrue(all(call.args[1] == "/v1/tokenize" for call in client.request.call_args_list))


class StreamValidatorTest(unittest.TestCase):
    def test_comments_crlf_and_multiline_sse_data(self):
        raw = ': heartbeat\r\n\r\nevent: fixture\r\ndata: {"type":\r\ndata: "fixture"}\r\n\r\ndata: [DONE]\r\n\r\n'
        self.assertEqual(decode_sse(raw), [{"type": "fixture"}])

    def messages(self):
        return [
            {"type": "message_start", "message": {}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "c", "name": "lookup", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"key":"alpha"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]

    def test_messages_reassembles_tool_arguments(self):
        content, stop = completed_messages(self.messages())
        self.assertEqual(stop, "tool_use")
        self.assertEqual(content[0]["input"], {"key": "alpha"})

    def test_messages_rejects_error_and_late_argument_fragment(self):
        events = self.messages()
        events.insert(-1, {"type": "error"})
        with self.assertRaises(AssertionError):
            completed_messages(events)
        events = self.messages()
        events[2], events[3] = events[3], events[2]
        with self.assertRaises(AssertionError):
            completed_messages(events)

    def responses(self):
        call = {"type": "function_call", "id": "f", "call_id": "c", "name": "lookup", "arguments": '{"key":"alpha"}'}
        return [
            {"type": "response.output_item.added", "sequence_number": 0, "item": call},
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 1,
                "item_id": "f",
                "delta": call["arguments"],
            },
            {"type": "response.completed", "sequence_number": 2, "response": {"status": "completed", "output": [call]}},
        ]

    def test_responses_checks_sequence_and_argument_integrity(self):
        self.assertEqual(completed_responses(self.responses())["output"][0]["name"], "lookup")
        events = self.responses()
        events[1]["sequence_number"] = 0
        with self.assertRaises(AssertionError):
            completed_responses(events)
        events = self.responses()
        events[1]["delta"] = json.dumps({"key": "wrong"})
        with self.assertRaises(AssertionError):
            completed_responses(events)


if __name__ == "__main__":
    unittest.main()
