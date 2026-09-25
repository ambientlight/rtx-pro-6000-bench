import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dsv41_load import ACTIVE_TOKEN_FIELDS, DrainWindow, LOAD_PATH, has_contention, is_idle, parse_loads
from dsv41_live_acceptance import cancellation_checks
from dsv41_long_tps import Client, Observer


def fixture(running=0, waiting=0, rank=0, stamp=None):
    return {
        "loads": [
            {
                "timestamp": time.time() if stamp is None else stamp,
                "dp_rank": rank,
                "num_running_reqs": running,
                "num_waiting_reqs": waiting,
                **dict.fromkeys(ACTIVE_TOKEN_FIELDS, 0),
                "queues": {
                    "waiting": waiting,
                    "grammar": 0,
                    "paused": 0,
                    "retracted": 0,
                    "prealloc_ready": 0,
                },
                "disaggregation": {"mode": "null", "decode_transfer_queue_reqs": 0},
            }
        ]
    }


class LoadSchemaTest(unittest.TestCase):
    def test_new_envelope_normalizes_without_mutating_and_idle(self):
        payload = fixture()
        original = copy.deepcopy(payload)
        rows = parse_loads(payload)
        self.assertTrue(is_idle(rows))
        self.assertEqual(rows[0]["num_reqs"], 0)
        self.assertEqual(payload, original)

    def test_running_and_waiting_independently_block_idle(self):
        for running, waiting in ((1, 0), (0, 1), (1, 2)):
            with self.subTest(running=running, waiting=waiting):
                rows = parse_loads(fixture(running, waiting))
                self.assertFalse(is_idle(rows))
                self.assertEqual(rows[0]["num_reqs"], running + waiting)

    def test_other_queues_block_idle_even_with_zero_core_counts(self):
        for key in fixture()["loads"][0]["queues"]:
            payload = fixture()
            payload["loads"][0]["queues"][key] = 1
            with self.subTest(queue=key):
                self.assertFalse(is_idle(parse_loads(payload)))
        payload = fixture()
        payload["loads"][0]["disaggregation"]["decode_transfer_queue_reqs"] = 1
        self.assertFalse(is_idle(parse_loads(payload)))

    def test_chunked_prefill_with_zero_request_counts_is_not_idle(self):
        # Real 400k RAM-prefill snapshot: no running/queued requests reported,
        # although one request is actively consuming prefill compute and KV.
        payload = fixture()
        payload["loads"][0].update(
            num_used_tokens=239616,
            num_total_tokens=239616,
            num_active_tokens=239616,
            num_waiting_uncached_tokens=171829,
        )
        rows = parse_loads(payload)
        self.assertEqual(rows[0]["num_reqs"], 0)
        self.assertFalse(is_idle(rows))
        self.assertFalse(has_contention(rows))  # One prefill is not multiple requests.
        for field in ACTIVE_TOKEN_FIELDS:
            payload = fixture()
            payload["loads"][0][field] = 1
            with self.subTest(field=field):
                self.assertFalse(is_idle(parse_loads(payload)))

    def test_cumulative_prefill_metrics_do_not_block_idle(self):
        payload = fixture()
        payload["loads"][0].update(total_prefill_uncached_tokens=409397, total_prefill_busy_us=73636110)
        self.assertTrue(is_idle(parse_loads(payload)))

    def test_empty_legacy_and_malformed_payloads_fail_closed(self):
        payloads = [None, [], fixture()["loads"], {}, {"loads": []}, {"loads": [None]}]
        for field in (
            "timestamp",
            "dp_rank",
            "num_running_reqs",
            "num_waiting_reqs",
            "queues",
            *ACTIVE_TOKEN_FIELDS,
        ):
            payload = fixture()
            del payload["loads"][0][field]
            payloads.append(payload)
        payload = fixture()
        del payload["loads"][0]["queues"]["grammar"]
        payloads.append(payload)
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_loads(payload)
        with self.assertRaises(ValueError):
            is_idle([])

    def test_invalid_counts_fail_closed(self):
        for value in (-1, False, 0.5, "0", None):
            for field in ("num_running_reqs", "num_waiting_reqs", "dp_rank", *ACTIVE_TOKEN_FIELDS):
                payload = fixture()
                payload["loads"][0][field] = value
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    parse_loads(payload)
            for section, key in (("queues", "grammar"), ("disaggregation", "decode_transfer_queue_reqs")):
                payload = fixture()
                payload["loads"][0][section][key] = value
                with self.subTest(value=value, field=key), self.assertRaises(ValueError):
                    parse_loads(payload)

    def test_stale_future_nonfinite_timestamps_fail_closed(self):
        for stamp in (1, 106, float("nan"), float("inf"), False, "100"):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                parse_loads(fixture(stamp=stamp), now=100)

    def test_multiple_ranks_and_duplicate_rank(self):
        payload = fixture(running=1)
        payload["loads"].extend(fixture(running=1, rank=1)["loads"])
        self.assertTrue(has_contention(parse_loads(payload)))
        self.assertFalse(has_contention(parse_loads(fixture(running=1))))
        payload["loads"][1]["dp_rank"] = 0
        with self.assertRaises(ValueError):
            parse_loads(payload)

    def test_old_observer_artifacts_remain_analyzable(self):
        self.assertFalse(has_contention([{"num_reqs": 1, "num_waiting_reqs": 0}]))
        self.assertTrue(has_contention([{"num_reqs": 1, "num_waiting_reqs": 1}]))


class DrainWindowTest(unittest.TestCase):
    def rows(self, now, running=0):
        return parse_loads(fixture(running=running, stamp=now), now=now)

    def test_requires_continuous_idle_period_and_resets_on_activity(self):
        window = DrainWindow(20, {0})
        self.assertFalse(window.observe(self.rows(100), 100))
        self.assertFalse(window.observe(self.rows(110), 110))
        self.assertFalse(window.observe(self.rows(115, running=1), 115))
        self.assertFalse(window.observe(self.rows(120), 120))
        self.assertFalse(window.observe(self.rows(130), 130))
        self.assertTrue(window.observe(self.rows(140), 140))

    def test_missing_rank_and_repeated_snapshot_cannot_drain(self):
        window = DrainWindow(20, {0, 1})
        with self.assertRaises(ValueError):
            window.observe(self.rows(100), 100)
        window = DrainWindow(20, {0})
        self.assertFalse(window.observe(self.rows(100), 100))
        with self.assertRaises(ValueError):
            window.observe(self.rows(100), 120)
        self.assertFalse(window.observe(self.rows(125), 125))

    def test_chunked_prefill_resets_idle_window(self):
        window = DrainWindow(20, {0})
        self.assertFalse(window.observe(self.rows(100), 100))
        active = self.rows(120)
        active[0]["num_active_tokens"] = 409600
        self.assertFalse(window.observe(active, 120))
        self.assertFalse(window.observe(self.rows(125), 125))
        self.assertTrue(window.observe(self.rows(145), 145))


class CollectorEndpointTest(unittest.TestCase):
    def test_tps_client_uses_new_endpoint_and_detects_queued_work(self):
        client = Client(SimpleNamespace(key_file=None))
        client.json = Mock(return_value=fixture())
        client.idle()
        client.json.assert_called_once_with(LOAD_PATH)
        client.json.return_value = fixture(waiting=1)
        with self.assertRaises(RuntimeError):
            client.idle()
        client.json.return_value = {"loads": []}
        with self.assertRaises(ValueError):
            client.idle()

    def test_observer_writes_compatible_rows_from_new_endpoint(self):
        client = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            observer = Observer(client, Path(tmp))

            def load(*args, **kwargs):
                observer.stop.set()
                return fixture(running=1, waiting=1)

            client.json.side_effect = load
            with patch(
                "dsv41_long_tps.subprocess.run", return_value=SimpleNamespace(stdout="0,100,20,100,40\n")
            ):
                observer.thread.start()
                observer.thread.join(timeout=5)
            self.assertFalse(observer.thread.is_alive())
            client.json.assert_called_once_with(LOAD_PATH, timeout=5)
            result = observer.finish()
            self.assertTrue(result["load_contention_observed"])
            self.assertEqual(result["observer_errors"], 0)
            sample = json.loads((Path(tmp) / "observer.jsonl").read_text())
            self.assertEqual(sample["load"][0]["num_reqs"], 2)

    def test_cancellation_checks_use_new_endpoint_throughout(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Mock(base="http://fixture", headers={}, timeout=5, model="fixture", output=Path(tmp))
            client.request.side_effect = lambda name, path: (
                fixture(running=int(name.endswith("-load-active"))) if path == LOAD_PATH else None
            )

            def stream(request, **kwargs):
                kind = (
                    "response.output_text.delta"
                    if request.full_url.endswith("responses")
                    else "content_block_delta"
                )
                response = Mock()
                response.__enter__ = Mock(
                    return_value=[b"data: " + json.dumps({"type": kind}).encode() + b"\n"] * 3
                )
                response.__exit__ = Mock(return_value=False)
                return response

            with patch("dsv41_live_acceptance.urllib.request.urlopen", side_effect=stream):
                cancellation_checks(client)
            paths = [call.args[1] for call in client.request.call_args_list]
            self.assertEqual(paths.count(LOAD_PATH), 6)
            self.assertTrue(all(path in (LOAD_PATH, "/health") for path in paths))
            self.assertEqual(client.passed.call_count, 2)

    def test_no_deprecated_load_calls_remain_in_collectors(self):
        scripts = Path(__file__).resolve().parent
        for name in ("dsv41_long_tps.py", "dsv41_live_acceptance.py", "dsv41_prepare_promotion.py"):
            source = (scripts / name).read_text()
            self.assertNotIn("/get_load", source)
            self.assertIn("LOAD_PATH", source)


if __name__ == "__main__":
    unittest.main()
