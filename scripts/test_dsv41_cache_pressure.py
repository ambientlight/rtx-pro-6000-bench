import contextlib
import io
import unittest

from dsv41_cache_pressure import (
    REPLAYS,
    calibrate,
    parse_args,
    random_words,
    render_prompt,
    selected_metrics,
    validate_answer,
)


class CachePressureTest(unittest.TestCase):
    def test_bounded_defaults(self):
        args = parse_args(["--output", "/tmp/test"])
        self.assertEqual((args.count, args.tokens, args.ram_floor_gib), (64, 256000, 24))
        self.assertEqual(sum(map(len, REPLAYS.values())), 8)
        self.assertTrue(all(0 <= index < count for count, indices in REPLAYS.items() for index in indices))

    def test_rejects_unbounded_or_unsafe_inputs(self):
        for option, value in (("--count", "65"), ("--tokens", "524288"), ("--ram-floor-gib", "16")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--output", "/tmp/test", option, value])

    def test_random_prefixes_are_independent_and_reproducible(self):
        first = random_words("fixture", 0, 1000)
        self.assertEqual(first, random_words("fixture", 0, 1000))
        self.assertNotEqual(first[:100], random_words("fixture", 1, 1000)[:100])
        self.assertEqual(len(set(first)), len(set(random_words("fixture", 0, 2000))))
        self.assertNotEqual(render_prompt("fixture", 0, first)[:100], render_prompt("fixture", 1, first)[:100])

    def test_calibration_exact_and_bounded(self):
        calls = []

        def counter(text):
            calls.append(text)
            return len(text.split())

        text, attempts = calibrate("fixture", 0, 4096, counter)
        self.assertEqual(len(text.split()), 4096)
        self.assertLessEqual(len(attempts), 8)
        self.assertIn("700000", text)

    def test_bad_tokenizer_is_rejected(self):
        with self.assertRaises(RuntimeError):
            calibrate("fixture", 0, 1024, lambda text: 500)

    def test_metrics_avoid_tp_double_count(self):
        raw = "\n".join(
            [
                'sglang:prefill_effective_tokens_total{mode="host_hit",tp_rank="0"} 100',
                'sglang:prefill_effective_tokens_total{mode="host_hit",tp_rank="1"} 100',
                'sglang:hicache_host_used_tokens{tp_rank="0"} 1000',
                'sglang:load_back_bytes_total{cache_type="UnifiedRadixCache"} 9999',
                'sglang:load_back_tokens_total{pool="kv"} 1024',
                'sglang:load_back_tokens_total{pool="swa"} 256',
                'sglang:hicache_swa_host_used_tokens{tp_rank="0"} 512',
                'sglang:hicache_swa_host_used_tokens{tp_rank="1"} 512',
            ]
        )
        result = selected_metrics(raw)
        self.assertEqual(result["host_hit"], 100)
        self.assertEqual(result["hicache_host_used_tokens"], 1000)
        self.assertEqual(result["load_back_bytes_total"], 9999)
        self.assertEqual(result["load_back_kv_tokens_all_ranks"], 1024)
        self.assertEqual(result["load_back_swa_tokens_all_ranks"], 256)
        self.assertEqual(result["hicache_swa_host_used_tokens"], 512)

    def timeline(self, text="700000", count=256000):
        return [
            {"seconds": 1, "event": {"type": "response.output_text.delta", "sequence_number": 0, "delta": text}},
            {
                "seconds": 2,
                "event": {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": count},
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                    },
                },
            },
        ]

    def test_requires_answer_and_exact_length(self):
        validate_answer(self.timeline(), "700000", 256000)
        for timeline in (self.timeline("wrong"), self.timeline(count=1000)):
            with self.assertRaises(RuntimeError):
                validate_answer(timeline, "700000", 256000)


if __name__ == "__main__":
    unittest.main()
