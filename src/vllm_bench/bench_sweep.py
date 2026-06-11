#!/usr/bin/env python3
"""
Benchmark sweep harness for vLLM serving.

Sweeps concurrency levels upward (1,2,4,8,16,24,32,...) until benchmark
duration stops decreasing, then aggregates results and produces matplotlib
charts.

Usage:
  # Full sweep
  python bench_sweep.py \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250

  # Start from concurrency 64
  python bench_sweep.py \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250 --min-concurrency 64

  # Plot-only (no benchmarks, just aggregate existing results)
  python bench_sweep.py --plot-only \
    --model-id qwen35-397b-a17b-nvfp4 --watt 250

  # Dry run (print commands without executing)
  python bench_sweep.py --dry-run \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250

  # Compare multiple input lengths on the same chart
  python bench_sweep.py --compare \
    --model-id qwen35-397b-a17b-nvfp4 --watt 250 \
    --input-lens 2048,7167 --output-len 1024

  # Matrix sweep: all combinations of input/output lengths
  python bench_sweep.py --matrix \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250 --input-lens 2048,7167,15359 --output-len 1024

  # Matrix sweep with GPU telemetry (power draw + KV cache monitoring)
  python bench_sweep.py --matrix --telemetry \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250 --input-lens 2048,7167 --output-lens 512,1024

  # Matrix sweep with per-input-len max concurrency (avoid OOM on large inputs)
  python bench_sweep.py --matrix --telemetry \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250 --input-lens 2048,7167,15359,31743,64511 \
    --max-concurrency 128,64,32,16,8 --output-len 1024

  # Re-aggregate matrix results (no new benchmarks)
  python bench_sweep.py --matrix --plot-only \
    --model-id qwen35-397b-a17b-nvfp4 --watt 250 \
    --input-lens 2048,7167,15359 --output-len 1024

  # Single sweep with telemetry
  python bench_sweep.py --telemetry \
    --model-id qwen35-397b-a17b-nvfp4 \
    --tokenizer /mnt/hot/ambientlight/models/qwen35-397b-a17b-nvfp4 \
    --watt 250
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isnan(x: float) -> bool:
    return x != x  # NaN is the only float not equal to itself


def _percentile(data: list[float], pct: float) -> float:
    """Simple percentile (nearest rank).  pct in 0-100."""
    if not data:
        return 0.0
    s = sorted(data)
    k = max(0, min(int(len(s) * pct / 100), len(s) - 1))
    return s[k]


# ---------------------------------------------------------------------------
# GPU Telemetry
# ---------------------------------------------------------------------------

@dataclass
class TelemetrySample:
    """Single point-in-time GPU + vLLM telemetry reading."""
    timestamp: float
    elapsed_s: float
    # Per-GPU metrics (lists, one entry per GPU)
    gpu_power_w: list[float] = field(default_factory=list)
    gpu_mem_used_gb: list[float] = field(default_factory=list)
    gpu_util_pct: list[float] = field(default_factory=list)
    gpu_temp_c: list[float] = field(default_factory=list)
    gpu_mem_bw_util_pct: list[float] = field(default_factory=list)
    gpu_pcie_tx_mb_s: list[float] = field(default_factory=list)
    gpu_pcie_rx_mb_s: list[float] = field(default_factory=list)
    # vLLM /metrics (scalars, NaN if unavailable)
    kv_cache_pct: float = float("nan")
    requests_running: float = float("nan")
    requests_waiting: float = float("nan")


@dataclass
class TelemetrySummary:
    """Aggregated statistics from a telemetry collection run."""
    num_samples: int = 0
    duration_s: float = 0.0
    # Power (sum across GPUs per sample, then aggregated over time)
    mean_power_w: float = 0.0
    max_power_w: float = 0.0
    min_power_w: float = 0.0
    p50_power_w: float = 0.0
    p95_power_w: float = 0.0
    # GPU utilization (mean across GPUs per sample, then over time)
    mean_gpu_util_pct: float = 0.0
    max_gpu_util_pct: float = 0.0
    # Memory bandwidth utilization (mean across GPUs per sample, then over time)
    mean_gpu_mem_bw_util_pct: float = 0.0
    max_gpu_mem_bw_util_pct: float = 0.0
    # Memory (sum across GPUs per sample)
    mean_gpu_mem_used_gb: float = 0.0
    max_gpu_mem_used_gb: float = 0.0
    # Temperature (hottest GPU per sample)
    mean_gpu_temp_c: float = 0.0
    max_gpu_temp_c: float = 0.0
    # PCIe throughput (sum across GPUs per sample, MB/s)
    mean_pcie_tx_mb_s: float = 0.0
    max_pcie_tx_mb_s: float = 0.0
    mean_pcie_rx_mb_s: float = 0.0
    max_pcie_rx_mb_s: float = 0.0
    # KV cache
    mean_kv_cache_pct: float = 0.0
    max_kv_cache_pct: float = 0.0
    p50_kv_cache_pct: float = 0.0
    p95_kv_cache_pct: float = 0.0
    # Requests
    mean_requests_running: float = 0.0
    max_requests_running: float = 0.0
    mean_requests_waiting: float = 0.0
    max_requests_waiting: float = 0.0


class TelemetryCollector:
    """Collects GPU telemetry in a background daemon thread."""

    def __init__(
        self,
        interval_s: float = 0.25,
        base_url: str = "http://127.0.0.1:8000",
        gpu_ids: list[int] | None = None,
    ) -> None:
        self.interval_s = interval_s
        self.metrics_url = base_url.rstrip("/") + "/metrics"
        self._samples: list[TelemetrySample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._gpu_handles: list = []
        self._gpu_indices: list[int] = []
        self._nvml_available = False
        self._pynvml: Any = None
        self._vllm_metrics_available = True

        # Try to initialise pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = gpu_ids if gpu_ids is not None else list(range(count))
            self._gpu_indices = indices
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
            self._nvml_available = True
            self._pynvml = pynvml
            if gpu_ids is not None:
                print(f"  [TELEMETRY] Monitoring GPUs: {gpu_ids}")
        except Exception as e:
            print(f"  [TELEMETRY] pynvml unavailable: {e} — skipping GPU metrics")

    # -- sampling ----------------------------------------------------------

    def _sample_once(self) -> TelemetrySample:
        now = time.time()
        elapsed = now - self._start_time

        power: list[float] = []
        mem: list[float] = []
        util: list[float] = []
        mem_bw_util: list[float] = []
        temp: list[float] = []
        pcie_tx: list[float] = []
        pcie_rx: list[float] = []

        if self._nvml_available:
            pynvml = self._pynvml
            for h in self._gpu_handles:
                try:
                    power.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mem.append(info.used / (1024**3))
                    rates = pynvml.nvmlDeviceGetUtilizationRates(h)
                    util.append(float(rates.gpu))
                    mem_bw_util.append(float(rates.memory))
                    temp.append(float(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    ))
                    # PCIe throughput (KB/s -> MB/s)
                    pcie_tx.append(
                        pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
                    )
                    pcie_rx.append(
                        pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
                    )
                except Exception:
                    power.append(float("nan"))
                    mem.append(float("nan"))
                    util.append(float("nan"))
                    mem_bw_util.append(float("nan"))
                    temp.append(float("nan"))
                    pcie_tx.append(float("nan"))
                    pcie_rx.append(float("nan"))

        kv_pct = float("nan")
        req_running = float("nan")
        req_waiting = float("nan")

        if self._vllm_metrics_available:
            try:
                with urllib.request.urlopen(self.metrics_url, timeout=1.0) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                for line in body.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("vllm:kv_cache_usage_perc"):
                        kv_pct = float(line.split()[-1]) * 100.0
                    elif line.startswith("vllm:num_requests_running"):
                        req_running = float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_waiting"):
                        req_waiting = float(line.split()[-1])
                    # sglang exposes different prometheus names (needs enable_metrics=true).
                    elif line.startswith("sglang:token_usage"):
                        kv_pct = float(line.split()[-1]) * 100.0
                    elif line.startswith("sglang:num_running_reqs"):
                        req_running = float(line.split()[-1])
                    elif line.startswith("sglang:num_queue_reqs"):
                        req_waiting = float(line.split()[-1])
            except Exception:
                pass

        return TelemetrySample(
            timestamp=now, elapsed_s=elapsed,
            gpu_power_w=power, gpu_mem_used_gb=mem,
            gpu_util_pct=util, gpu_mem_bw_util_pct=mem_bw_util,
            gpu_temp_c=temp,
            gpu_pcie_tx_mb_s=pcie_tx, gpu_pcie_rx_mb_s=pcie_rx,
            kv_cache_pct=kv_pct,
            requests_running=req_running,
            requests_waiting=req_waiting,
        )

    def _collection_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._samples.append(self._sample_once())
            except Exception:
                pass  # never break the benchmark
            self._stop_event.wait(self.interval_s)

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        self._start_time = time.time()
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self.compute_summary()

    def compute_summary(self) -> TelemetrySummary:
        samples = self._samples
        if not samples:
            return TelemetrySummary()

        # Per-sample aggregates
        total_power = []   # sum across GPUs
        avg_util = []      # mean across GPUs
        avg_mem_bw_util = [] # mean across GPUs (memory bandwidth utilization %)
        total_mem = []     # sum across GPUs
        max_temp = []      # max across GPUs
        total_pcie_tx = [] # sum across GPUs (MB/s)
        total_pcie_rx = [] # sum across GPUs (MB/s)
        kv_vals = []
        running_vals = []
        waiting_vals = []

        for s in samples:
            if s.gpu_power_w:
                pw = [v for v in s.gpu_power_w if not _isnan(v)]
                if pw:
                    total_power.append(sum(pw))
            if s.gpu_util_pct:
                ut = [v for v in s.gpu_util_pct if not _isnan(v)]
                if ut:
                    avg_util.append(statistics.mean(ut))
            if s.gpu_mem_bw_util_pct:
                mb = [v for v in s.gpu_mem_bw_util_pct if not _isnan(v)]
                if mb:
                    avg_mem_bw_util.append(statistics.mean(mb))
            if s.gpu_mem_used_gb:
                mg = [v for v in s.gpu_mem_used_gb if not _isnan(v)]
                if mg:
                    total_mem.append(sum(mg))
            if s.gpu_temp_c:
                tp = [v for v in s.gpu_temp_c if not _isnan(v)]
                if tp:
                    max_temp.append(max(tp))
            if s.gpu_pcie_tx_mb_s:
                tx = [v for v in s.gpu_pcie_tx_mb_s if not _isnan(v)]
                if tx:
                    total_pcie_tx.append(sum(tx))
            if s.gpu_pcie_rx_mb_s:
                rx = [v for v in s.gpu_pcie_rx_mb_s if not _isnan(v)]
                if rx:
                    total_pcie_rx.append(sum(rx))
            if not _isnan(s.kv_cache_pct):
                kv_vals.append(s.kv_cache_pct)
            if not _isnan(s.requests_running):
                running_vals.append(s.requests_running)
            if not _isnan(s.requests_waiting):
                waiting_vals.append(s.requests_waiting)

        ts = TelemetrySummary(
            num_samples=len(samples),
            duration_s=samples[-1].elapsed_s if samples else 0.0,
        )
        if total_power:
            ts.mean_power_w = statistics.mean(total_power)
            ts.max_power_w = max(total_power)
            ts.min_power_w = min(total_power)
            ts.p50_power_w = _percentile(total_power, 50)
            ts.p95_power_w = _percentile(total_power, 95)
        if avg_util:
            ts.mean_gpu_util_pct = statistics.mean(avg_util)
            ts.max_gpu_util_pct = max(avg_util)
        if avg_mem_bw_util:
            ts.mean_gpu_mem_bw_util_pct = statistics.mean(avg_mem_bw_util)
            ts.max_gpu_mem_bw_util_pct = max(avg_mem_bw_util)
        if total_mem:
            ts.mean_gpu_mem_used_gb = statistics.mean(total_mem)
            ts.max_gpu_mem_used_gb = max(total_mem)
        if max_temp:
            ts.mean_gpu_temp_c = statistics.mean(max_temp)
            ts.max_gpu_temp_c = max(max_temp)
        if total_pcie_tx:
            ts.mean_pcie_tx_mb_s = statistics.mean(total_pcie_tx)
            ts.max_pcie_tx_mb_s = max(total_pcie_tx)
        if total_pcie_rx:
            ts.mean_pcie_rx_mb_s = statistics.mean(total_pcie_rx)
            ts.max_pcie_rx_mb_s = max(total_pcie_rx)
        if kv_vals:
            ts.mean_kv_cache_pct = statistics.mean(kv_vals)
            ts.max_kv_cache_pct = max(kv_vals)
            ts.p50_kv_cache_pct = _percentile(kv_vals, 50)
            ts.p95_kv_cache_pct = _percentile(kv_vals, 95)
        if running_vals:
            ts.mean_requests_running = statistics.mean(running_vals)
            ts.max_requests_running = max(running_vals)
        if waiting_vals:
            ts.mean_requests_waiting = statistics.mean(waiting_vals)
            ts.max_requests_waiting = max(waiting_vals)
        return ts

    def save_csv(self, path: str) -> None:
        if not self._samples:
            return
        num_gpus = len(self._samples[0].gpu_power_w) if self._samples else 0
        gpu_labels = self._gpu_indices if self._gpu_indices else list(range(num_gpus))
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp", "elapsed_s"]
            for g in gpu_labels:
                header += [f"gpu{g}_power_w", f"gpu{g}_mem_used_gb",
                           f"gpu{g}_util_pct", f"gpu{g}_mem_bw_util_pct",
                           f"gpu{g}_temp_c",
                           f"gpu{g}_pcie_tx_mb_s", f"gpu{g}_pcie_rx_mb_s"]
            header += ["kv_cache_pct", "requests_running", "requests_waiting"]
            writer.writerow(header)
            for s in self._samples:
                row: list[str] = [f"{s.timestamp:.3f}", f"{s.elapsed_s:.3f}"]
                for g in range(num_gpus):
                    row.append(f"{s.gpu_power_w[g]:.1f}" if g < len(s.gpu_power_w) and not _isnan(s.gpu_power_w[g]) else "")
                    row.append(f"{s.gpu_mem_used_gb[g]:.2f}" if g < len(s.gpu_mem_used_gb) and not _isnan(s.gpu_mem_used_gb[g]) else "")
                    row.append(f"{s.gpu_util_pct[g]:.0f}" if g < len(s.gpu_util_pct) and not _isnan(s.gpu_util_pct[g]) else "")
                    row.append(f"{s.gpu_mem_bw_util_pct[g]:.0f}" if g < len(s.gpu_mem_bw_util_pct) and not _isnan(s.gpu_mem_bw_util_pct[g]) else "")
                    row.append(f"{s.gpu_temp_c[g]:.0f}" if g < len(s.gpu_temp_c) and not _isnan(s.gpu_temp_c[g]) else "")
                    row.append(f"{s.gpu_pcie_tx_mb_s[g]:.1f}" if g < len(s.gpu_pcie_tx_mb_s) and not _isnan(s.gpu_pcie_tx_mb_s[g]) else "")
                    row.append(f"{s.gpu_pcie_rx_mb_s[g]:.1f}" if g < len(s.gpu_pcie_rx_mb_s) and not _isnan(s.gpu_pcie_rx_mb_s[g]) else "")
                row.append(f"{s.kv_cache_pct:.1f}" if not _isnan(s.kv_cache_pct) else "")
                row.append(f"{s.requests_running:.0f}" if not _isnan(s.requests_running) else "")
                row.append(f"{s.requests_waiting:.0f}" if not _isnan(s.requests_waiting) else "")
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Concurrency schedule
# ---------------------------------------------------------------------------

def build_concurrency_schedule(
    min_conc: int,
    max_conc: int,
    step: int,
) -> list[int]:
    """Build the list of concurrency levels to sweep.

    Geometric ramp: 1, 2, 4, 8  then linear steps of `step` after that.
    If min_conc > 8, skip geometric part entirely and start from min_conc
    rounded down to the nearest step boundary (or min_conc itself).
    """
    geometric = [1, 2, 4, 8]
    schedule: list[int] = []

    if min_conc <= 8:
        # include geometric portion that is >= min_conc
        for g in geometric:
            if g >= min_conc and g <= max_conc:
                schedule.append(g)
        # continue with linear steps after 8
        c = 8 + step
        while c <= max_conc:
            schedule.append(c)
            c += step
    else:
        # start directly at min_conc, step linearly
        c = min_conc
        while c <= max_conc:
            schedule.append(c)
            c += step

    # ensure max_conc is included if it's not already the last entry
    if schedule and schedule[-1] != max_conc:
        schedule.append(max_conc)
    if not schedule:
        schedule = [max_conc]

    return schedule


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    concurrency: int
    duration: float
    output_throughput: float
    max_output_tokens_per_s: float
    total_token_throughput: float
    request_throughput: float
    # TTFT
    mean_ttft_ms: float
    median_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    # TPOT
    mean_tpot_ms: float
    median_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float
    # ITL
    mean_itl_ms: float
    median_itl_ms: float
    p50_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    # E2EL
    mean_e2el_ms: float
    median_e2el_ms: float
    p50_e2el_ms: float
    p95_e2el_ms: float
    p99_e2el_ms: float
    # Meta
    completed: int = 0
    failed: int = 0
    num_prompts: int = 0
    json_path: str = ""
    # Telemetry (populated when --telemetry is active)
    telemetry_csv: str = ""
    mean_power_w: float = 0.0
    max_power_w: float = 0.0
    p50_power_w: float = 0.0
    p95_power_w: float = 0.0
    mean_gpu_util_pct: float = 0.0
    max_gpu_util_pct: float = 0.0
    mean_gpu_mem_bw_util_pct: float = 0.0
    max_gpu_mem_bw_util_pct: float = 0.0
    mean_gpu_mem_used_gb: float = 0.0
    max_gpu_mem_used_gb: float = 0.0
    mean_gpu_temp_c: float = 0.0
    max_gpu_temp_c: float = 0.0
    mean_pcie_tx_mb_s: float = 0.0
    max_pcie_tx_mb_s: float = 0.0
    mean_pcie_rx_mb_s: float = 0.0
    max_pcie_rx_mb_s: float = 0.0
    mean_kv_cache_pct: float = 0.0
    max_kv_cache_pct: float = 0.0
    p95_kv_cache_pct: float = 0.0
    mean_requests_running: float = 0.0
    max_requests_running: float = 0.0
    mean_requests_waiting: float = 0.0
    max_requests_waiting: float = 0.0

    def error_rate(self) -> float:
        """Fraction of requests that failed (0.0-1.0)."""
        total = self.completed + self.failed
        if total == 0:
            return 1.0  # no data at all → treat as error
        return self.failed / total

    def has_errors(self, max_error_rate: float = 0.0) -> bool:
        """True if the error rate exceeds the allowed threshold."""
        return self.error_rate() > max_error_rate


def result_dir_name(
    model_id: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    watt: int,
) -> str:
    safe_model = model_id.replace("/", "__")
    return f"{safe_model}_random_{input_len}in_{output_len}out_c{concurrency}_W{watt}"


def load_result(json_path: str) -> dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def find_latest_result(directory: str) -> str | None:
    """Find the most recently modified JSON file in a result directory.

    Excludes telemetry_summary.json which is not a benchmark result.
    """
    pattern = os.path.join(directory, "*.json")
    files = [f for f in glob.glob(pattern) if not f.endswith("telemetry_summary.json")]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_result(json_path: str, concurrency: int) -> BenchResult:
    data = load_result(json_path)
    return BenchResult(
        concurrency=concurrency,
        duration=data["duration"],
        output_throughput=data["output_throughput"],
        max_output_tokens_per_s=data.get("max_output_tokens_per_s", 0),
        total_token_throughput=data["total_token_throughput"],
        request_throughput=data["request_throughput"],
        mean_ttft_ms=data["mean_ttft_ms"],
        median_ttft_ms=data["median_ttft_ms"],
        p50_ttft_ms=data["p50_ttft_ms"],
        p95_ttft_ms=data["p95_ttft_ms"],
        p99_ttft_ms=data["p99_ttft_ms"],
        mean_tpot_ms=data["mean_tpot_ms"],
        median_tpot_ms=data["median_tpot_ms"],
        p50_tpot_ms=data["p50_tpot_ms"],
        p95_tpot_ms=data["p95_tpot_ms"],
        p99_tpot_ms=data["p99_tpot_ms"],
        mean_itl_ms=data["mean_itl_ms"],
        median_itl_ms=data["median_itl_ms"],
        p50_itl_ms=data["p50_itl_ms"],
        p95_itl_ms=data["p95_itl_ms"],
        p99_itl_ms=data["p99_itl_ms"],
        mean_e2el_ms=data["mean_e2el_ms"],
        median_e2el_ms=data["median_e2el_ms"],
        p50_e2el_ms=data["p50_e2el_ms"],
        p95_e2el_ms=data["p95_e2el_ms"],
        p99_e2el_ms=data["p99_e2el_ms"],
        completed=data.get("completed", 0),
        failed=data.get("failed", 0),
        num_prompts=data.get("num_prompts", 0),
        json_path=json_path,
    )


def _attach_telemetry_summary(result: BenchResult, rdir: str) -> None:
    """If a telemetry_summary.json exists in rdir, populate result fields."""
    summary_path = os.path.join(rdir, "telemetry_summary.json")
    if not os.path.exists(summary_path):
        return
    try:
        with open(summary_path) as f:
            ts = json.load(f)
        result.mean_power_w = ts.get("mean_power_w", 0.0)
        result.max_power_w = ts.get("max_power_w", 0.0)
        result.p50_power_w = ts.get("p50_power_w", 0.0)
        result.p95_power_w = ts.get("p95_power_w", 0.0)
        result.mean_gpu_util_pct = ts.get("mean_gpu_util_pct", 0.0)
        result.max_gpu_util_pct = ts.get("max_gpu_util_pct", 0.0)
        result.mean_gpu_mem_bw_util_pct = ts.get("mean_gpu_mem_bw_util_pct", 0.0)
        result.max_gpu_mem_bw_util_pct = ts.get("max_gpu_mem_bw_util_pct", 0.0)
        result.mean_gpu_mem_used_gb = ts.get("mean_gpu_mem_used_gb", 0.0)
        result.max_gpu_mem_used_gb = ts.get("max_gpu_mem_used_gb", 0.0)
        result.mean_gpu_temp_c = ts.get("mean_gpu_temp_c", 0.0)
        result.max_gpu_temp_c = ts.get("max_gpu_temp_c", 0.0)
        result.mean_pcie_tx_mb_s = ts.get("mean_pcie_tx_mb_s", 0.0)
        result.max_pcie_tx_mb_s = ts.get("max_pcie_tx_mb_s", 0.0)
        result.mean_pcie_rx_mb_s = ts.get("mean_pcie_rx_mb_s", 0.0)
        result.max_pcie_rx_mb_s = ts.get("max_pcie_rx_mb_s", 0.0)
        result.mean_kv_cache_pct = ts.get("mean_kv_cache_pct", 0.0)
        result.max_kv_cache_pct = ts.get("max_kv_cache_pct", 0.0)
        result.p95_kv_cache_pct = ts.get("p95_kv_cache_pct", 0.0)
        result.mean_requests_running = ts.get("mean_requests_running", 0.0)
        result.max_requests_running = ts.get("max_requests_running", 0.0)
        result.mean_requests_waiting = ts.get("mean_requests_waiting", 0.0)
        result.max_requests_waiting = ts.get("max_requests_waiting", 0.0)
        csv_path = os.path.join(rdir, "telemetry.csv")
        if os.path.exists(csv_path):
            result.telemetry_csv = csv_path
    except (json.JSONDecodeError, KeyError, OSError):
        pass


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    concurrency: int,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> BenchResult | None:
    """Run a single vllm bench serve at the given concurrency level."""
    rdir = os.path.join(
        args.result_dir,
        result_dir_name(args.model_id, args.input_len, args.output_len, concurrency, args.watt),
    )

    # Bench launcher. Default invokes vLLM's CLI entry point via this interpreter
    # (avoids relying on a `vllm` console script on PATH, whose shebang can go stale).
    # Override with --bench-cmd for other OpenAI-compatible bench tools.
    prefix = (
        shlex.split(args.bench_cmd)
        if getattr(args, "bench_cmd", None)
        else [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve"]
    )
    cmd = [
        *prefix,
        "--backend", "openai",
        "--base-url", args.base_url,
        "--endpoint", "/v1/completions",
        "--model", args.model_id,
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--num-prompts", str(args.num_prompts),
        "--num-warmups", str(concurrency),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--tokenizer", args.tokenizer,
        "--temperature", "0",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir", rdir,
    ]

    print(f"\n{'='*70}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Result dir:  {rdir}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return None

    # Start telemetry collection (if enabled)
    collector = None
    if getattr(args, "telemetry", False):
        try:
            collector = TelemetryCollector(
                interval_s=getattr(args, "telemetry_interval", 0.25),
                base_url=args.base_url,
                gpu_ids=getattr(args, "gpu_ids", None),
            )
            collector.start()
        except Exception as e:
            print(f"  [TELEMETRY] Failed to start: {e}")
            collector = None

    t0 = time.time()
    proc = subprocess.run(cmd)
    _ = time.time() - t0  # wall-clock elapsed (used for logging if needed)

    # Stop telemetry and save
    telem_summary = None
    if collector is not None:
        try:
            telem_summary = collector.stop()
            os.makedirs(rdir, exist_ok=True)
            csv_path = os.path.join(rdir, "telemetry.csv")
            collector.save_csv(csv_path)
            summary_path = os.path.join(rdir, "telemetry_summary.json")
            with open(summary_path, "w") as f:
                json.dump(asdict(telem_summary), f, indent=2)
            print(f"  Telemetry:   {telem_summary.num_samples} samples → {csv_path}")
            print(f"  Power:       mean={telem_summary.mean_power_w:.0f}W  "
                  f"max={telem_summary.max_power_w:.0f}W  "
                  f"total={telem_summary.mean_power_w * telem_summary.duration_s / 3600:.1f}Wh")
            print(f"  GPU util:    mean={telem_summary.mean_gpu_util_pct:.0f}%  "
                  f"mem BW util: mean={telem_summary.mean_gpu_mem_bw_util_pct:.0f}%  "
                  f"max={telem_summary.max_gpu_mem_bw_util_pct:.0f}%")
            print(f"  KV cache:    mean={telem_summary.mean_kv_cache_pct:.1f}%  "
                  f"max={telem_summary.max_kv_cache_pct:.1f}%")
            print(f"  PCIe TX/RX:  mean={telem_summary.mean_pcie_tx_mb_s:.0f}/"
                  f"{telem_summary.mean_pcie_rx_mb_s:.0f} MB/s  "
                  f"max={telem_summary.max_pcie_tx_mb_s:.0f}/"
                  f"{telem_summary.max_pcie_rx_mb_s:.0f} MB/s")
        except Exception as e:
            print(f"  [TELEMETRY] Error during stop/save: {e}")

    if proc.returncode != 0:
        print(f"  [ERROR] vllm bench exited with code {proc.returncode}")
        return None

    json_path = find_latest_result(rdir)
    if not json_path:
        print(f"  [ERROR] No JSON result found in {rdir}")
        return None

    result = parse_result(json_path, concurrency)
    print(f"  Duration:    {result.duration:.1f}s")
    print(f"  Throughput:  {result.output_throughput:.1f} tok/s (peak {result.max_output_tokens_per_s:.0f})")
    print(f"  Completed:   {result.completed}/{result.num_prompts} (failed: {result.failed})")

    # Attach telemetry to result
    if telem_summary is not None:
        result.telemetry_csv = os.path.join(rdir, "telemetry.csv")
        result.mean_power_w = telem_summary.mean_power_w
        result.max_power_w = telem_summary.max_power_w
        result.p50_power_w = telem_summary.p50_power_w
        result.p95_power_w = telem_summary.p95_power_w
        result.mean_gpu_util_pct = telem_summary.mean_gpu_util_pct
        result.max_gpu_util_pct = telem_summary.max_gpu_util_pct
        result.mean_gpu_mem_bw_util_pct = telem_summary.mean_gpu_mem_bw_util_pct
        result.max_gpu_mem_bw_util_pct = telem_summary.max_gpu_mem_bw_util_pct
        result.mean_gpu_mem_used_gb = telem_summary.mean_gpu_mem_used_gb
        result.max_gpu_mem_used_gb = telem_summary.max_gpu_mem_used_gb
        result.mean_gpu_temp_c = telem_summary.mean_gpu_temp_c
        result.max_gpu_temp_c = telem_summary.max_gpu_temp_c
        result.mean_pcie_tx_mb_s = telem_summary.mean_pcie_tx_mb_s
        result.max_pcie_tx_mb_s = telem_summary.max_pcie_tx_mb_s
        result.mean_pcie_rx_mb_s = telem_summary.mean_pcie_rx_mb_s
        result.max_pcie_rx_mb_s = telem_summary.max_pcie_rx_mb_s
        result.mean_kv_cache_pct = telem_summary.mean_kv_cache_pct
        result.max_kv_cache_pct = telem_summary.max_kv_cache_pct
        result.p95_kv_cache_pct = telem_summary.p95_kv_cache_pct
        result.mean_requests_running = telem_summary.mean_requests_running
        result.max_requests_running = telem_summary.max_requests_running
        result.mean_requests_waiting = telem_summary.mean_requests_waiting
        result.max_requests_waiting = telem_summary.max_requests_waiting

    # Generate per-run telemetry time-series plots
    if telem_summary is not None and not getattr(args, "no_plot", False):
        try:
            csv_path = os.path.join(rdir, "telemetry.csv")
            plot_telemetry_timeseries(csv_path, concurrency, args)
        except Exception as e:
            print(f"  [TELEMETRY] Failed to generate plots: {e}")

    if result.has_errors(args.max_error_rate):
        print(f"  [SKIPPED] Error rate {result.error_rate():.1%} exceeds threshold "
              f"{args.max_error_rate:.1%} — discarding result")
        return None

    return result


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> list[BenchResult]:
    schedule = build_concurrency_schedule(args.min_concurrency, args.max_concurrency, args.step_size)

    print(f"Concurrency schedule: {schedule}")
    if args.dry_run:
        print("[DRY RUN MODE — no benchmarks will be executed]\n")

    results: list[BenchResult] = []
    best_duration: float | None = None
    points_past_best: int = 0  # how many data points collected after best

    for i, conc in enumerate(schedule):
        # --reuse: skip if result already exists
        if args.reuse and not args.dry_run:
            rdir = os.path.join(
                args.result_dir,
                result_dir_name(args.model_id, args.input_len, args.output_len, conc, args.watt),
            )
            existing_json = find_latest_result(rdir)
            if existing_json:
                try:
                    result = parse_result(existing_json, conc)
                    if result.has_errors(args.max_error_rate):
                        print(f"\n[REUSE] c={conc}: existing result has error rate "
                              f"{result.error_rate():.1%} (>{args.max_error_rate:.1%}), re-running")
                    else:
                        _attach_telemetry_summary(result, rdir)
                        print(f"\n[REUSE] c={conc}: loaded existing result "
                              f"(duration={result.duration:.1f}s, {result.output_throughput:.1f} tok/s)")
                        results.append(result)
                        if best_duration is None or result.duration < best_duration:
                            best_duration = result.duration
                            points_past_best = 0
                        else:
                            points_past_best += 1
                        continue
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"\n[REUSE] c={conc}: failed to parse existing result, re-running ({e})")

        result = run_benchmark(conc, args, dry_run=args.dry_run)

        if args.dry_run:
            continue

        if result is None:
            print(f"  Skipping concurrency {conc} due to error")
            continue

        results.append(result)

        # Early stop: collect early_stop_extra points past the best duration
        if not args.no_early_stop:
            if best_duration is None or result.duration < best_duration:
                best_duration = result.duration
                points_past_best = 0
            else:
                points_past_best += 1
                if points_past_best >= args.early_stop_extra:
                    print(f"\n*** SATURATION DETECTED at concurrency {conc} ***")
                    print(f"    Best duration was {best_duration:.1f}s; "
                          f"collected {points_past_best} extra point(s) past best.")
                    print("    Stopping sweep.")
                    break

    return results


# ---------------------------------------------------------------------------
# Collect existing results (for plot-only mode)
# ---------------------------------------------------------------------------

def collect_existing_results(
    args: argparse.Namespace,
    input_len: int | None = None,
    output_len: int | None = None,
) -> list[BenchResult]:
    """Scan result_dir for existing benchmark results matching the config.

    If input_len / output_len are given they override args.input_len / args.output_len,
    which is useful for the --compare mode that collects multiple sweeps.
    """
    _input_len = input_len if input_len is not None else args.input_len
    _output_len = output_len if output_len is not None else args.output_len
    safe_model = args.model_id.replace("/", "__")
    pattern = f"{safe_model}_random_{_input_len}in_{_output_len}out_c*_W{args.watt}"
    base = Path(args.result_dir)
    results: list[BenchResult] = []

    for d in sorted(base.glob(pattern)):
        if not d.is_dir():
            continue
        # extract concurrency from dir name
        dirname = d.name
        try:
            c_part = dirname.split("_c")[1].split("_W")[0]
            conc = int(c_part)
        except (IndexError, ValueError):
            continue

        json_path = find_latest_result(str(d))
        if json_path:
            try:
                result = parse_result(json_path, conc)
                if result.has_errors(args.max_error_rate):
                    print(f"  [SKIP] c={conc:>4d}  error_rate={result.error_rate():.1%} "
                          f"(>{args.max_error_rate:.1%})  "
                          f"failed={result.failed}/{result.completed + result.failed}  "
                          f"from {os.path.basename(json_path)}")
                    continue
                _attach_telemetry_summary(result, str(d))
                results.append(result)
                print(f"  Loaded c={conc:>4d}  duration={result.duration:.1f}s  "
                      f"throughput={result.output_throughput:.1f} tok/s  "
                      f"from {os.path.basename(json_path)}")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [WARN] Failed to parse {json_path}: {e}")

    results.sort(key=lambda r: r.concurrency)
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[BenchResult]) -> None:
    if not results:
        print("No results to summarize.")
        return

    has_telemetry = any(r.mean_power_w > 0 or r.max_kv_cache_pct > 0 for r in results)
    has_pcie = any(r.mean_pcie_tx_mb_s > 0 or r.mean_pcie_rx_mb_s > 0 for r in results)
    has_mem_bw = any(r.mean_gpu_mem_bw_util_pct > 0 for r in results)

    header = (
        f"{'Conc':>6s} | {'Duration':>10s} | {'Out tok/s':>10s} | {'Peak tok/s':>10s} | "
        f"{'Total tok/s':>11s} | {'TTFT p50':>10s} | {'TTFT p95':>10s} | "
        f"{'TPOT p50':>10s} | {'ITL p50':>10s} | {'E2EL p50':>10s}"
    )
    if has_telemetry:
        header += f" | {'Power(W)':>10s} | {'KV cache%':>10s}"
    if has_mem_bw:
        header += f" | {'MemBW%':>10s}"
    if has_pcie:
        header += f" | {'TX MB/s':>10s} | {'RX MB/s':>10s}"

    print(f"\n{'='*len(header)}")
    print("  SWEEP SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r.concurrency:>6d} | {r.duration:>10.1f} | {r.output_throughput:>10.1f} | "
            f"{r.max_output_tokens_per_s:>10.0f} | {r.total_token_throughput:>11.1f} | "
            f"{r.p50_ttft_ms:>10.1f} | {r.p95_ttft_ms:>10.1f} | "
            f"{r.p50_tpot_ms:>10.2f} | {r.p50_itl_ms:>10.2f} | {r.p50_e2el_ms:>10.0f}"
        )
        if has_telemetry:
            line += f" | {r.mean_power_w:>10.0f} | {r.max_kv_cache_pct:>10.1f}"
        if has_mem_bw:
            line += f" | {r.mean_gpu_mem_bw_util_pct:>10.1f}"
        if has_pcie:
            line += f" | {r.mean_pcie_tx_mb_s:>10.0f} | {r.mean_pcie_rx_mb_s:>10.0f}"
        print(line)

    # Highlight best throughput
    best = max(results, key=lambda r: r.output_throughput)
    print(f"\n  Best output throughput: {best.output_throughput:.1f} tok/s at concurrency {best.concurrency}")
    print(f"  Best peak throughput:   {best.max_output_tokens_per_s:.0f} tok/s at concurrency {best.concurrency}")

    best_dur = min(results, key=lambda r: r.duration)
    print(f"  Shortest duration:      {best_dur.duration:.1f}s at concurrency {best_dur.concurrency}")

    if has_telemetry:
        peak_power = max(results, key=lambda r: r.max_power_w)
        print(f"  Peak power draw:        {peak_power.max_power_w:.0f}W at concurrency {peak_power.concurrency}")
        peak_kv = max(results, key=lambda r: r.max_kv_cache_pct)
        print(f"  Peak KV cache usage:    {peak_kv.max_kv_cache_pct:.1f}% at concurrency {peak_kv.concurrency}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list[BenchResult], args: argparse.Namespace) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("\n[WARN] matplotlib not installed. Skipping plots.")
        print("  Install with: pip install matplotlib")
        return

    # Apply concurrency cutoff for charts
    if args.max_plot_concurrency is not None:
        results = [r for r in results if r.concurrency <= args.max_plot_concurrency]

    if len(results) < 2:
        print("\n[WARN] Need at least 2 data points to plot. Skipping.")
        return

    safe_model = args.model_id.replace("/", "__")
    plot_dir = os.path.join(
        args.result_dir, "plots",
        f"{safe_model}_{args.input_len}in_{args.output_len}out_W{args.watt}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    concs = [r.concurrency for r in results]
    seq_len_part = f"  |  seq {args.max_seq_len}" if args.max_seq_len is not None else ""
    suptitle_base = f"{args.model_id}{seq_len_part}  |  {args.input_len}in/{args.output_len}out  |  W{args.watt}"

    def _save(fig: Any, name: str) -> None:
        path = os.path.join(plot_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    def _style_ax(ax: Any, xlabel: str = "Concurrency", ylabel: str = "") -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=9)

    # 1) Throughput vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concs, [r.output_throughput for r in results], "o-", label="Output tok/s", linewidth=2)
    ax.plot(concs, [r.max_output_tokens_per_s for r in results], "s--", label="Peak output tok/s", linewidth=1.5, alpha=0.7)
    ax.plot(concs, [r.total_token_throughput for r in results], "^--", label="Total tok/s", linewidth=1.5, alpha=0.7)
    ax.set_title(f"Throughput vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="Tokens / second")
    _save(fig, "throughput_vs_concurrency")

    # 2) Duration vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concs, [r.duration for r in results], "o-", label="Benchmark duration", linewidth=2, color="tab:red")
    ax.set_title(f"Benchmark Duration vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="Duration (seconds)")
    _save(fig, "duration_vs_concurrency")

    # 3) TTFT vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concs, [r.mean_ttft_ms for r in results], "o-", label="Mean", linewidth=2)
    ax.plot(concs, [r.p50_ttft_ms for r in results], "s--", label="P50", linewidth=1.5)
    ax.plot(concs, [r.p95_ttft_ms for r in results], "^--", label="P95", linewidth=1.5)
    ax.plot(concs, [r.p99_ttft_ms for r in results], "d:", label="P99", linewidth=1.5, alpha=0.7)
    ax.set_title(f"Time to First Token vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="TTFT (ms)")
    _save(fig, "ttft_vs_concurrency")

    # 4) TPOT vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concs, [r.mean_tpot_ms for r in results], "o-", label="Mean", linewidth=2)
    ax.plot(concs, [r.p50_tpot_ms for r in results], "s--", label="P50", linewidth=1.5)
    ax.plot(concs, [r.p95_tpot_ms for r in results], "^--", label="P95", linewidth=1.5)
    ax.plot(concs, [r.p99_tpot_ms for r in results], "d:", label="P99", linewidth=1.5, alpha=0.7)
    ax.set_title(f"Time per Output Token vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="TPOT (ms)")
    _save(fig, "tpot_vs_concurrency")

    # 5) ITL vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concs, [r.mean_itl_ms for r in results], "o-", label="Mean", linewidth=2)
    ax.plot(concs, [r.p50_itl_ms for r in results], "s--", label="P50", linewidth=1.5)
    ax.plot(concs, [r.p95_itl_ms for r in results], "^--", label="P95", linewidth=1.5)
    ax.plot(concs, [r.p99_itl_ms for r in results], "d:", label="P99", linewidth=1.5, alpha=0.7)
    ax.set_title(f"Inter-Token Latency vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="ITL (ms)")
    _save(fig, "itl_vs_concurrency")

    # 6) E2EL vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(concs, [r.mean_e2el_ms / 1000 for r in results], "o-", label="Mean", linewidth=2)
    ax.plot(concs, [r.p50_e2el_ms / 1000 for r in results], "s--", label="P50", linewidth=1.5)
    ax.plot(concs, [r.p95_e2el_ms / 1000 for r in results], "^--", label="P95", linewidth=1.5)
    ax.set_title(f"End-to-End Latency vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="E2EL (seconds)")
    _save(fig, "e2el_vs_concurrency")

    # 7) Combined overview (2x2 grid of key metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark Sweep Overview\n{suptitle_base}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(concs, [r.output_throughput for r in results], "o-", linewidth=2, label="Output tok/s")
    ax.plot(concs, [r.max_output_tokens_per_s for r in results], "s--", linewidth=1.5, alpha=0.7, label="Peak tok/s")
    _style_ax(ax, ylabel="Tokens / second")
    ax.set_title("Throughput")

    ax = axes[0, 1]
    ax.plot(concs, [r.duration for r in results], "o-", linewidth=2, color="tab:red", label="Duration")
    _style_ax(ax, ylabel="Duration (s)")
    ax.set_title("Benchmark Duration")

    ax = axes[1, 0]
    ax.plot(concs, [r.p50_ttft_ms / 1000 for r in results], "s--", linewidth=1.5, label="TTFT P50")
    ax.plot(concs, [r.p95_ttft_ms / 1000 for r in results], "^--", linewidth=1.5, label="TTFT P95")
    _style_ax(ax, ylabel="Seconds")
    ax.set_title("Time to First Token")

    ax = axes[1, 1]
    ax.plot(concs, [r.p50_tpot_ms for r in results], "s--", linewidth=1.5, label="TPOT P50")
    ax.plot(concs, [r.p50_itl_ms for r in results], "o-", linewidth=1.5, label="ITL P50")
    _style_ax(ax, ylabel="ms")
    ax.set_title("TPOT / ITL")

    fig.tight_layout()
    _save(fig, "overview")

    print(f"\n  All plots saved to: {plot_dir}/")


# ---------------------------------------------------------------------------
# Compare plots — overlay multiple sweeps on the same chart
# ---------------------------------------------------------------------------

def plot_compare(
    series: dict[str, list[BenchResult]],
    args: argparse.Namespace,
) -> None:
    """Produce comparison charts with one curve per sweep (keyed by label)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("\n[WARN] matplotlib not installed. Skipping plots.")
        print("  Install with: pip install matplotlib")
        return

    if not series:
        print("\n[WARN] No series to compare.")
        return

    # Apply concurrency cutoff for charts
    if args.max_plot_concurrency is not None:
        series = {
            label: [r for r in res if r.concurrency <= args.max_plot_concurrency]
            for label, res in series.items()
        }
        series = {label: res for label, res in series.items() if res}

    if not series:
        print("\n[WARN] No series with data after concurrency cutoff.")
        return

    safe_model = args.model_id.replace("/", "__")
    plot_dir = os.path.join(args.result_dir, "plots", f"{safe_model}_compare_W{args.watt}")
    os.makedirs(plot_dir, exist_ok=True)

    seq_len_part = f"  |  seq {args.max_seq_len}" if args.max_seq_len is not None else ""
    suptitle_base = f"{args.model_id}{seq_len_part}  |  W{args.watt}"

    # consistent colors per series
    cmap = matplotlib.colormaps["tab10"]
    labels = sorted(series.keys())
    colors = {label: cmap(i) for i, label in enumerate(labels)}

    def _save(fig: Any, name: str) -> None:
        path = os.path.join(plot_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    def _style_ax(ax: Any, xlabel: str = "Concurrency", ylabel: str = "") -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=9)

    # --- 1) Throughput vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.output_throughput for r in res], "o-", label=f"{label}", linewidth=2, color=colors[label])
    ax.set_title(f"Output Throughput vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="Tokens / second")
    _save(fig, "compare_throughput_vs_concurrency")

    # --- 2) Duration vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.duration for r in res], "o-", label=f"{label}", linewidth=2, color=colors[label])
    ax.set_title(f"Benchmark Duration vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="Duration (seconds)")
    _save(fig, "compare_duration_vs_concurrency")

    # --- 3) TTFT P50 vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.p50_ttft_ms for r in res], "o-", label=f"{label}", linewidth=2, color=colors[label])
    ax.set_title(f"TTFT P50 vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="TTFT P50 (ms)")
    _save(fig, "compare_ttft_p50_vs_concurrency")

    # --- 4) TPOT P50 vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.p50_tpot_ms for r in res], "o-", label=f"{label}", linewidth=2, color=colors[label])
    ax.set_title(f"TPOT P50 vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="TPOT P50 (ms)")
    _save(fig, "compare_tpot_p50_vs_concurrency")

    # --- 5) ITL P50 vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.p50_itl_ms for r in res], "o-", label=f"{label}", linewidth=2, color=colors[label])
    ax.set_title(f"ITL P50 vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="ITL P50 (ms)")
    _save(fig, "compare_itl_p50_vs_concurrency")

    # --- 6) E2EL P50 vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.p50_e2el_ms / 1000 for r in res], "o-", label=f"{label}", linewidth=2, color=colors[label])
    ax.set_title(f"E2EL P50 vs Concurrency\n{suptitle_base}")
    _style_ax(ax, ylabel="E2EL P50 (seconds)")
    _save(fig, "compare_e2el_p50_vs_concurrency")

    # --- 7) Combined overview P50 (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparison Overview (P50)\n{suptitle_base}", fontsize=13)

    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        c = colors[label]

        axes[0, 0].plot(concs, [r.output_throughput for r in res], "o-", linewidth=2, color=c, label=label)
        axes[0, 1].plot(concs, [r.duration for r in res], "o-", linewidth=2, color=c, label=label)
        axes[1, 0].plot(concs, [r.p50_ttft_ms / 1000 for r in res], "o-", linewidth=1.5, color=c, label=label)
        axes[1, 1].plot(concs, [r.p50_tpot_ms for r in res], "o-", linewidth=1.5, color=c, label=label)

    _style_ax(axes[0, 0], ylabel="Tokens / second")
    axes[0, 0].set_title("Output Throughput")
    _style_ax(axes[0, 1], ylabel="Duration (s)")
    axes[0, 1].set_title("Benchmark Duration")
    _style_ax(axes[1, 0], ylabel="Seconds")
    axes[1, 0].set_title("TTFT P50")
    _style_ax(axes[1, 1], ylabel="ms")
    axes[1, 1].set_title("TPOT P50")

    fig.tight_layout()
    _save(fig, "compare_overview_p50")

    # --- 8) Combined overview P95/P99 (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparison Overview (P95 / P99)\n{suptitle_base}", fontsize=13)

    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        c = colors[label]

        axes[0, 0].plot(concs, [r.p95_ttft_ms / 1000 for r in res], "o-", linewidth=2, color=c, label=f"{label} P95")
        axes[0, 0].plot(concs, [r.p99_ttft_ms / 1000 for r in res], "x--", linewidth=1.5, alpha=0.7, color=c, label=f"{label} P99")
        axes[0, 1].plot(concs, [r.p95_tpot_ms for r in res], "o-", linewidth=2, color=c, label=f"{label} P95")
        axes[0, 1].plot(concs, [r.p99_tpot_ms for r in res], "x--", linewidth=1.5, alpha=0.7, color=c, label=f"{label} P99")
        axes[1, 0].plot(concs, [r.p95_itl_ms for r in res], "o-", linewidth=2, color=c, label=f"{label} P95")
        axes[1, 0].plot(concs, [r.p99_itl_ms for r in res], "x--", linewidth=1.5, alpha=0.7, color=c, label=f"{label} P99")
        axes[1, 1].plot(concs, [r.p95_e2el_ms / 1000 for r in res], "o-", linewidth=2, color=c, label=f"{label} P95")
        axes[1, 1].plot(concs, [r.p99_e2el_ms / 1000 for r in res], "x--", linewidth=1.5, alpha=0.7, color=c, label=f"{label} P99")

    _style_ax(axes[0, 0], ylabel="Seconds")
    axes[0, 0].set_title("TTFT P95 / P99")
    _style_ax(axes[0, 1], ylabel="ms")
    axes[0, 1].set_title("TPOT P95 / P99")
    _style_ax(axes[1, 0], ylabel="ms")
    axes[1, 0].set_title("ITL P95 / P99")
    _style_ax(axes[1, 1], ylabel="Seconds")
    axes[1, 1].set_title("E2EL P95 / P99")

    fig.tight_layout()
    _save(fig, "compare_overview_p95_p99")

    print(f"\n  All comparison plots saved to: {plot_dir}/")


# ---------------------------------------------------------------------------
# Telemetry plots
# ---------------------------------------------------------------------------

def plot_telemetry_timeseries(csv_path: str, concurrency: int, args: argparse.Namespace) -> None:
    """Generate time-series plots from a single run's telemetry CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not os.path.exists(csv_path):
        return

    samples: list[dict[str, str]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    if len(samples) < 2:
        return

    plot_dir = os.path.dirname(csv_path)
    elapsed = [float(s["elapsed_s"]) for s in samples]

    # Detect GPU indices from CSV headers (handles non-sequential like gpu2, gpu3)
    import re
    gpu_indices = sorted({int(m.group(1)) for k in samples[0] for m in [re.match(r"gpu(\d+)_power_w", k)] if m})

    seq_len_part = f"  |  seq {args.max_seq_len}" if args.max_seq_len is not None else ""
    subtitle = f"{args.model_id}{seq_len_part}  |  {args.input_len}in/{args.output_len}out  |  c={concurrency}  |  W{args.watt}"

    # Plot 1: Power draw over time
    fig, ax = plt.subplots(figsize=(10, 5))
    total_power = []
    for s in samples:
        total = sum(float(s.get(f"gpu{g}_power_w") or 0) for g in gpu_indices)
        total_power.append(total)
    ax.plot(elapsed, total_power, "k-", linewidth=2, label="Total")
    for g in gpu_indices:
        vals = [float(s.get(f"gpu{g}_power_w") or 0) for s in samples]
        ax.plot(elapsed, vals, "--", linewidth=1, alpha=0.6, label=f"GPU {g}")
    ax.set_title(f"Power Draw Over Time\n{subtitle}")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Power (W)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plot_dir, "telemetry_power.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: KV cache + requests over time (dual y-axis)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    kv = [float(s.get("kv_cache_pct") or 0) for s in samples]
    ax1.plot(elapsed, kv, "b-", linewidth=2, label="KV Cache %")
    ax1.set_ylabel("KV Cache Usage (%)", color="b")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    running = [float(s.get("requests_running") or 0) for s in samples]
    waiting = [float(s.get("requests_waiting") or 0) for s in samples]
    ax2.step(elapsed, running, where="post", color="g", linewidth=1.2, alpha=0.7, label="Running")
    ax2.step(elapsed, waiting, where="post", color="r", linewidth=1.2, alpha=0.7, label="Waiting")
    ax2.set_ylabel("Request Count")
    max_req = max(max(running, default=0), max(waiting, default=0), 1)
    ax2.set_ylim(0, max_req * 1.3)
    ax1.set_title(f"KV Cache & Requests Over Time\n{subtitle}")
    ax1.set_xlabel("Elapsed (s)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plot_dir, "telemetry_kv_cache.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  Telemetry plots saved to: {plot_dir}/")


def plot_telemetry_compare(
    series: dict[str, list[BenchResult]],
    args: argparse.Namespace,
) -> None:
    """Telemetry overlay charts: power and KV cache vs concurrency across series."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return

    # Only plot if any series has telemetry data
    has_telemetry = any(
        any(r.mean_power_w > 0 or r.max_kv_cache_pct > 0 for r in res)
        for res in series.values()
    )
    if not has_telemetry:
        return

    # Apply concurrency cutoff
    if args.max_plot_concurrency is not None:
        series = {
            label: [r for r in res if r.concurrency <= args.max_plot_concurrency]
            for label, res in series.items()
        }
        series = {label: res for label, res in series.items() if res}
    if not series:
        return

    safe_model = args.model_id.replace("/", "__")
    plot_dir = os.path.join(args.result_dir, "plots", f"{safe_model}_compare_W{args.watt}")
    os.makedirs(plot_dir, exist_ok=True)

    seq_len_part = f"  |  seq {args.max_seq_len}" if args.max_seq_len is not None else ""
    suptitle = f"{args.model_id}{seq_len_part}  |  W{args.watt}"

    cmap = matplotlib.colormaps["tab10"]
    labels = sorted(series.keys())
    colors = {label: cmap(i) for i, label in enumerate(labels)}

    def _save(fig: Any, name: str) -> None:
        path = os.path.join(plot_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    def _style_ax(ax: Any, xlabel: str = "Concurrency", ylabel: str = "") -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=8)

    # --- Power vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.mean_power_w for r in res], "o-", label=f"{label} (mean)",
                color=colors[label], linewidth=2)
        ax.plot(concs, [r.max_power_w for r in res], "^--", label=f"{label} (peak)",
                color=colors[label], linewidth=1, alpha=0.5)
    ax.set_title(f"System Power Draw vs Concurrency\n{suptitle}")
    _style_ax(ax, ylabel="Power (W)")
    _save(fig, "compare_power_vs_concurrency")

    # --- KV Cache vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.mean_kv_cache_pct for r in res], "o-", label=f"{label} (mean)",
                color=colors[label], linewidth=2)
        ax.plot(concs, [r.max_kv_cache_pct for r in res], "^--", label=f"{label} (peak)",
                color=colors[label], linewidth=1, alpha=0.5)
    ax.set_title(f"KV Cache Usage vs Concurrency\n{suptitle}")
    ax.set_ylim(0, 105)
    _style_ax(ax, ylabel="KV Cache Usage (%)")
    _save(fig, "compare_kv_cache_vs_concurrency")

    # --- GPU Utilization vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.mean_gpu_util_pct for r in res], "o-", label=label,
                color=colors[label], linewidth=2)
    ax.set_title(f"GPU Utilization vs Concurrency\n{suptitle}")
    _style_ax(ax, ylabel="GPU Utilization (%)")
    _save(fig, "compare_gpu_util_vs_concurrency")

    # --- GPU Memory Bandwidth Utilization vs Concurrency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        ax.plot(concs, [r.mean_gpu_mem_bw_util_pct for r in res], "o-", label=f"{label} (mean)",
                color=colors[label], linewidth=2)
        ax.plot(concs, [r.max_gpu_mem_bw_util_pct for r in res], "^--", label=f"{label} (peak)",
                color=colors[label], linewidth=1, alpha=0.5)
    ax.set_title(f"Memory Bandwidth Utilization vs Concurrency\n{suptitle}")
    _style_ax(ax, ylabel="Memory BW Utilization (%)")
    _save(fig, "compare_mem_bw_util_vs_concurrency")

    # --- Power Efficiency: tok/s per Watt ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in labels:
        res = series[label]
        concs = [r.concurrency for r in res]
        efficiency = [r.output_throughput / r.mean_power_w if r.mean_power_w > 0 else 0 for r in res]
        ax.plot(concs, efficiency, "o-", label=label, color=colors[label], linewidth=2)
    ax.set_title(f"Power Efficiency vs Concurrency\n{suptitle}")
    _style_ax(ax, ylabel="Output tok/s/W")
    _save(fig, "compare_efficiency_vs_concurrency")

    print(f"\n  Telemetry comparison plots saved to: {plot_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM benchmark sweep harness — sweep concurrency, detect saturation, plot results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--model-id", required=True, help="Model ID passed to vllm bench (also used for result dir naming)")
    p.add_argument("--tokenizer", default=None, help="Path to tokenizer (defaults to /mnt/hot/ambientlight/models/{model-id})")
    p.add_argument("--watt", type=int, required=True, help="Wattage label for result dir naming (e.g. 250)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server base URL")
    p.add_argument("--bench-cmd", default=None,
                   help="Override the bench launcher (shell string). Default: "
                        "'python -m vllm.entrypoints.cli.main bench serve'. Works against any "
                        "OpenAI-compatible endpoint (vLLM or sglang).")
    p.add_argument("--input-len", type=int, default=2048, help="Random input token length")
    p.add_argument("--output-len", type=int, default=1024, help="Random output token length")
    p.add_argument("--num-prompts", type=int, default=512, help="Number of prompts per benchmark run")

    p.add_argument("--max-concurrency", type=str, default="128",
                   help="Hard ceiling for concurrency sweep. Either a single integer (applied to all input lengths) "
                        "or a comma-separated list corresponding 1:1 with --input-lens "
                        "(e.g. --input-lens 2048,7167,15359 --max-concurrency 128,64,32). Default: 128")
    p.add_argument("--min-concurrency", type=int, default=1, help="Starting concurrency level")
    p.add_argument("--step-size", type=int, default=8, help="Linear step size after initial geometric ramp (1,2,4,8)")

    p.add_argument("--max-seq-len", type=int, default=None, help="vLLM max_model_len / max sequence length (displayed in chart subtitles)")

    p.add_argument("--result-dir", default="./bench", help="Base directory for results")
    p.add_argument("--plot-only", action="store_true", help="Skip benchmarks, just aggregate existing results and plot")
    p.add_argument("--compare", action="store_true",
                   help="Compare mode: overlay multiple sweeps on the same chart. "
                        "Use --input-lens and/or --output-lens to specify which sweeps to compare.")
    p.add_argument("--input-lens", type=str, default=None,
                   help="Comma-separated list of input lengths to compare (e.g. '2048,7167'). "
                        "Used with --compare and --plot-only.")
    p.add_argument("--output-lens", type=str, default=None,
                   help="Comma-separated list of output lengths to compare (e.g. '512,1024'). "
                        "Used with --compare.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    p.add_argument("--max-plot-concurrency", type=int, default=None,
                   help="Right-side cutoff for charts: only plot data points with concurrency <= this value")
    p.add_argument("--reuse", action="store_true", default=False, help="Skip concurrency levels that already have results in result-dir")
    p.add_argument("--no-early-stop", action="store_true", default=False, help="Disable early stopping — sweep all concurrency levels regardless of duration")
    p.add_argument("--early-stop-extra", type=int, default=2,
                   help="Number of extra data points to collect past the best duration before stopping (default: 2)")
    p.add_argument("--max-error-rate", type=float, default=0.0,
                   help="Maximum allowed error rate (0.0–1.0). Results with error rate above this are discarded. "
                        "Default 0.0 = any failed request disqualifies the result.")

    # Matrix sweep mode
    p.add_argument("--matrix", action="store_true",
                   help="Matrix sweep mode: run full concurrency sweeps for all (input_len, output_len) "
                        "combinations from --input-lens and --output-lens against the same running server. "
                        "Generates per-combination plots and automatic comparison overlays.")

    # Telemetry
    p.add_argument("--telemetry", action="store_true", default=False,
                   help="Enable GPU power/utilization and vLLM KV cache telemetry during benchmark runs")
    p.add_argument("--no-telemetry", action="store_true", default=False,
                   help="Explicitly disable telemetry (overrides --telemetry)")
    p.add_argument("--telemetry-interval", type=float, default=0.25,
                   help="Telemetry sampling interval in seconds (default: 0.25)")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated GPU indices to monitor for telemetry (e.g. '2,3'). Default: all GPUs.")

    args = p.parse_args()

    if args.tokenizer is None:
        args.tokenizer = f"/mnt/hot/ambientlight/models/{args.model_id}"

    # Parse --gpu-ids
    if args.gpu_ids is not None:
        args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    # Resolve telemetry: --no-telemetry wins over --telemetry
    if args.no_telemetry:
        args.telemetry = False

    # Parse --max-concurrency: single int or comma-separated list
    max_conc_parts = [int(x.strip()) for x in args.max_concurrency.split(",")]
    if len(max_conc_parts) == 1:
        # Single value — used for all input lengths
        args.max_concurrency = max_conc_parts[0]
        args.max_concurrency_map = None  # no per-input-len mapping
    else:
        # Multiple values — must be in --matrix mode and match --input-lens count
        input_lens = [int(x) for x in args.input_lens.split(",")] if args.input_lens else [args.input_len]
        if len(max_conc_parts) != len(input_lens):
            p.error(f"--max-concurrency has {len(max_conc_parts)} values but --input-lens has "
                    f"{len(input_lens)} values — they must match 1:1 "
                    f"(or provide a single --max-concurrency for all)")
        args.max_concurrency = max(max_conc_parts)  # set to max for display; actual cap per-input-len
        args.max_concurrency_map = dict(zip(input_lens, max_conc_parts))

    # Validate --matrix usage
    if args.matrix and not args.input_lens and not args.output_lens:
        p.error("--matrix requires at least --input-lens or --output-lens to be specified")

    if args.matrix and args.compare:
        p.error("--matrix and --compare are mutually exclusive. "
                "--matrix auto-generates comparison plots after sweeping.")

    return args


# ---------------------------------------------------------------------------
# Matrix sweep — sweep all (input_len, output_len) combinations
# ---------------------------------------------------------------------------

def run_matrix_sweep(
    args: argparse.Namespace,
) -> dict[tuple[int, int], list[BenchResult]]:
    """Run full concurrency sweeps for all (input_len, output_len) combinations.

    Returns dict keyed by (input_len, output_len) -> list of BenchResult.
    """
    input_lens = (
        [int(x) for x in args.input_lens.split(",")]
        if args.input_lens
        else [args.input_len]
    )
    output_lens = (
        [int(x) for x in args.output_lens.split(",")]
        if args.output_lens
        else [args.output_len]
    )

    combos = [(il, ol) for il in input_lens for ol in output_lens]
    total = len(combos)

    conc_map = getattr(args, "max_concurrency_map", None)

    print("=== MATRIX SWEEP MODE ===")
    print(f"Input lengths:  {input_lens}")
    print(f"Output lengths: {output_lens}")
    if conc_map:
        print("Max concurrency per input length:")
        for il in input_lens:
            print(f"  {il:>6d} tokens -> max_concurrency={conc_map[il]}")
    else:
        print(f"Max concurrency: {args.max_concurrency} (all input lengths)")
    print(f"Combinations:   {total}")
    print()

    all_results: dict[tuple[int, int], list[BenchResult]] = {}

    for idx, (il, ol) in enumerate(combos, 1):
        mc = conc_map[il] if conc_map else args.max_concurrency
        print(f"\n{'#'*70}")
        print(f"  Matrix sweep [{idx}/{total}]: input_len={il}, output_len={ol}, max_concurrency={mc}")
        print(f"{'#'*70}")

        # Temporarily override args for this combination
        orig_input_len = args.input_len
        orig_output_len = args.output_len
        orig_max_concurrency = args.max_concurrency
        args.input_len = il
        args.output_len = ol
        args.max_concurrency = mc

        try:
            if args.plot_only:
                results = collect_existing_results(args)
            else:
                results = run_sweep(args)
        finally:
            args.input_len = orig_input_len
            args.output_len = orig_output_len
            args.max_concurrency = orig_max_concurrency

        if results:
            all_results[(il, ol)] = results
            print(f"\n  [{il}in/{ol}out] Completed: {len(results)} data points")

            # Immediate per-combo summary + plots (so results are visible
            # even if a later combo OOMs / crashes)
            print(f"\n{'='*70}")
            print(f"  Summary: {il}in/{ol}out")
            print(f"{'='*70}")
            print_summary(results)
            if not getattr(args, "no_plot", False):
                # Temporarily set args for plot_results (needs input_len/output_len)
                args.input_len = il
                args.output_len = ol
                plot_results(results, args)
                args.input_len = orig_input_len
                args.output_len = orig_output_len
        else:
            print(f"\n  [{il}in/{ol}out] No results")

    return all_results


def main() -> None:
    args = parse_args()

    print(f"Model:       {args.model_id}")
    print(f"Tokenizer:   {args.tokenizer}")
    print(f"Watt:        {args.watt}")
    if args.max_seq_len is not None:
        print(f"Max seq len: {args.max_seq_len}")
    print(f"Input/Output: {args.input_len}/{args.output_len}")
    print(f"Num prompts: {args.num_prompts}")
    conc_map = getattr(args, "max_concurrency_map", None)
    if conc_map:
        print(f"Concurrency: {args.min_concurrency} -> per-input-len caps (step {args.step_size})")
        for il, mc in sorted(conc_map.items()):
            print(f"  input_len={il}: max_concurrency={mc}")
    else:
        print(f"Concurrency: {args.min_concurrency} -> {args.max_concurrency} (step {args.step_size})")
    if args.reuse:
        print("Reuse:       ON (skip concurrency levels with existing results)")
    if args.no_early_stop:
        print("Early stop:  DISABLED (sweep all concurrency levels)")
    if args.telemetry:
        print(f"Telemetry:   ON (interval {args.telemetry_interval}s)")
    print(f"Result dir:  {args.result_dir}")
    print()

    if args.matrix:
        # --- Matrix sweep mode ---
        all_results = run_matrix_sweep(args)

        if not all_results:
            print("\nNo results from matrix sweep. Nothing to summarize or plot.")
            sys.exit(1)

        # Auto-generate comparison overlay (per-combo plots already done inside run_matrix_sweep)
        if len(all_results) > 1 and not args.no_plot:
            print(f"\n{'='*70}")
            print("  Generating comparison overlay plots...")
            print(f"{'='*70}")
            series = {
                f"{il}in/{ol}out": results
                for (il, ol), results in sorted(all_results.items())
            }
            plot_compare(series, args)
            plot_telemetry_compare(series, args)
        return

    elif args.plot_only:
        print("=== PLOT-ONLY MODE ===")
        print("Scanning for existing results...\n")
        results = collect_existing_results(args)
    elif args.compare:
        # --- Compare mode: collect multiple sweeps and overlay ---
        input_lens = [int(x) for x in args.input_lens.split(",")] if args.input_lens else [args.input_len]
        output_lens = [int(x) for x in args.output_lens.split(",")] if args.output_lens else [args.output_len]

        print("=== COMPARE MODE ===")
        series: dict[str, list[BenchResult]] = {}
        for il in input_lens:
            for ol in output_lens:
                label = f"{il}in/{ol}out"
                print(f"\nCollecting: {label}")
                res = collect_existing_results(args, input_len=il, output_len=ol)
                if res:
                    series[label] = res
                    print(f"  {len(res)} data points")
                else:
                    print("  No valid results found")

        if not series:
            print("\nNo results found for any sweep. Nothing to plot.")
            sys.exit(1)

        # Print per-series summaries
        for label, res in sorted(series.items()):
            best = max(res, key=lambda r: r.output_throughput)
            print(f"\n  [{label}] Best: {best.output_throughput:.1f} tok/s at c={best.concurrency}")

        print("\nGenerating comparison plots...")
        plot_compare(series, args)
        plot_telemetry_compare(series, args)
        return
    elif args.dry_run:
        print("=== DRY RUN MODE ===\n")
        run_sweep(args)
        return
    else:
        results = run_sweep(args)

    if not results:
        print("\nNo results found. Nothing to summarize or plot.")
        sys.exit(1)

    print_summary(results)

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(results, args)


if __name__ == "__main__":
    _t0 = time.time()
    try:
        main()
    finally:
        _elapsed = time.time() - _t0
        _h, _rem = divmod(_elapsed, 3600)
        _m, _s = divmod(_rem, 60)
        print(f"\nTotal runtime: {_elapsed:.1f}s ({int(_h)}h {int(_m)}m {_s:.0f}s)")
