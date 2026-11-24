"""Prepare the two proof layouts for an immutable public-replay pool job."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

import numpy as np


@dataclass
class Layout:
    weight: object
    host_noise: object
    device_noise: object = None


class PoolLayouts:
    def __init__(self, pool, op, bt):
        self.pool, self.op, self.bt = pool, op, bt
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pearl-weight")
        self.pending_job = self.future = self.active_job = self.failed_job = None
        self.layouts = {}

    @staticmethod
    def _retire(layouts):
        # Outstanding proof contexts retain their own reference to the Rust weight tree.
        for layout in layouts.values():
            layout.weight.close()
        layouts.clear()

    def _prepare(self, job):
        layouts = {}
        try:
            for compact in (True, False):
                noise = np.empty_like(self.bt)
                weight = self.op.prepare_weight(job.header, self.bt.ctypes.data, noise.ctypes.data,
                                                len(self.bt), compact)
                layouts[compact] = Layout(weight, noise)
            return layouts
        except BaseException:
            self._retire(layouts)
            raise

    def select(self, compact, stream):
        import torch
        job = self.pool.snapshot()
        if self.active_job != job:
            self._retire(self.layouts)
            self.active_job = None
        if self.future is not None and self.future.done():
            try:
                prepared = self.future.result()
            except Exception:
                self.failed_job = self.pending_job
                logging.exception("Pearl replay weight preparation failed")
            else:
                if job == self.pending_job:
                    self.active_job, self.layouts = job, prepared
                else:
                    self._retire(prepared)
            self.pending_job = self.future = None
        if job is None or job == self.failed_job:
            return None
        if self.active_job != job:
            if self.future is None:
                self.pending_job = job
                self.future = self.executor.submit(self._prepare, job)
            return None
        layout = self.layouts[compact]
        if layout.device_noise is None:
            layout.device_noise = torch.from_numpy(layout.host_noise).to(stream.device)
            layout.host_noise = None
        # The torch allocator must retain these raw CUDA inputs across job rotation.
        layout.device_noise.record_stream(stream)
        return job, layout.weight, layout.device_noise

    def close(self):
        self.executor.shutdown(wait=True)
        self._retire(self.layouts)
        try:
            if self.future is not None:
                self._retire(self.future.result())
                self.future = None
        finally:
            self.pool.close()
