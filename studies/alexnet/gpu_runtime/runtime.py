from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

import torch

from compile_trt import BUCKETS, ShapeBucket, compile_all


@dataclass
class ImageRequest:
    req_id: int
    tensor: torch.Tensor


@dataclass
class RequestTable:
    requests: list[ImageRequest] = field(default_factory=list)

    def add(self, request: ImageRequest) -> None:
        self.requests.append(request)


@dataclass
class BatchAssignment:
    bucket: ShapeBucket
    requests: list[ImageRequest]


@dataclass
class InferenceResult:
    req_id: int
    bucket_name: str
    output: torch.Tensor
    stream_name: str


def normalize_request_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected CHW or NCHW tensor, got {tuple(x.shape)}")
    return x.contiguous()


class ShapeBucketSelector:
    def __init__(self, buckets: Iterable[ShapeBucket]):
        self.buckets = tuple(buckets)

    def select(self, n: int, c: int, h: int, w: int) -> ShapeBucket:
        for bucket in self.buckets:
            max_n, max_c, max_h, max_w = bucket.max_shape
            if c == max_c and n <= max_n and h <= max_h and w <= max_w:
                return bucket
        raise ValueError(f"no bucket can serve shape {(n, c, h, w)}")


class TensorRTRuntime:
    def __init__(
        self,
        device: str = "cuda",
        engine_dir: str = "./engine_cache",
        lookahead_window: int = 8,
    ):
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.selector = ShapeBucketSelector(BUCKETS)
        self.compiled = compile_all(out_dir=engine_dir, device=self.device)
        self.lookahead_window = max(1, lookahead_window)
        self.streams = {
            bucket.name: (torch.cuda.Stream() if self.device == "cuda" else None)
            for bucket in BUCKETS
        }

    def bucket_for_request(self, request: ImageRequest) -> ShapeBucket:
        x = normalize_request_tensor(request.tensor)
        n, c, h, w = x.shape
        return self.selector.select(n, c, h, w)

    def build_assignments(self, table: RequestTable) -> list[BatchAssignment]:
        bucket_map = {bucket.name: bucket for bucket in BUCKETS}
        assignments: list[BatchAssignment] = []

        pending: deque[tuple[ImageRequest, ShapeBucket]] = deque()
        cursor = 0
        requests = table.requests

        def refill_window() -> None:
            nonlocal cursor
            while len(pending) < self.lookahead_window and cursor < len(requests):
                req = requests[cursor]
                pending.append((req, self.bucket_for_request(req)))
                cursor += 1

        def choose_bucket_name() -> str:
            counts = Counter(bucket.name for _, bucket in pending)
            return max(
                counts,
                key=lambda name: (
                    counts[name],
                    -next(i for i, bucket in enumerate(BUCKETS) if bucket.name == name),
                ),
            )

        refill_window()
        while pending:
            chosen_name = choose_bucket_name()
            chosen_bucket = bucket_map[chosen_name]
            max_batch = chosen_bucket.max_shape[0]

            batch_reqs: list[ImageRequest] = []
            survivors: deque[tuple[ImageRequest, ShapeBucket]] = deque()

            while pending:
                req, bucket = pending.popleft()
                if bucket.name == chosen_name and len(batch_reqs) < max_batch:
                    batch_reqs.append(req)
                else:
                    survivors.append((req, bucket))

            pending = survivors
            assignments.append(BatchAssignment(bucket=chosen_bucket, requests=batch_reqs))
            refill_window()

        return assignments

    @torch.inference_mode()
    def execute(self, table: RequestTable) -> list[InferenceResult]:
        assignments = self.build_assignments(table)
        scheduled: list[tuple[BatchAssignment, torch.Tensor, str]] = []

        for assignment in assignments:
            xs = [normalize_request_tensor(req.tensor) for req in assignment.requests]
            batch = torch.cat(xs, dim=0).to(self.device, dtype=torch.float32, non_blocking=self.device == "cuda")
            module = self.compiled[assignment.bucket.name]

            stream_name = f"stream_{assignment.bucket.name}"
            stream = self.streams[assignment.bucket.name]
            if self.device == "cuda" and stream is not None:
                with torch.cuda.stream(stream):
                    out = module(batch)
            else:
                out = module(batch)
            scheduled.append((assignment, out, stream_name))

        if self.device == "cuda":
            for stream in self.streams.values():
                if stream is not None:
                    stream.synchronize()

        results: list[InferenceResult] = []
        for assignment, out, stream_name in scheduled:
            for idx, request in enumerate(assignment.requests):
                results.append(
                    InferenceResult(
                        req_id=request.req_id,
                        bucket_name=assignment.bucket.name,
                        output=out[idx : idx + 1],
                        stream_name=stream_name,
                    )
                )
        results.sort(key=lambda item: item.req_id)
        return results
