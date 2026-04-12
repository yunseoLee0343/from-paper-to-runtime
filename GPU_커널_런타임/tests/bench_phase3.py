import pathlib
import sys
from collections import defaultdict

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from runtime import ImageRequest, RequestTable, TensorRTRuntime  # noqa: E402


def measure_runtime(rt: TensorRTRuntime, requests: list[ImageRequest], warmup: int = 3, iters: int = 10) -> None:
    for _ in range(warmup):
        rt.execute(RequestTable(list(requests)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        results = rt.execute(RequestTable(list(requests)))
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    by_bucket = defaultdict(int)
    for result in results:
        by_bucket[result.bucket_name] += 1

    print(f"avg_batch_ms={avg_ms:.3f}")
    print(f"bucket_counts={dict(sorted(by_bucket.items()))}")
    for result in results:
        print(
            f"req={result.req_id} bucket={result.bucket_name} "
            f"stream={result.stream_name} out_shape={tuple(result.output.shape)}"
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for phase 3 benchmark")

    rt = TensorRTRuntime()
    requests = [
        ImageRequest(0, torch.randn(3, 224, 224, device="cuda", dtype=torch.float32)),
        ImageRequest(1, torch.randn(3, 224, 224, device="cuda", dtype=torch.float32)),
        ImageRequest(2, torch.randn(3, 256, 256, device="cuda", dtype=torch.float32)),
        ImageRequest(3, torch.randn(3, 512, 512, device="cuda", dtype=torch.float32)),
    ]
    measure_runtime(rt, requests)


if __name__ == "__main__":
    main()
