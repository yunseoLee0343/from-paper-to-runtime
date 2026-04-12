import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Profile:
    name: str
    batch_size: int
    warmup: int
    iters: int


PROFILES = (
    Profile(name="EDGE_LOW_MEM", batch_size=1, warmup=10, iters=30),
    Profile(name="A6000_HIGH_THROUGHPUT", batch_size=8, warmup=20, iters=50),
)


def load_extension():
    import myimg_ext  # noqa: F401


def pytorch_ref(x, w, b):
    y = F.conv2d(x, w, b, stride=4, padding=2)
    y = F.relu(y)
    y = F.max_pool2d(y, kernel_size=3, stride=2)
    return y


def time_ms(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out


def run_profile(profile: Profile):
    torch.manual_seed(0)
    x = torch.randn(profile.batch_size, 3, 224, 224, device="cuda", dtype=torch.float32)
    w = torch.randn(64, 3, 11, 11, device="cuda", dtype=torch.float32)
    b = torch.randn(64, device="cuda", dtype=torch.float32)

    ref_ms, ref_out = time_ms(pytorch_ref, x, w, b, warmup=profile.warmup, iters=profile.iters)
    opt_ms, opt_out = time_ms(
        torch.ops.myimg.alex_conv1_fused,
        x,
        w,
        b,
        4,
        4,
        2,
        2,
        True,
        True,
        warmup=profile.warmup,
        iters=profile.iters,
    )

    print(
        f"{profile.name}: pytorch={ref_ms:.3f} ms custom={opt_ms:.3f} ms "
        f"speedup={ref_ms / opt_ms:.2f}x shape={tuple(opt_out.shape)}"
    )
    print(f"  correctness={torch.allclose(opt_out, ref_out, rtol=1e-4, atol=1e-4)}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarking")
    load_extension()
    for profile in PROFILES:
        run_profile(profile)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
