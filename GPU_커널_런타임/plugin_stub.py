from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F


def _try_load_extension() -> bool:
    try:
        importlib.import_module("myimg_ext")
        return True
    except Exception:
        return False


@torch.library.custom_op("myimg::alex_conv1_plugin_stub", mutates_args=())
def alex_conv1_plugin_stub(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    if (
        x.is_cuda
        and w.is_cuda
        and b.is_cuda
        and x.dtype == torch.float32
        and w.dtype == torch.float32
        and b.dtype == torch.float32
        and _try_load_extension()
    ):
        return torch.ops.myimg.alex_conv1_fused(
            x, w, b, stride_h, stride_w, pad_h, pad_w, True, True
        )

    y = F.conv2d(x, w, b, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    y = F.relu(y)
    y = F.max_pool2d(y, kernel_size=3, stride=2)
    return y


@alex_conv1_plugin_stub.register_fake
def _fake_alex_conv1_plugin_stub(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    n = x.shape[0]
    k = w.shape[0]
    out_h = (x.shape[2] + 2 * pad_h - w.shape[2]) // stride_h + 1
    out_w = (x.shape[3] + 2 * pad_w - w.shape[3]) // stride_w + 1
    pool_h = (out_h - 3) // 2 + 1
    pool_w = (out_w - 3) // 2 + 1
    return x.new_empty((n, k, pool_h, pool_w))
