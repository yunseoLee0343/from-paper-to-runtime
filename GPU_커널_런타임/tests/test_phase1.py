import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the custom extension test",
)


def _load_extension():
    import myimg_ext  # noqa: F401


def _reference(x, w, b):
    y = F.conv2d(x, w, b, stride=4, padding=2)
    y = F.relu(y)
    y = F.max_pool2d(y, kernel_size=3, stride=2)
    return y


def test_alex_conv1_fused_matches_pytorch():
    _load_extension()
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(8, 3, 224, 224, device=device, dtype=torch.float32)
    w = torch.randn(64, 3, 11, 11, device=device, dtype=torch.float32)
    b = torch.randn(64, device=device, dtype=torch.float32)

    ref = _reference(x, w, b)
    out = torch.ops.myimg.alex_conv1_fused(x, w, b, 4, 4, 2, 2, True, True)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
