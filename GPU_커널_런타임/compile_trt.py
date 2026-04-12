from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from model import AlexNetImageToImage


@dataclass(frozen=True)
class ShapeBucket:
    name: str
    min_shape: tuple[int, int, int, int]
    opt_shape: tuple[int, int, int, int]
    max_shape: tuple[int, int, int, int]


BUCKETS: tuple[ShapeBucket, ...] = (
    ShapeBucket("224", (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)),
    ShapeBucket("256", (1, 3, 256, 256), (4, 3, 256, 256), (8, 3, 256, 256)),
    ShapeBucket("512", (1, 3, 512, 512), (2, 3, 512, 512), (4, 3, 512, 512)),
)


class EagerFallbackModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.backend = "eager-fallback"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def make_model(device: str = "cuda", use_plugin: bool = True) -> nn.Module:
    model = AlexNetImageToImage(use_plugin=use_plugin).eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    return model


def _save_bucket_manifest(out_dir: Path, bucket: ShapeBucket, backend: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"bucket": asdict(bucket), "backend": backend}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def compile_bucket(
    model: nn.Module,
    bucket: ShapeBucket,
    enabled_precisions: set[torch.dtype],
    workspace_bytes: int,
    engine_cache_dir: Path,
) -> nn.Module:
    try:
        import torch_tensorrt
    except Exception:
        fallback = EagerFallbackModule(model)
        _save_bucket_manifest(engine_cache_dir, bucket, fallback.backend)
        return fallback

    if not torch.cuda.is_available():
        fallback = EagerFallbackModule(model)
        _save_bucket_manifest(engine_cache_dir, bucket, fallback.backend)
        return fallback

    try:
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[
                torch_tensorrt.Input(
                    min_shape=bucket.min_shape,
                    opt_shape=bucket.opt_shape,
                    max_shape=bucket.max_shape,
                    dtype=torch.float32,
                    name="input",
                )
            ],
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_bytes,
            engine_cache_dir=str(engine_cache_dir),
            cache_built_engines=True,
            reuse_cached_engines=True,
            min_block_size=1,
            immutable_weights=True,
            use_python_runtime=False,
        )
        _save_bucket_manifest(engine_cache_dir, bucket, "torch_tensorrt")
        return compiled
    except Exception:
        fallback = EagerFallbackModule(model)
        _save_bucket_manifest(engine_cache_dir, bucket, fallback.backend)
        return fallback


def compile_all(
    out_dir: str = "./engine_cache",
    workspace_bytes: int = 1 << 30,
    enabled_precisions: Iterable[torch.dtype] = (torch.float32,),
    device: str = "cuda",
    use_plugin: bool = True,
) -> dict[str, nn.Module]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(device=device, use_plugin=use_plugin)

    compiled_map: dict[str, nn.Module] = {}
    for bucket in BUCKETS:
        compiled_map[bucket.name] = compile_bucket(
            model=model,
            bucket=bucket,
            enabled_precisions=set(enabled_precisions),
            workspace_bytes=workspace_bytes,
            engine_cache_dir=out / bucket.name,
        )
    return compiled_map


if __name__ == "__main__":
    engines = compile_all()
    summary = {name: getattr(mod, "backend", mod.__class__.__name__) for name, mod in engines.items()}
    print("compiled buckets:", summary)
