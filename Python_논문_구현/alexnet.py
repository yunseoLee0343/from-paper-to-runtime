import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# AlexNet 2012: full practical training pipeline
# ------------------------------------------------------------
# 포함 기능
# 1) AlexNet 2012 single-device / split-GPU 모델
# 2) PCA lighting augmentation
# 3) ImageFolder 기반 ImageNet-subset 학습
# 4) SGD + momentum + weight decay
# 5) batch-size scaling helper
# 6) layer-wise activation memory profiling
# 7) async prefetch loader
# 8) benchmark (params / flops / latency)
# 9) AMP / channels_last / torch.compile
# 10) gradient clipping / grad accumulation
# 11) label smoothing / EMA
# 12) early stopping / auto resume
# 13) CSV + JSON logging
# 14) confusion matrix export
# 15) ONNX export
#
# 기대 디렉터리 구조
# data_root/
#   train/
#     class_a/xxx.jpeg
#     class_b/yyy.jpeg
#   val/
#     class_a/zzz.jpeg
#     class_b/www.jpeg
#
# 예시
# python train_alexnet_full.py \
#   --data-root /path/to/imagenet_subset \
#   --epochs 30 \
#   --batch-size 128 \
#   --workers 8 \
#   --save-dir ./runs/alexnet_full
# ============================================================


# ============================================================
# 0. Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[torch.Tensor]:
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv_row(path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 1. PCA Lighting Augmentation
# ============================================================
class AlexNetLighting:
    def __init__(
        self,
        alphastd: float = 0.1,
        eigval: Optional[torch.Tensor] = None,
        eigvec: Optional[torch.Tensor] = None,
    ):
        self.alphastd = alphastd
        self.eigval = eigval if eigval is not None else torch.tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = eigvec if eigvec is not None else torch.tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.alphastd == 0:
            return img
        if not isinstance(img, torch.Tensor):
            raise TypeError("AlexNetLighting expects a torch.Tensor image after ToTensor().")
        alpha = torch.normal(mean=torch.zeros(3), std=self.alphastd * torch.ones(3))
        rgb = (self.eigvec * (alpha * self.eigval).view(1, 3)).sum(dim=1)
        return img + rgb.view(3, 1, 1)


# ============================================================
# 2. Memory profiling
# ============================================================
class MemoryProfiler:
    def __init__(self):
        self.records: List[Tuple[str, int, Tuple[int, ...]]] = []

    def register(self, name: str):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                size_bytes = out.numel() * out.element_size()
                shape = tuple(out.shape)
            elif isinstance(out, (list, tuple)):
                tensors = [o for o in out if isinstance(o, torch.Tensor)]
                size_bytes = sum(o.numel() * o.element_size() for o in tensors)
                shape = tuple(tensors[0].shape) if tensors else ()
            else:
                size_bytes = 0
                shape = ()
            self.records.append((name, size_bytes, shape))
        return hook

    def clear(self) -> None:
        self.records.clear()

    def summary(self) -> None:
        total = 0
        print("\n=== Activation Memory Profile ===")
        for name, size_bytes, shape in self.records:
            print(f"{name:30s} shape={str(shape):20s} out={size_bytes/1024/1024:8.2f} MB")
            total += size_bytes
        print(f"{'TOTAL':30s} {'':20s} out={total/1024/1024:8.2f} MB\n")


def attach_memory_hooks(model: nn.Module) -> MemoryProfiler:
    profiler = MemoryProfiler()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.LocalResponseNorm)):
            module.register_forward_hook(profiler.register(name))
    return profiler


# ============================================================
# 3. Models
# ============================================================
class AlexNet2012(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        conv_layers: List[nn.Conv2d] = []
        linear_layers: List[nn.Linear] = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
                conv_layers.append(m)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
                linear_layers.append(m)

        if len(conv_layers) >= 5:
            nn.init.constant_(conv_layers[1].bias, 1.0)
            nn.init.constant_(conv_layers[3].bias, 1.0)
            nn.init.constant_(conv_layers[4].bias, 1.0)

        if len(linear_layers) >= 2:
            nn.init.constant_(linear_layers[0].bias, 1.0)
            nn.init.constant_(linear_layers[1].bias, 1.0)


class SplitAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_p: float = 0.5):
        super().__init__()
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SplitAlexNet requires at least 2 CUDA devices.")

        self.dev0 = torch.device("cuda:0")
        self.dev1 = torch.device("cuda:1")

        self.features_0 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev0)

        self.features_1 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        ).to(self.dev1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dev0, non_blocking=True)
        x = self.features_0(x)
        x = x.to(self.dev1, non_blocking=True)
        x = self.features_1(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @property
    def output_device(self) -> torch.device:
        return self.dev1

    def _initialize_weights(self) -> None:
        conv_layers: List[nn.Conv2d] = []
        linear_layers: List[nn.Linear] = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
                conv_layers.append(m)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
                linear_layers.append(m)

        if len(conv_layers) >= 5:
            nn.init.constant_(conv_layers[1].bias, 1.0)
            nn.init.constant_(conv_layers[3].bias, 1.0)
            nn.init.constant_(conv_layers[4].bias, 1.0)

        if len(linear_layers) >= 2:
            nn.init.constant_(linear_layers[0].bias, 1.0)
            nn.init.constant_(linear_layers[1].bias, 1.0)


# ============================================================
# 4. EMA
# ============================================================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.state_dict().items():
            if torch.is_tensor(param):
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for name in self.shadow:
            self.backup[name] = state[name].detach().clone()
            state[name].copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        state = model.state_dict()
        for name in self.backup:
            state[name].copy_(self.backup[name])
        self.backup = {}


# ============================================================
# 5. Batch-size scaling helper
# ============================================================
@dataclass
class BatchScalingResult:
    new_lr: float
    new_weight_decay: float


def scale_hparams_for_batch(
    base_lr: float,
    base_weight_decay: float,
    old_batch_size: int,
    new_batch_size: int,
    rule: str = "paper_practice",
) -> BatchScalingResult:
    if old_batch_size <= 0 or new_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

    k = new_batch_size / old_batch_size

    if rule == "paper_theory":
        new_lr = math.sqrt(k) * base_lr
        new_wd = (1.0 / new_lr) * (1.0 - (1.0 - base_lr * base_weight_decay) ** k)
    elif rule == "paper_practice":
        new_lr = k * base_lr
        new_wd = base_weight_decay
    else:
        raise ValueError("rule must be either 'paper_theory' or 'paper_practice'.")

    return BatchScalingResult(new_lr=new_lr, new_weight_decay=new_wd)


# ============================================================
# 6. Data transforms
# ============================================================
def build_transforms(train: bool, resize_size: int = 256, crop_size: int = 224, disable_lighting: bool = False):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_ops = [
        transforms.Resize(resize_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if not disable_lighting:
        train_ops.append(AlexNetLighting(alphastd=0.1))
    train_ops.append(transforms.Normalize(imagenet_mean, imagenet_std))

    if train:
        return transforms.Compose(train_ops)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


# ============================================================
# 7. Dataset subset helpers
# ============================================================
def filter_dataset_by_classes(
    dataset: datasets.ImageFolder,
    keep_classes: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Subset:
    indices = list(range(len(dataset)))

    if keep_classes:
        class_to_idx = dataset.class_to_idx
        missing = [c for c in keep_classes if c not in class_to_idx]
        if missing:
            raise ValueError(f"Requested classes not found in dataset: {missing}")
        keep_idx = {class_to_idx[c] for c in keep_classes}
        indices = [i for i in indices if dataset.targets[i] in keep_idx]

    if max_samples is not None and max_samples < len(indices):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:max_samples]

    return Subset(dataset, indices)


def build_dataloaders(
    data_root: str,
    batch_size: int,
    workers: int,
    subset_classes: Optional[List[str]],
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    seed: int,
    resize_size: int,
    crop_size: int,
    disable_lighting: bool,
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    train_full = datasets.ImageFolder(
        str(train_dir),
        transform=build_transforms(train=True, resize_size=resize_size, crop_size=crop_size, disable_lighting=disable_lighting),
    )
    val_full = datasets.ImageFolder(
        str(val_dir),
        transform=build_transforms(train=False, resize_size=resize_size, crop_size=crop_size, disable_lighting=disable_lighting),
    )

    if train_full.classes != val_full.classes:
        raise ValueError("Train/val class lists differ. Check folder structure.")

    train_subset = filter_dataset_by_classes(train_full, subset_classes, max_train_samples, seed)
    val_subset = filter_dataset_by_classes(val_full, subset_classes, max_val_samples, seed)

    classes = subset_classes if subset_classes else train_full.classes
    num_classes = len(classes)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, num_classes, classes


# ============================================================
# 8. Prefetch loader
# ============================================================
class PrefetchLoader:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.enabled = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return

        first = True
        for batch_input, batch_target in self.loader:
            with torch.cuda.stream(self.stream):
                next_input = batch_input.to(self.device, non_blocking=True)
                next_target = batch_target.to(self.device, non_blocking=True)

            if not first:
                yield current_input, current_target
            else:
                first = False

            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            current_input = next_input
            current_target = next_target

        if not first:
            yield current_input, current_target


# ============================================================
# 9. Optimizer / scheduler / loss
# ============================================================
def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: optim.Optimizer, scheduler_name: str, epochs: int, min_lr: float = 1e-6):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "multistep":
        milestones = [max(1, epochs // 2), max(1, (epochs * 3) // 4)]
        milestones = sorted(list(set(milestones)))
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_criterion(label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# ============================================================
# 10. Train / validate helpers
# ============================================================
def get_primary_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    if isinstance(model, SplitAlexNet):
        return model.dev0
    return fallback


def get_output_device(model: nn.Module, fallback: torch.device) -> torch.device:
    if isinstance(model, SplitAlexNet):
        return model.output_device
    return fallback


def maybe_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def clip_gradients(model: nn.Module, max_norm: float) -> Optional[float]:
    if max_norm <= 0:
        return None
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def export_confusion_matrix(
    model: nn.Module,
    loader,
    default_device: torch.device,
    num_classes: int,
    save_path: str,
    channels_last: bool = False,
    amp_enabled: bool = False,
) -> None:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    output_device = get_output_device(model, default_device)

    with torch.no_grad():
        for images, target in loader:
            if not isinstance(model, SplitAlexNet):
                images = images.to(default_device, non_blocking=True)
            images = maybe_channels_last(images, channels_last)
            target = target.to(output_device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(images)

            pred = output.argmax(dim=1)
            target_np = target.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            for t, p in zip(target_np, pred_np):
                cm[t, p] += 1

    np.save(save_path, cm)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    default_device: torch.device,
    epoch: int,
    print_freq: int = 50,
    amp_enabled: bool = False,
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    grad_clip_norm: float = 0.0,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    model.train()

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("loss")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    grad_norm_meter = AverageMeter("grad_norm")

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_device = get_output_device(model, default_device)
    grad_accum_steps = max(1, grad_accum_steps)

    end = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, (images, target) in enumerate(loader):
        data_time.update(time.time() - end)

        if not isinstance(model, SplitAlexNet):
            images = images.to(default_device, non_blocking=True)
        images = maybe_channels_last(images, channels_last)
        target = target.to(output_device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
            loss = criterion(output, target)
            loss_for_backward = loss / grad_accum_steps

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        scaler.scale(loss_for_backward).backward()

        do_step = ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(loader))
        grad_norm_value = None

        if do_step:
            if amp_enabled:
                scaler.unscale_(optimizer)

            grad_norm_value = clip_gradients(model, grad_clip_norm)
            if grad_norm_value is not None:
                grad_norm_meter.update(grad_norm_value)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0:
            grad_norm_str = f"grad_norm {grad_norm_meter.val:.3f} ({grad_norm_meter.avg:.3f}) " if grad_norm_meter.count > 0 else ""
            print(
                f"[Train] Epoch {epoch:03d} Step {step:04d}/{len(loader):04d} "
                f"data {data_time.val:.3f}s ({data_time.avg:.3f}s) "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                f"loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"top1 {top1.val:.2f} ({top1.avg:.2f}) "
                f"top5 {top5.val:.2f} ({top5.avg:.2f}) "
                f"{grad_norm_str}"
            )

    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
        "grad_norm": grad_norm_meter.avg if grad_norm_meter.count > 0 else 0.0,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    default_device: torch.device,
    epoch: int,
    print_freq: int = 50,
    amp_enabled: bool = False,
    channels_last: bool = False,
) -> Dict[str, float]:
    model.eval()

    batch_time = AverageMeter("batch_time")
    losses = AverageMeter("loss")
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")

    output_device = get_output_device(model, default_device)
    end = time.time()

    for step, (images, target) in enumerate(loader):
        if not isinstance(model, SplitAlexNet):
            images = images.to(default_device, non_blocking=True)
        images = maybe_channels_last(images, channels_last)
        target = target.to(output_device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0:
            print(
                f"[Val]   Epoch {epoch:03d} Step {step:04d}/{len(loader):04d} "
                f"batch {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                f"loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"top1 {top1.val:.2f} ({top1.avg:.2f}) "
                f"top5 {top5.val:.2f} ({top5.avg:.2f})"
            )

    return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}


# ============================================================
# 11. Checkpoint / resume
# ============================================================
def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    best_top1: float,
    args: argparse.Namespace,
    is_best: bool,
    ema: Optional[ModelEMA] = None,
) -> None:
    ensure_dir(save_dir)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_top1": best_top1,
        "args": vars(args),
        "ema_state": ema.shadow if ema is not None else None,
    }

    last_path = os.path.join(save_dir, "checkpoint_last.pt")
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(save_dir, "checkpoint_best.pt")
        shutil.copyfile(last_path, best_path)


def try_auto_resume(save_dir: str) -> str:
    last_path = os.path.join(save_dir, "checkpoint_last.pt")
    return last_path if os.path.exists(last_path) else ""


# ============================================================
# 12. Benchmark / params / ONNX export
# ============================================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dense_ffn_first_layer_params(input_dim: int = 224 * 224 * 3, hidden_dim: int = 4096) -> int:
    return input_dim * hidden_dim + hidden_dim


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    default_device: torch.device,
    input_size=(1, 3, 224, 224),
    channels_last: bool = False,
) -> None:
    model.eval()

    if isinstance(model, SplitAlexNet):
        dummy_input = torch.randn(input_size, device=model.dev0)
    else:
        dummy_input = torch.randn(input_size, device=default_device)

    dummy_input = maybe_channels_last(dummy_input, channels_last)

    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception:
        flops, params = float("nan"), count_params(model)

    if isinstance(model, SplitAlexNet):
        warmup = 20
        reps = 100
        for _ in range(warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize(model.dev0)
        torch.cuda.synchronize(model.dev1)
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(dummy_input)
        torch.cuda.synchronize(model.dev0)
        torch.cuda.synchronize(model.dev1)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0 / reps
    elif default_device.type == "cuda":
        warmup = 20
        reps = 100
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            _ = model(dummy_input)
        starter.record()
        for _ in range(reps):
            _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize(default_device)
        latency_ms = starter.elapsed_time(ender) / reps
    else:
        warmup = 5
        reps = 20
        for _ in range(warmup):
            _ = model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(dummy_input)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0 / reps

    print("=" * 70)
    print("Model benchmark")
    print(f"Input size                : {input_size}")
    print(f"Parameters                : {params / 1e6:.2f}M")
    if not math.isnan(flops):
        print(f"FLOPs                     : {flops / 1e9:.2f}G")
    else:
        print("FLOPs                     : unavailable for this model path")
    print(f"Latency                   : {latency_ms:.3f} ms")
    print(f"Dense FFN first-layer param comparison: {dense_ffn_first_layer_params():,}")
    print("=" * 70)


def export_onnx(model: nn.Module, default_device: torch.device, out_path: str, channels_last: bool = False) -> None:
    model.eval()
    if isinstance(model, SplitAlexNet):
        print("ONNX export for SplitAlexNet is skipped; export single-device AlexNet2012 instead.")
        return

    dummy = torch.randn(1, 3, 224, 224, device=default_device)
    dummy = maybe_channels_last(dummy, channels_last)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX exported to {out_path}")


# ============================================================
# 13. Argparse
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="AlexNet 2012 full practical training pipeline")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--subset-classes", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--model-type", type=str, default="single", choices=["single", "split"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="multistep", choices=["multistep", "cosine", "none"])
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--batch-scale-rule", type=str, default="paper_practice", choices=["paper_theory", "paper_practice"])
    parser.add_argument("--base-batch-size-for-scaling", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="./runs/alexnet2012_full")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--auto-resume", action="store_true")

    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--early-stop-patience", type=int, default=0)

    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--disable-lighting", action="store_true")

    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-path", type=str, default="alexnet2012.onnx")
    parser.add_argument("--save-confusion-matrix", action="store_true")

    return parser.parse_args()


# ============================================================
# 14. Main
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)

    default_device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using default device: {default_device}")

    subset_classes = [c.strip() for c in args.subset_classes.split(",") if c.strip()] if args.subset_classes else None
    max_train_samples = args.max_train_samples if args.max_train_samples > 0 else None
    max_val_samples = args.max_val_samples if args.max_val_samples > 0 else None

    train_loader_raw, val_loader_raw, num_classes, classes = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_classes=subset_classes,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        seed=args.seed,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        disable_lighting=args.disable_lighting,
    )

    print(f"Num classes : {num_classes}")
    print(f"Train iters : {len(train_loader_raw)}")
    print(f"Val iters   : {len(val_loader_raw)}")

    save_json(os.path.join(args.save_dir, "classes.json"), classes)
    save_json(os.path.join(args.save_dir, "config.json"), vars(args))

    scaling = scale_hparams_for_batch(
        base_lr=args.lr,
        base_weight_decay=args.weight_decay,
        old_batch_size=args.base_batch_size_for_scaling,
        new_batch_size=args.batch_size,
        rule=args.batch_scale_rule,
    )
    print(f"Scaled hyperparams ({args.batch_scale_rule}) -> lr={scaling.new_lr}, wd={scaling.new_weight_decay}")

    if args.model_type == "single":
        model: nn.Module = AlexNet2012(num_classes=num_classes, dropout_p=args.dropout)
        if default_device.type == "cuda":
            model = model.to(default_device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
    else:
        model = SplitAlexNet(num_classes=num_classes, dropout_p=args.dropout)

    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    criterion = build_criterion(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, name=args.optimizer, lr=scaling.new_lr, weight_decay=scaling.new_weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name=args.scheduler, epochs=args.epochs, min_lr=args.min_lr)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    start_epoch = 0
    best_top1 = 0.0
    epochs_without_improve = 0

    resume_path = args.resume
    if not resume_path and args.auto_resume:
        resume_path = try_auto_resume(args.save_dir)

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ema is not None and ckpt.get("ema_state") is not None:
            ema.shadow = ckpt["ema_state"]
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        print(f"Resumed from {resume_path} at epoch {start_epoch}, best_top1={best_top1:.2f}")

    if args.profile_memory:
        profiler = attach_memory_hooks(model)
        if args.model_type == "single":
            probe = torch.randn(1, 3, args.crop_size, args.crop_size, device=default_device)
            probe = maybe_channels_last(probe, args.channels_last)
        else:
            probe = torch.randn(1, 3, args.crop_size, args.crop_size, device=torch.device("cuda:0"))
        with torch.no_grad():
            _ = model(probe)
        profiler.summary()

    if args.prefetch:
        primary_input_device = get_primary_input_device(model, default_device)
        train_loader = PrefetchLoader(train_loader_raw, primary_input_device)
        val_loader = PrefetchLoader(val_loader_raw, primary_input_device)
    else:
        train_loader = train_loader_raw
        val_loader = val_loader_raw

    if args.benchmark_only:
        benchmark_model(model, default_device=default_device, input_size=(1, 3, args.crop_size, args.crop_size), channels_last=args.channels_last)
        print(f"Actual model parameters: {count_params(model):,}")
        return

    history: List[Dict[str, float]] = []
    csv_path = os.path.join(args.save_dir, "history.csv")

    for epoch in range(start_epoch, args.epochs):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch:03d}/{args.epochs - 1:03d}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            default_device=default_device,
            epoch=epoch,
            print_freq=args.print_freq,
            amp_enabled=args.amp,
            channels_last=args.channels_last,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip_norm=args.grad_clip_norm,
            ema=ema,
        )

        if ema is not None:
            ema.apply_shadow(model)

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            default_device=default_device,
            epoch=epoch,
            print_freq=args.print_freq,
            amp_enabled=args.amp,
            channels_last=args.channels_last,
        )

        if ema is not None:
            ema.restore(model)

        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_stats["loss"]),
            "train_top1": float(train_stats["top1"]),
            "train_top5": float(train_stats["top5"]),
            "train_grad_norm": float(train_stats["grad_norm"]),
            "val_loss": float(val_stats["loss"]),
            "val_top1": float(val_stats["top1"]),
            "val_top5": float(val_stats["top5"]),
        }
        history.append(epoch_record)

        save_json(os.path.join(args.save_dir, "history.json"), history)
        append_csv_row(csv_path, epoch_record)

        is_best = val_stats["top1"] > best_top1
        if is_best:
            best_top1 = val_stats["top1"]
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        save_checkpoint(
            save_dir=args.save_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_top1=best_top1,
            args=args,
            is_best=is_best,
            ema=ema,
        )

        print("\n[Epoch Summary]")
        print(
            f"train loss={train_stats['loss']:.4f}, "
            f"train top1={train_stats['top1']:.2f}, "
            f"train top5={train_stats['top5']:.2f}, "
            f"grad_norm={train_stats['grad_norm']:.4f}"
        )
        print(
            f"val   loss={val_stats['loss']:.4f}, "
            f"val   top1={val_stats['top1']:.2f}, "
            f"val   top5={val_stats['top5']:.2f}"
        )
        print(f"best val top1={best_top1:.2f}")

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            print(f"Early stopping triggered after {epochs_without_improve} epochs without improvement.")
            break

    print("\nTraining finished.")

    if ema is not None:
        ema.apply_shadow(model)

    benchmark_model(
        model,
        default_device=default_device,
        input_size=(1, 3, args.crop_size, args.crop_size),
        channels_last=args.channels_last,
    )
    print(f"Actual model parameters: {count_params(model):,}")
    print(f"Dense FFN first-layer params only: {dense_ffn_first_layer_params():,}")

    if args.save_confusion_matrix:
        cm_path = os.path.join(args.save_dir, "confusion_matrix.npy")
        export_confusion_matrix(
            model=model,
            loader=val_loader,
            default_device=default_device,
            num_classes=num_classes,
            save_path=cm_path,
            channels_last=args.channels_last,
            amp_enabled=args.amp,
        )
        print(f"Confusion matrix saved to {cm_path}")

    if args.export_onnx:
        export_onnx(model, default_device=default_device, out_path=os.path.join(args.save_dir, args.onnx_path), channels_last=args.channels_last)

    if ema is not None:
        ema.restore(model)


if __name__ == "__main__":
    main()