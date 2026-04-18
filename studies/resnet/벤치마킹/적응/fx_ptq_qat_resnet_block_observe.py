import copy
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx

torch.manual_seed(0)

# -----------------------------
# Helpers
# -----------------------------
def get_backend_name():
    # Torch 2.4+ often uses "x86". Older setups may prefer "fbgemm".
    for name in ("x86", "fbgemm"):
        try:
            torch.backends.quantized.engine = name
            return name
        except Exception:
            pass
    return "fbgemm"

BACKEND = get_backend_name()

def get_default_qconfig_mapping_safe(backend):
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        return get_default_qconfig_mapping(backend)
    except Exception:
        from torch.ao.quantization import get_default_qconfig
        return QConfigMapping().set_global(get_default_qconfig(backend))

def get_default_qat_qconfig_mapping_safe(backend):
    try:
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        return get_default_qat_qconfig_mapping(backend)
    except Exception:
        from torch.ao.quantization import get_default_qat_qconfig
        return QConfigMapping().set_global(get_default_qat_qconfig(backend))

def tensor_stats(x: torch.Tensor):
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {}
    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99, 0.999]))
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "p999": float(q[3]),
    }

def mse(a: torch.Tensor, b: torch.Tensor):
    return float(torch.mean((a.detach().float() - b.detach().float()) ** 2))

def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    a = a.detach().float().reshape(1, -1)
    b = b.detach().float().reshape(1, -1)
    denom = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    return float((a * b).sum(dim=1) / denom)

def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def dump_named_module_types(model, only_interesting=True):
    rows = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        t = type(mod).__name__
        if only_interesting:
            keys = (
                "Conv", "Linear", "ReLU", "BatchNorm", "Observer",
                "FakeQuant", "Quantize", "DeQuantize", "QFunctional"
            )
            if not any(k in t for k in keys):
                continue
        rows.append((name, t))
    for name, t in rows:
        print(f"{name:50s} {t}")

def dump_graph(model, title):
    print_header(title)
    if hasattr(model, "graph"):
        print(model.graph)
    else:
        print("No FX graph attached.")

def extract_add_region_from_graph(model, title):
    print_header(title)
    if not hasattr(model, "graph"):
        print("No FX graph")
        return
    nodes = list(model.graph.nodes)
    for i, n in enumerate(nodes):
        if "add" in str(n.target).lower() or n.op == "call_function" and "add" in str(n.target).lower():
            lo = max(0, i - 6)
            hi = min(len(nodes), i + 7)
            print(f"[around node {i}: {n.format_node()}]")
            for j in range(lo, hi):
                print(f"  {j:02d}: {nodes[j].format_node()}")
            print("-" * 80)

def find_fake_quant_modules(model):
    out = []
    for name, mod in model.named_modules():
        t = type(mod).__name__.lower()
        if "fakequant" in t:
            out.append((name, mod))
    return out

# -----------------------------
# Instrumented ResNet BasicBlock
# -----------------------------
class InstrumentedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch=8, out_ch=16, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.debug = {}

    def clear_debug(self):
        self.debug = {}

    def forward(self, x):
        self.debug = {}
        identity = self.downsample(x)
        self.debug["skip_out"] = identity

        out = self.conv1(x)
        self.debug["conv1_out"] = out

        out = self.bn1(out)
        self.debug["bn1_out"] = out

        out = self.relu(out)
        self.debug["relu1_out"] = out

        out = self.conv2(out)
        self.debug["conv2_out"] = out

        out = self.bn2(out)
        self.debug["bn2_out"] = out

        self.debug["pre_add_main"] = out
        self.debug["pre_add_skip"] = identity

        add_out = out + identity
        self.debug["post_add"] = add_out

        out = self.relu(add_out)
        self.debug["post_relu"] = out
        return out

# Wrapper so FX sees a clean top-level module
class BlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = InstrumentedBasicBlock()

    def forward(self, x):
        return self.block(x)

# -----------------------------
# Data
# -----------------------------
def make_calib_data(num_batches=16, batch_size=4):
    # Stable random input. You can swap this with real image-like tensors later.
    xs = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 8, 32, 32)
        xs.append(x)
    return xs

# -----------------------------
# Calibration observation
# -----------------------------
@torch.no_grad()
def observe_float_calibration(model, data):
    model.eval()
    acc = defaultdict(list)

    for x in data:
        y = model(x)
        dbg = model.block.debug
        for k in ["conv1_out", "conv2_out", "skip_out", "pre_add_main", "pre_add_skip", "post_add", "post_relu"]:
            acc[k].append(tensor_stats(dbg[k]))

    summary = {}
    for k, rows in acc.items():
        # aggregate mean of summary stats across batches
        keys = rows[0].keys()
        summary[k] = {kk: float(sum(r[kk] for r in rows) / len(rows)) for kk in keys}
    return summary

# -----------------------------
# PTQ flow
# -----------------------------
@torch.no_grad()
def run_ptq_observation(float_model, example_inputs, calib_data):
    print_header(f"PTQ prepare_fx / convert_fx  (backend={BACKEND})")
    qconfig_mapping = get_default_qconfig_mapping_safe(BACKEND)

    prepared = prepare_fx(copy.deepcopy(float_model).eval(), qconfig_mapping, example_inputs)
    dump_graph(prepared, "PTQ prepared FX graph")
    dump_named_module_types(prepared)

    # calibration pass
    for x in calib_data:
        prepared(x)

    # print observers
    print_header("PTQ observer modules")
    for name, mod in prepared.named_modules():
        t = type(mod).__name__
        if "Observer" in t or "FakeQuant" in t:
            print(f"{name:60s} {t}")

    converted = convert_fx(copy.deepcopy(prepared).eval())
    dump_graph(converted, "PTQ converted FX graph")
    extract_add_region_from_graph(converted, "PTQ converted graph add-region")
    dump_named_module_types(converted)

    # compare output with float
    x = calib_data[0]
    float_out = float_model.eval()(x)
    quant_out = converted.eval()(x)
    print_header("PTQ float vs quant output error")
    print("mse       :", mse(float_out, quant_out))
    print("cosine    :", cosine_sim(float_out, quant_out))

    return prepared, converted

# -----------------------------
# QAT flow
# -----------------------------
def run_qat_observation(float_model, example_inputs, train_data, steps=8):
    print_header(f"QAT prepare_qat_fx / convert_fx  (backend={BACKEND})")
    qconfig_mapping = get_default_qat_qconfig_mapping_safe(BACKEND)

    qat_model = prepare_qat_fx(copy.deepcopy(float_model).train(), qconfig_mapping, example_inputs)
    dump_graph(qat_model, "QAT prepared FX graph")
    dump_named_module_types(qat_model)

    # lightweight fake training to let fake-quant/observers move
    opt = torch.optim.SGD(qat_model.parameters(), lr=1e-3, momentum=0.0)
    loss_history = []

    ref_model = copy.deepcopy(float_model).train()

    for step in range(steps):
        x = train_data[step % len(train_data)]

        # reference float forward
        ref_out = ref_model(x)

        # QAT forward
        qat_out = qat_model(x)

        # simple self-distillation style objective
        loss = torch.mean((qat_out - ref_out.detach()) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(float(loss.item()))

    print_header("QAT training proxy loss")
    for i, v in enumerate(loss_history):
        print(f"step {i:02d}: {v:.8f}")

    print_header("QAT fake quant modules")
    fqs = find_fake_quant_modules(qat_model)
    for name, mod in fqs:
        scale = getattr(mod, "scale", None)
        zp = getattr(mod, "zero_point", None)
        s = scale.detach().cpu().flatten()[:8].tolist() if isinstance(scale, torch.Tensor) else scale
        z = zp.detach().cpu().flatten()[:8].tolist() if isinstance(zp, torch.Tensor) else zp
        print(f"{name:60s} type={type(mod).__name__} scale(sample)={s} zero_point(sample)={z}")

    # one-step compare
    x = train_data[0]
    with torch.no_grad():
        float_out = float_model.eval()(x)
        qat_fake_out = qat_model.eval()(x)

    print_header("Float vs QAT-fake output error")
    print("mse       :", mse(float_out, qat_fake_out))
    print("cosine    :", cosine_sim(float_out, qat_fake_out))

    # convert
    qat_converted = convert_fx(copy.deepcopy(qat_model).eval())
    dump_graph(qat_converted, "QAT converted FX graph")
    extract_add_region_from_graph(qat_converted, "QAT converted graph add-region")
    dump_named_module_types(qat_converted)

    with torch.no_grad():
        q_out = qat_converted.eval()(x)
    print_header("Float vs QAT-converted output error")
    print("mse       :", mse(float_out, q_out))
    print("cosine    :", cosine_sim(float_out, q_out))

    return qat_model, qat_converted

# -----------------------------
# Main
# -----------------------------
def main():
    print_header("Environment")
    print("torch version:", torch.__version__)
    print("quant backend:", BACKEND)
    print("cuda available:", torch.cuda.is_available())
    print("NOTE: FX/PTQ/QAT observation is run on CPU backend on purpose.")

    model = BlockModel().cpu().eval()
    example_inputs = (torch.randn(1, 8, 32, 32),)
    calib_data = make_calib_data(num_batches=16, batch_size=4)

    print_header("FLOAT calibration stats")
    float_stats = observe_float_calibration(copy.deepcopy(model), calib_data)
    print(json.dumps(float_stats, indent=2))

    # PTQ observe
    ptq_prepared, ptq_converted = run_ptq_observation(copy.deepcopy(model), example_inputs, calib_data)

    # QAT observe
    qat_prepared, qat_converted = run_qat_observation(copy.deepcopy(model), example_inputs, calib_data, steps=8)

    print_header("Save concise report files")
    with open("float_calibration_stats.json", "w") as f:
        json.dump(float_stats, f, indent=2)

    with open("README_observe.txt", "w") as f:
        f.write(
            "Generated artifacts:\\n"
            "- float_calibration_stats.json\\n"
            "- stdout includes PTQ/QAT prepared and converted FX graphs\\n"
            "- compare add-region and module types around convert\\n"
        )

    print("Saved: float_calibration_stats.json, README_observe.txt")

if __name__ == "__main__":
    main()