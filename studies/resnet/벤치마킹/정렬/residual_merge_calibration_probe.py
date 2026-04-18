import copy
import json
import operator
from collections import defaultdict

import torch
import torch.nn as nn

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx

torch.manual_seed(0)

# =========================================================
# Common helpers
# =========================================================
def get_backend_name():
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

def make_calib_data(num_batches=16, batch_size=4):
    xs = []
    for _ in range(num_batches):
        xs.append(torch.randn(batch_size, 8, 32, 32))
    return xs

def extract_graph_metrics(model):
    metrics = {
        "graph_num_nodes": None,
        "standalone_add_count": 0,
        "quantized_add_relu_count": 0,
        "quantized_add_count": 0,
        "quantize_per_tensor_count": 0,
        "dequantize_count": 0,
        "reformat_count": None,   # backend compile result placeholder
        "backend_final_layer_count": None,  # backend compile result placeholder
    }

    if not hasattr(model, "graph"):
        return metrics

    nodes = list(model.graph.nodes)
    metrics["graph_num_nodes"] = len(nodes)

    for n in nodes:
        tgt = str(n.target)
        if n.op == "call_function" and n.target == operator.add:
            metrics["standalone_add_count"] += 1
        if "quantized.add_relu" in tgt:
            metrics["quantized_add_relu_count"] += 1
        elif "quantized.add" in tgt:
            metrics["quantized_add_count"] += 1
        if "quantize_per_tensor" in tgt:
            metrics["quantize_per_tensor_count"] += 1
        if n.op == "call_method" and tgt == "dequantize":
            metrics["dequantize_count"] += 1

    return metrics

def dump_graph_text(model):
    if hasattr(model, "graph"):
        return str(model.graph)
    return "No FX graph"

# =========================================================
# ResNet v1 block
# Conv-BN-ReLU -> Conv-BN -> Add -> ReLU
# =========================================================
class InstrumentedBasicBlockV1(nn.Module):
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

    def forward(self, x):
        self.debug = {}
        identity = self.downsample(x)
        self.debug["skip_out"] = identity

        out = self.conv1(x)
        self.debug["conv1_out"] = out

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        self.debug["conv2_out"] = out

        out = self.bn2(out)
        self.debug["pre_add_main"] = out
        self.debug["pre_add_skip"] = identity

        add_out = out + identity
        self.debug["post_add"] = add_out

        out = self.relu(add_out)
        self.debug["post_relu"] = out
        return out

# =========================================================
# ResNet v2 block
# BN-ReLU-Conv -> BN-ReLU-Conv -> Add
# projection path kept simple for stride/channel match
# =========================================================
class InstrumentedBasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, in_ch=8, out_ch=16, stride=2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.debug = {}

    def forward(self, x):
        self.debug = {}
        identity = self.downsample(x)
        self.debug["skip_out"] = identity

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        self.debug["conv1_out"] = out

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        self.debug["conv2_out"] = out

        self.debug["pre_add_main"] = out
        self.debug["pre_add_skip"] = identity

        add_out = out + identity
        self.debug["post_add"] = add_out

        # v2 block ends at add
        self.debug["post_relu"] = add_out
        return add_out

# =========================================================
# Wrapper
# =========================================================
class BlockModel(nn.Module):
    def __init__(self, block_ctor):
        super().__init__()
        self.block = block_ctor()

    def forward(self, x):
        return self.block(x)

# =========================================================
# Observation functions
# =========================================================
@torch.no_grad()
def observe_float_calibration(model, data):
    model.eval()
    acc = defaultdict(list)
    for x in data:
        _ = model(x)
        dbg = model.block.debug
        for k in ["conv1_out", "conv2_out", "skip_out", "pre_add_main", "pre_add_skip", "post_add", "post_relu"]:
            acc[k].append(tensor_stats(dbg[k]))

    summary = {}
    for k, rows in acc.items():
        keys = rows[0].keys()
        summary[k] = {kk: float(sum(r[kk] for r in rows) / len(rows)) for kk in keys}
    return summary

@torch.no_grad()
def run_ptq(float_model, example_inputs, calib_data):
    qconfig_mapping = get_default_qconfig_mapping_safe(BACKEND)
    prepared = prepare_fx(copy.deepcopy(float_model).eval(), qconfig_mapping, example_inputs)
    for x in calib_data:
        prepared(x)
    converted = convert_fx(copy.deepcopy(prepared).eval())

    x = calib_data[0]
    float_out = float_model.eval()(x)
    quant_out = converted.eval()(x)

    return {
        "prepared_graph": dump_graph_text(prepared),
        "converted_graph": dump_graph_text(converted),
        "graph_metrics": extract_graph_metrics(converted),
        "float_vs_quant_mse": mse(float_out, quant_out),
        "float_vs_quant_cosine": cosine_sim(float_out, quant_out),
    }

def run_qat(float_model, example_inputs, train_data, steps=8):
    qconfig_mapping = get_default_qat_qconfig_mapping_safe(BACKEND)
    qat_model = prepare_qat_fx(copy.deepcopy(float_model).train(), qconfig_mapping, example_inputs)

    opt = torch.optim.SGD(qat_model.parameters(), lr=1e-3, momentum=0.0)
    ref_model = copy.deepcopy(float_model).train()
    loss_history = []

    for step in range(steps):
        x = train_data[step % len(train_data)]
        ref_out = ref_model(x)
        qat_out = qat_model(x)
        loss = torch.mean((qat_out - ref_out.detach()) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))

    x = train_data[0]
    with torch.no_grad():
        float_out = float_model.eval()(x)
        qat_fake_out = qat_model.eval()(x)

    converted = convert_fx(copy.deepcopy(qat_model).eval())
    with torch.no_grad():
        q_out = converted.eval()(x)

    return {
        "prepared_graph": dump_graph_text(qat_model),
        "converted_graph": dump_graph_text(converted),
        "graph_metrics": extract_graph_metrics(converted),
        "loss_history": loss_history,
        "float_vs_qat_fake_mse": mse(float_out, qat_fake_out),
        "float_vs_qat_fake_cosine": cosine_sim(float_out, qat_fake_out),
        "float_vs_qat_converted_mse": mse(float_out, q_out),
        "float_vs_qat_converted_cosine": cosine_sim(float_out, q_out),
    }

def compute_comparison_metrics(float_stats, ptq_info, qat_info):
    main = float_stats["pre_add_main"]
    skip = float_stats["pre_add_skip"]
    post_add = float_stats["post_add"]

    out = {
        "branch_ratio_p99": skip["p99"] / max(main["p99"], 1e-12),
        "branch_ratio_std": skip["std"] / max(main["std"], 1e-12),
        "post_add_p99_over_max_preadd_p99": post_add["p99"] / max(main["p99"], skip["p99"], 1e-12),

        "ptq_convert_has_add_relu": ptq_info["graph_metrics"]["quantized_add_relu_count"] > 0,
        "qat_convert_has_add_relu": qat_info["graph_metrics"]["quantized_add_relu_count"] > 0,

        "ptq_backend_final_layer_count": None,
        "qat_backend_final_layer_count": None,
        "ptq_standalone_add_count": ptq_info["graph_metrics"]["standalone_add_count"],
        "qat_standalone_add_count": qat_info["graph_metrics"]["standalone_add_count"],
        "ptq_reformat_count": None,
        "qat_reformat_count": None,
    }
    return out

def analyze_structure_score(cmp_metrics):
    # lower is better for ratios, standalone add, reformat, backend layer count
    # add_relu presence is good
    score = 0.0
    score += cmp_metrics["branch_ratio_p99"]
    score += cmp_metrics["branch_ratio_std"]
    score += cmp_metrics["post_add_p99_over_max_preadd_p99"]
    score += 3.0 * cmp_metrics["ptq_standalone_add_count"]
    score += 3.0 * cmp_metrics["qat_standalone_add_count"]
    score += 0.0 if cmp_metrics["ptq_convert_has_add_relu"] else 2.0
    score += 0.0 if cmp_metrics["qat_convert_has_add_relu"] else 2.0
    return score

def run_one(block_name, block_ctor, calib_data):
    model = BlockModel(block_ctor).cpu().eval()
    example_inputs = (torch.randn(1, 8, 32, 32),)

    float_stats = observe_float_calibration(copy.deepcopy(model), calib_data)
    ptq_info = run_ptq(copy.deepcopy(model), example_inputs, calib_data)
    qat_info = run_qat(copy.deepcopy(model), example_inputs, calib_data, steps=8)
    cmp_metrics = compute_comparison_metrics(float_stats, ptq_info, qat_info)
    score = analyze_structure_score(cmp_metrics)

    return {
        "block_name": block_name,
        "float_calibration_stats": float_stats,
        "ptq": ptq_info,
        "qat": qat_info,
        "comparison_metrics": cmp_metrics,
        "structure_score_lower_is_better": score,
    }

def main():
    print(f"torch={torch.__version__}, backend={BACKEND}, cuda_available={torch.cuda.is_available()}")
    print("NOTE: backend compile metrics (final layer count / standalone add / reformat) are placeholders here.")
    print("      Fill them later from TensorRT/OpenVINO compile logs if needed.\n")

    calib_data = make_calib_data(num_batches=16, batch_size=4)

    results = {
        "env": {
            "torch_version": torch.__version__,
            "quant_backend": BACKEND,
            "cuda_available": torch.cuda.is_available(),
        },
        "v1": run_one("ResNetV1_BasicBlock", InstrumentedBasicBlockV1, calib_data),
        "v2": run_one("ResNetV2_BasicBlock", InstrumentedBasicBlockV2, calib_data),
    }

    # simple winner by current structural score
    if results["v1"]["structure_score_lower_is_better"] < results["v2"]["structure_score_lower_is_better"]:
        winner = "v1"
    else:
        winner = "v2"

    results["summary"] = {
        "winner_by_current_score": winner,
        "note": (
            "Current score uses float branch mismatch + PTQ/QAT FX converted graph cleanliness only. "
            "backend_final_layer_count / standalone add / reformat must be filled from backend compile logs."
        ),
        "wanted_compare_metrics": [
            "branch_ratio_p99 = pre_add_skip.p99 / pre_add_main.p99",
            "branch_ratio_std = pre_add_skip.std / pre_add_main.std",
            "post_add.p99 / max(pre_add_main.p99, pre_add_skip.p99)",
            "convert 후 add_relu 여부",
            "backend 최종 layer 수",
            "standalone add 수",
            "reformat 수",
        ],
    }

    with open("result.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["summary"], indent=2))
    print("\nSaved: result.json")

if __name__ == "__main__":
    main()