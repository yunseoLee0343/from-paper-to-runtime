#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python setup.py build_ext --inplace

SO_PATH="$(python - <<'PY'
import glob
matches = glob.glob("build/**/myimg_ext*.so", recursive=True)
if matches:
    print(matches[0])
else:
    local_matches = glob.glob("myimg_ext*.so")
    print(local_matches[0] if local_matches else "")
PY
)"

if [[ -z "$SO_PATH" ]]; then
  echo "Could not locate built extension shared object" >&2
  exit 1
fi

OUT_DIR="sass_dump"
mkdir -p "$OUT_DIR"

cuobjdump --dump-sass "$SO_PATH" > "$OUT_DIR/cuobjdump_sass.txt"
nvdisasm "$SO_PATH" > "$OUT_DIR/nvdisasm.txt" || true

grep -n "conv1_bias_relu_kernel" "$OUT_DIR/cuobjdump_sass.txt" || true
grep -n "conv1_bias_relu_kernel" "$OUT_DIR/nvdisasm.txt" || true

CUOBJ_FFMA="$(grep -c "FFMA" "$OUT_DIR/cuobjdump_sass.txt" || true)"
CUOBJ_INST="$(grep -Ec '^[[:space:]]*/\*|^[[:space:]]+[0-9a-f]+' "$OUT_DIR/cuobjdump_sass.txt" || true)"
NVDIS_FFMA="$(grep -c "FFMA" "$OUT_DIR/nvdisasm.txt" || true)"
NVDIS_INST="$(grep -Ec '^[[:space:]]*/\*|^[[:space:]]+[0-9a-f]+' "$OUT_DIR/nvdisasm.txt" || true)"

python - <<PY
def ratio(ffma, total):
    return 0.0 if total == 0 else 100.0 * ffma / total

stats = {
    "cuobjdump_ffma": int("${CUOBJ_FFMA}" or 0),
    "cuobjdump_total": int("${CUOBJ_INST}" or 0),
    "nvdisasm_ffma": int("${NVDIS_FFMA}" or 0),
    "nvdisasm_total": int("${NVDIS_INST}" or 0),
}
print("SASS summary")
print(f"  cuobjdump: FFMA={stats['cuobjdump_ffma']} total={stats['cuobjdump_total']} ratio={ratio(stats['cuobjdump_ffma'], stats['cuobjdump_total']):.2f}%")
print(f"  nvdisasm : FFMA={stats['nvdisasm_ffma']} total={stats['nvdisasm_total']} ratio={ratio(stats['nvdisasm_ffma'], stats['nvdisasm_total']):.2f}%")
PY
