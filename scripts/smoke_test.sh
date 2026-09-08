#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pipedream-smoke.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

llvm_bin_dir="${LLVM_BIN_DIR:-}"

resolve_tool() {
    local tool_name="$1"
    if [[ -n "${llvm_bin_dir}" && -x "${llvm_bin_dir}/${tool_name}" ]]; then
        printf '%s\n' "${llvm_bin_dir}/${tool_name}"
        return
    fi
    command -v "${tool_name}" 2>/dev/null || true
}

clang_bin="$(resolve_tool clang)"
opt_bin="$(resolve_tool opt)"
llvm_as_bin="$(resolve_tool llvm-as)"
llvm_dis_bin="$(resolve_tool llvm-dis)"

missing=()
for tool_path in "${clang_bin}" "${opt_bin}" "${llvm_as_bin}" "${llvm_dis_bin}"; do
    if [[ -z "${tool_path}" ]]; then
        missing+=("missing LLVM executable")
    fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
    cat >&2 <<'EOF'
Pipedream M0 smoke test cannot run.
Required tools: clang, opt, llvm-as, llvm-dis.
Install the pinned LLVM 20.x package set or run the test through Docker:
  docker build -t pipedream:llvm20 .
  docker run --rm -v "$PWD":/workspace pipedream:llvm20
EOF
    exit 2
fi

cat > "${work_dir}/smoke.c" <<'EOF'
#include <stdint.h>

static int32_t mix(int32_t value) {
    int32_t doubled = value * 2;
    return doubled + 0;
}

int main(void) {
    return mix(21) == 42 ? 0 : 1;
}
EOF

"${clang_bin}" \
    -O0 \
    -Xclang -disable-O0-optnone \
    -fno-discard-value-names \
    -emit-llvm \
    -c "${work_dir}/smoke.c" \
    -o "${work_dir}/input.bc"

"${opt_bin}" \
    -passes="mem2reg,instcombine,simplifycfg,gvn,dce,verify" \
    "${work_dir}/input.bc" \
    -o "${work_dir}/optimized.bc"

"${opt_bin}" \
    -passes=verify \
    "${work_dir}/optimized.bc" \
    -o /dev/null

"${llvm_dis_bin}" \
    "${work_dir}/optimized.bc" \
    -o "${work_dir}/optimized.ll"

"${llvm_as_bin}" \
    "${work_dir}/optimized.ll" \
    -o "${work_dir}/roundtrip.bc"

if [[ ! -s "${work_dir}/optimized.ll" || ! -s "${work_dir}/roundtrip.bc" ]]; then
    echo "Smoke test failed: LLVM output artifacts are empty." >&2
    exit 1
fi

printf 'Pipedream M0 smoke test passed\n'
printf 'Repository: %s\n' "${root_dir}"
printf 'clang: %s\n' "$("${clang_bin}" --version | head -n 1)"
printf 'opt: %s\n' "$("${opt_bin}" --version | head -n 1)"
printf 'Output IR: %s\n' "${work_dir}/optimized.ll"
