#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

uv run pipedream-manifest --manifest benchmarks/manifest.json --validate
uv run pipedream-split-manifest \
    --input benchmarks/manifest.json \
    --output benchmarks/pilot_manifest.json \
    --tier pilot
uv run pipedream-fit-normalizer \
    --manifest benchmarks/pilot_manifest.json \
    --schema configs/autophase_schema.yaml \
    --split train \
    --output results/raw/pilot.normalization.json
uv run pipedream-baselines \
    --manifest benchmarks/manifest.json \
    --catalog configs/pass_catalog.yaml \
    --split smoke \
    --limit 2 \
    --seeds 0 1 \
    --max-steps 3 \
    --output results/raw/baselines.smoke.jsonl

printf 'Smoke reproduction artifacts written under results/raw\n'
