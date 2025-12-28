#!/usr/bin/env bash
set -euo pipefail
INPUT=${1:-data/processed.sample.parquet}
CONFIG=${2:-configs/default.yaml}
python3 -m src.train.train --config "${CONFIG}" --input "${INPUT}"
