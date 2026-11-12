#!/usr/bin/env bash
# Download the final weights (1.3 GB) into policy_bundle/checkpoint/.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download Metric-AI/roboeval-bimanual-2026 --include 'checkpoint/*' --local-dir policy_bundle
ls -la policy_bundle/checkpoint
