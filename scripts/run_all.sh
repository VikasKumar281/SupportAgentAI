#!/usr/bin/env bash
# Reproduces every headline result in this repo end to end.
# Expected runtime: 1-3 minutes without an API key, ~5-10 minutes with one
# (network-bound on LLM calls), well under the 15-minute budget.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6: Generating synthetic dataset (Kaggle-schema-compatible) =="
python data/generate_synthetic_data.py

echo
echo "== 2/6: Building conversation pairs for brand =="
python -m src.data_prep

echo
echo "== 3/6: Building golden evaluation set (stratified sample + labels) =="
python eval/build_golden_set.py

echo
echo "== 4/6: Training baseline intent classifier (TF-IDF + LogReg) =="
python -m src.classify_baseline

echo
echo "== 5/6: Running tests =="
python -m pytest tests/ -q

echo
echo "== 6/6: Running full evaluation harness =="
python eval/run_eval.py

echo
echo "Done. See outputs/metrics.json, outputs/predictions.csv, outputs/judge_scores.csv"
echo "and REPORT.md for the written analysis."
