#!/usr/bin/env bash
# reproduce_tables.sh -- Regenerate all manuscript tables from scratch.
# Run after training all seeds: python scripts/train.py --all_seeds

set -e
echo "=== Reproducing all tables ==="

echo "[1/4] Table 3: SHAP sensitivity analysis"
python experiments/shap_sensitivity.py --config config/default.yaml

echo "[2/4] Table 7+8: Detection metrics on test partition"
python scripts/evaluate.py --config config/default.yaml --split test

echo "[3/4] Table 9: Ablation study"
python experiments/ablation.py --config config/default.yaml

echo "[4/4] Table 10: LOIO cross-validation"
python scripts/train_loio.py --config config/default.yaml

echo "=== All tables saved to outputs/tables/ ==="
