#!/usr/bin/env bash
# reproduce_figures.sh -- Regenerate all manuscript figures.

set -e
echo "=== Reproducing all figures ==="

python visualization/plot_roc.py
python visualization/plot_attribution.py
python visualization/plot_drift.py
python visualization/plot_calibration.py
python visualization/plot_scalability.py
python visualization/plot_fidelity.py
python visualization/plot_ablation.py
python triage/analysis.py   # Figure 14

echo "=== All figures saved to outputs/figures/ ==="
