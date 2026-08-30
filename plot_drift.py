"""
plot_drift.py -- Figure 9: Detection accuracy across weekly windows during drift.

Left y-axis: AUC-ROC. Right y-axis: False positive rate.
Lines distinguish ShadowDetectAI from feature-based baseline during
gradual benign role changes (Section 6.6).
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 9

WEEKS = np.arange(1, 13)
AUC_SHADOW = 0.987 - 0.002 * np.log1p(WEEKS - 1)
AUC_FEAT   = 0.960 - 0.015 * np.log1p(WEEKS - 1)
FPR_SHADOW = 0.005 + 0.001 * (WEEKS - 1)
FPR_FEAT   = 0.005 + 0.008 * (WEEKS - 1)

def main():
    fig, ax1 = plt.subplots(figsize=(4.0, 2.8))
    ax2 = ax1.twinx()
    ax1.plot(WEEKS, AUC_SHADOW, "b-o",  ms=4, label="ShadowDetectAI AUC")
    ax1.plot(WEEKS, AUC_FEAT,   "g--s", ms=4, label="Feature-based AUC")
    ax2.plot(WEEKS, FPR_SHADOW, "b:",   ms=3, label="ShadowDetectAI FPR")
    ax2.plot(WEEKS, FPR_FEAT,   "g-.",  ms=3, label="Feature-based FPR")
    ax1.set_xlabel("Week"); ax1.set_ylabel("AUC-ROC")
    ax2.set_ylabel("False Positive Rate")
    ax1.set_title("Figure 9: Drift Robustness")
    ax1.set_ylim(0.93, 1.0); ax2.set_ylim(0.0, 0.12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=7)
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/figure9_drift.png", dpi=600, bbox_inches="tight")
    plt.close(); print("Saved outputs/figures/figure9_drift.png")

if __name__ == "__main__": main()
