"""
plot_fidelity.py -- Figure 12: Attribution fidelity across explainable baselines.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 9

METHODS   = ["Gaspar et al.", "EXAI",           "ShadowDetectAI\n(proposed)"]
FIDELITY  = [0.881,            0.869,             0.943]
STD_FID   = [0.006,            0.007,             0.003]
COLORS    = ["#888888",        "#AAAAAA",          "#CC3333"]

def main():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    bars = ax.bar(METHODS, FIDELITY, yerr=STD_FID, capsize=5, width=0.5,
                  color=COLORS, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Attribution Fidelity (Eq. 30)")
    ax.set_title("Figure 12: Attribution Fidelity")
    ax.set_ylim(0.83, 0.98)
    for bar, m in zip(bars, FIDELITY):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.004,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/figure12_fidelity.png", dpi=600, bbox_inches="tight")
    plt.close(); print("Saved outputs/figures/figure12_fidelity.png")

if __name__ == "__main__": main()
