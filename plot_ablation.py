"""
plot_ablation.py -- Figure 13: FPR at fixed recall for each ablation configuration.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"]    = "Times New Roman"
rcParams["font.size"]      = 8
rcParams["axes.labelsize"] = 9

CONFIGS = [
    "Full Model",
    "w/o Keystroke",
    "w/o Mouse",
    "w/o App",
    "w/o KS+Mouse",
    "w/o L_smooth",
    "Uniform Attn",
    "w/o SHAP",
]
FPR_AT_90_RECALL = [0.0048, 0.0071, 0.0065, 0.0093, 0.0099, 0.0055, 0.0078, 0.0048]
STD_FPR          = [0.0003, 0.0005, 0.0004, 0.0007, 0.0008, 0.0004, 0.0006, 0.0003]

def main():
    x = np.arange(len(CONFIGS))
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    colors = ["#CC3333"] + ["#4477AA"] * 7
    bars = ax.bar(x, FPR_AT_90_RECALL, yerr=STD_FPR, capsize=4, width=0.65,
                  color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CONFIGS, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("FPR at fixed Recall=0.90")
    ax.set_title("Figure 13: Ablation — FPR at Fixed Recall")
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/figure13_ablation.png", dpi=600, bbox_inches="tight")
    plt.close(); print("Saved outputs/figures/figure13_ablation.png")

if __name__ == "__main__": main()
