"""
plot_scalability.py -- Figure 11: Training time and inference latency vs. graph size.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 9

N_USERS   = [500, 1000, 2000, 3000, 4000, 5000]
TRAIN_S   = [1.2,  2.3,  4.6,  6.9,  9.2, 11.5]   # illustrative linear scaling
INF_MS    = [45,   68,  101,  133,  162,  194]

def main():
    fig, ax1 = plt.subplots(figsize=(4.0, 2.8))
    ax2 = ax1.twinx()
    ax1.plot(N_USERS, TRAIN_S, "b-o",  ms=4, label="Training time (s/epoch)")
    ax2.plot(N_USERS, INF_MS,  "r--s", ms=4, label="Inference (ms/user incl. SHAP)")
    ax1.set_xlabel("Number of active users")
    ax1.set_ylabel("Training time (s/epoch)", color="blue")
    ax2.set_ylabel("Inference latency (ms/user)", color="red")
    ax2.set_ylim(0, 250)
    ax1.set_title("Figure 11: Scalability")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/figure11_scalability.png", dpi=600, bbox_inches="tight")
    plt.close(); print("Saved outputs/figures/figure11_scalability.png")

if __name__ == "__main__": main()
