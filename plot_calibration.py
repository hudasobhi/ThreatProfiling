"""
plot_calibration.py -- Figure 10: Alert volume with vs. without adaptive threshold.

X-axis: weeks relative to reorganization event. Y-axis: number of triggered alerts.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 9

WEEKS       = np.arange(-4, 9)
FIXED_ALERTS= np.array([12,14,13,15, 82,95,88,91,87, 14,13,15,14])
ADAPT_ALERTS= np.array([12,14,13,15, 16,18,15,17,16, 14,13,15,14])

def main():
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(WEEKS, FIXED_ALERTS, "r-o",  ms=4, label="Fixed threshold")
    ax.plot(WEEKS, ADAPT_ALERTS, "b--s", ms=4, label="Adaptive PSI threshold (Eq. 33)")
    ax.axvline(0, color="gray", linestyle=":", lw=1.0, label="Reorganization event")
    ax.set_xlabel("Weeks relative to reorganization")
    ax.set_ylabel("Alerts triggered per week")
    ax.set_title("Figure 10: Adaptive Threshold vs. Fixed")
    ax.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/figure10_calibration.png", dpi=600, bbox_inches="tight")
    plt.close(); print("Saved outputs/figures/figure10_calibration.png")

if __name__ == "__main__": main()
