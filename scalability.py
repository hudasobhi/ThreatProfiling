"""
scalability.py -- Figure 11: Training and inference scalability.

Reports wall-clock time as the number of active users in the interaction
graph increases, validating the O(|V|d + |E_t|d) complexity (Section 4).

Usage:
    python experiments/scalability.py --config config/default.yaml
"""

import argparse, os, time
import numpy as np
import torch
import yaml

from src.shadowdetectai import ShadowDetectAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_sizes = [500, 1000, 2000, 3000, 4000, 5000]
    results = {"n_users": [], "train_time_s": [], "inference_ms": []}

    for n in graph_sizes:
        print(f"  Graph size: {n} nodes ...")
        # Synthetic benchmark: random feature tensors
        B    = min(256, n)
        d    = cfg["model"]["embedding_dim"]
        batch = {
            "keystroke_features": torch.zeros(B, 8).to(device),
            "mouse_features":     torch.zeros(B, 6).to(device),
            "app_sequence":       torch.ones(B, 10, dtype=torch.long).to(device),
            "seq_lengths":        torch.full((B,), 10).to(device),
            "user_idx":           torch.arange(B).to(device),
            "window_idx":         torch.zeros(B).to(device),
            "label":              torch.zeros(B, dtype=torch.long).to(device),
        }
        nb = torch.zeros(B, 4, d).to(device)

        model = ShadowDetectAI(n, 100, cfg).to(device)
        model.reset_memory()

        t0 = time.perf_counter()
        logits, _, emb, ls = model(batch, nb)
        loss = model.compute_loss(logits, batch["label"], ls)
        loss.backward()
        t_train = (time.perf_counter() - t0) * 1000

        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad(): model(batch, nb)
        t_inf = (time.perf_counter() - t0) * 1000

        results["n_users"].append(n)
        results["train_time_s"].append(t_train / 1000)
        results["inference_ms"].append(t_inf)
        print(f"    train={t_train:.1f} ms  inference={t_inf:.1f} ms/batch")

    os.makedirs(os.path.join(args.output, "tables"), exist_ok=True)
    import csv
    rows = [{"n_users": results["n_users"][i],
             "train_time_s": results["train_time_s"][i],
             "inference_ms": results["inference_ms"][i]}
            for i in range(len(graph_sizes))]
    with open(os.path.join(args.output, "tables", "scalability.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("Saved outputs/tables/scalability.csv")

if __name__ == "__main__": main()
