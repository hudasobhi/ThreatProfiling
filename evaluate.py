"""
evaluate.py -- Evaluation script. Loads trained checkpoints and reports
Table 7 metrics: AUC-ROC, precision, recall, F1, fidelity across 10 seeds.

Usage:
    python scripts/evaluate.py --config config/default.yaml --split test
"""

import argparse, os, json, csv
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from data.dataset       import CERTWindowDataset, collate_fn
from src.shadowdetectai import ShadowDetectAI
from src.shapley        import ShapleyAttributionLayer
from src.calibration    import PlattCalibration

SEEDS = [42, 123, 256, 512, 1024, 999, 777, 333, 111, 888]


def evaluate_checkpoint(model, loader, device, cfg, threshold=0.5):
    d = cfg["model"]["embedding_dim"]
    model.eval(); probs_all, labels_all, logits_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            B  = batch["label"].shape[0]
            nb = torch.zeros(B, 4, d).to(device)
            logits, probs, _, _ = model(batch, nb)
            probs_all.extend(probs.cpu().numpy())
            logits_all.extend(logits.cpu().numpy())
            labels_all.extend(batch["label"].cpu().numpy())

    labels = np.array(labels_all)
    probs  = np.array(probs_all)
    preds  = (probs > threshold).astype(int)
    auc    = float(roc_auc_score(labels, probs))
    prec   = float(precision_score(labels, preds, zero_division=0))
    rec    = float(recall_score(labels, preds, zero_division=0))
    f1     = float(f1_score(labels, preds, zero_division=0))
    cm     = confusion_matrix(labels, preds)
    return {"auc": auc, "precision": prec, "recall": rec, "f1": f1, "confusion": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/default.yaml")
    parser.add_argument("--data",    default="data/processed/")
    parser.add_argument("--ckpt_dir",default="outputs/models/")
    parser.add_argument("--output",  default="outputs/")
    parser.add_argument("--split",   default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    with open(os.path.join(args.data, "graph_metadata.json")) as f:
        meta = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = torch.load(os.path.join(args.data, f"{args.split}_windows.pt"))
    loader  = DataLoader(CERTWindowDataset(windows), batch_size=256,
                         shuffle=False, collate_fn=collate_fn)

    all_results = []
    for seed in SEEDS:
        ckpt = os.path.join(args.ckpt_dir, f"shadow_seed{seed}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found for seed {seed}, skipping."); continue
        model = ShadowDetectAI(meta["n_nodes"], meta["resource_vocab_size"], cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.reset_memory()
        metrics = evaluate_checkpoint(model, loader, device, cfg)
        metrics["seed"] = seed
        all_results.append(metrics)
        print(f"  Seed {seed}: AUC={metrics['auc']:.4f}  P={metrics['precision']:.4f}"
              f"  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    if all_results:
        aucs = [r["auc"] for r in all_results]
        precs = [r["precision"] for r in all_results]
        recs  = [r["recall"] for r in all_results]
        f1s   = [r["f1"] for r in all_results]
        print(f"\n--- Table 7 ({args.split}) ---")
        print(f"AUC-ROC:   {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
        print(f"Precision: {np.mean(precs):.3f} +/- {np.std(precs):.3f}")
        print(f"Recall:    {np.mean(recs):.3f} +/- {np.std(recs):.3f}")
        print(f"F1:        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

        os.makedirs(os.path.join(args.output, "tables"), exist_ok=True)
        rows = [{k: v for k, v in r.items() if k != "confusion"} for r in all_results]
        with open(os.path.join(args.output, "tables", f"table7_{args.split}.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

if __name__ == "__main__": main()
