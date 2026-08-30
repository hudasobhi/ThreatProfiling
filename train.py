"""
train.py -- Main training script for ShadowDetectAI.

Usage:
    python scripts/train.py --config config/default.yaml --seed 42
    python scripts/train.py --config config/default.yaml --all_seeds
"""

import argparse, os, random, json
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from data.dataset       import CERTWindowDataset, collate_fn
from src.shadowdetectai import ShadowDetectAI

SEEDS = [42, 123, 256, 512, 1024, 999, 777, 333, 111, 888]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, loader, device, d):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            B = batch["label"].shape[0]
            nb = torch.zeros(B, 4, d).to(device)
            _, p, _, _ = model(batch, nb)
            probs_all.extend(p.cpu().numpy())
            labels_all.extend(batch["label"].cpu().numpy())
    try:    return float(roc_auc_score(labels_all, probs_all))
    except: return 0.0


def train_one_run(cfg, seed, data_dir, out_dir):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_w = torch.load(os.path.join(data_dir, "train_windows.pt"))
    val_w   = torch.load(os.path.join(data_dir, "val_windows.pt"))
    with open(os.path.join(data_dir, "graph_metadata.json")) as f:
        meta = json.load(f)

    d = cfg["model"]["embedding_dim"]
    model = ShadowDetectAI(meta["n_nodes"], meta["resource_vocab_size"], cfg).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    train_loader = DataLoader(CERTWindowDataset(train_w),
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(CERTWindowDataset(val_w),
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    best_auc, prev_emb = 0.0, None
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); model.reset_memory(); total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Seed {seed} Ep {epoch+1}", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            B  = batch["label"].shape[0]
            nb = torch.zeros(B, 4, d).to(device)
            logits, _, emb, ls = model(batch, nb, prev_emb)
            loss = model.compute_loss(logits, batch["label"], ls)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); prev_emb = emb.detach()
        val_auc = evaluate(model, val_loader, device, d)
        print(f"  Epoch {epoch+1}  loss={total_loss/len(train_loader):.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "models", f"shadow_seed{seed}_best.pt"))
    return {"seed": seed, "best_val_auc": best_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config/default.yaml")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--all_seeds", action="store_true")
    parser.add_argument("--output",    default="outputs/")
    parser.add_argument("--data",      default="data/processed/")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    seeds = SEEDS if args.all_seeds else [args.seed]
    results = [train_one_run(cfg, s, args.data, args.output) for s in seeds]
    aucs = [r["best_val_auc"] for r in results]
    print(f"\nVal AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

if __name__ == "__main__": main()
