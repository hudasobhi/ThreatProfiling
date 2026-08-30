"""
preprocess.py — Algorithm 1: Behavioral Feature Encoding and Graph Construction.

Implements Section 3 and Section 4.1 of the manuscript.
Produces train/val/test partition files and LOIO fold files.

Usage:
    python data/preprocess.py --config config/data_config.yaml
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

from data.splits import chronological_split, loio_folds


# ---------------------------------------------------------------------------
# Raw log loading
# ---------------------------------------------------------------------------

def load_raw_logs(raw_dir: str) -> dict:
    files = ["logon", "device", "email", "file", "http"]
    logs = {}
    for name in files:
        path = os.path.join(raw_dir, f"{name}.csv")
        logs[name] = pd.read_csv(path, parse_dates=["date"])
        print(f"  Loaded {name}.csv  shape={logs[name].shape}")
    return logs


# ---------------------------------------------------------------------------
# Window generation (sliding non-overlapping 24-hour windows)
# ---------------------------------------------------------------------------

def generate_windows(logs: dict, window_hours: int = 24):
    """
    Concatenate all activity events, assign each to a 24-hour window index,
    and group events by (user, window_index).
    Returns a DataFrame with columns [user, window_idx, channel, timestamp, resource].
    """
    frames = []
    for channel, df in logs.items():
        tmp = df[["user", "date", "pc"]].copy()
        tmp.columns = ["user", "timestamp", "resource"]
        tmp["channel"] = channel
        frames.append(tmp)

    all_events = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    t_min = all_events["timestamp"].min()
    all_events["window_idx"] = ((all_events["timestamp"] - t_min).dt.total_seconds()
                                / (window_hours * 3600)).astype(int)
    return all_events


# ---------------------------------------------------------------------------
# Keystroke proxy features — Eqs. 11-13
# ---------------------------------------------------------------------------

def keystroke_proxy_features(user_events: pd.DataFrame) -> np.ndarray:
    """
    Approximate dwell time (d_i) and flight time (l_i) from inter-event
    intervals of logon-session activity records (Section 4.1).
    Returns 8-dimensional vector: [μ_d, σ_d, P50_d, P90_d, μ_l, σ_l, P50_l, P90_l].
    """
    logon_events = user_events[user_events["channel"] == "logon"].sort_values("timestamp")
    if len(logon_events) < 2:
        return np.zeros(8, dtype=np.float32)

    deltas = logon_events["timestamp"].diff().dt.total_seconds().dropna().values
    # dwell proxy: even-indexed intervals; flight proxy: odd-indexed intervals
    dwell  = deltas[0::2] if len(deltas[0::2]) > 0 else np.array([0.0])
    flight = deltas[1::2] if len(deltas[1::2]) > 0 else np.array([0.0])

    def stats(arr):
        return [
            float(np.mean(arr)),
            float(np.std(arr) + 1e-8),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 90)),
        ]

    return np.array(stats(dwell) + stats(flight), dtype=np.float32)


# ---------------------------------------------------------------------------
# Mouse proxy features — Eq. 14
# ---------------------------------------------------------------------------

def mouse_proxy_features(user_events: pd.DataFrame) -> np.ndarray:
    """
    Compute Shannon entropy of device-event frequency distribution and
    inter-event interval statistics as mouse movement proxy (Section 4.1).
    Returns 6-dimensional vector: [H_v, click_mean, click_std, iei_mean, iei_std, iei_p90].
    """
    device_events = user_events[user_events["channel"] == "device"].sort_values("timestamp")
    if len(device_events) < 2:
        return np.zeros(6, dtype=np.float32)

    # Shannon entropy over resource (device type) frequency distribution (Eq. 14)
    counts = device_events["resource"].value_counts(normalize=True).values
    entropy = float(-np.sum(counts * np.log(counts + 1e-8)))

    intervals = device_events["timestamp"].diff().dt.total_seconds().dropna().values
    if len(intervals) == 0:
        intervals = np.array([0.0])

    click_events = intervals[intervals < 1.0]
    click_mean = float(np.mean(click_events)) if len(click_events) > 0 else 0.0
    click_std  = float(np.std(click_events))  if len(click_events) > 0 else 0.0

    return np.array([
        entropy,
        click_mean,
        click_std,
        float(np.mean(intervals)),
        float(np.std(intervals) + 1e-8),
        float(np.percentile(intervals, 90)),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Application usage features — Eq. 15 (categorical index; GRU applied in encoder)
# ---------------------------------------------------------------------------

def application_usage_sequence(user_events: pd.DataFrame, resource_vocab: dict) -> list:
    """
    Map file/email/HTTP access events to categorical resource indices.
    Returns a list of integer resource indices (the GRU in encoder.application
    will embed and aggregate these into a fixed-length vector).
    """
    app_channels = ["file", "email", "http"]
    app_events = (user_events[user_events["channel"].isin(app_channels)]
                  .sort_values("timestamp"))
    indices = [resource_vocab.get(r, 0) for r in app_events["resource"].tolist()]
    return indices if indices else [0]


# ---------------------------------------------------------------------------
# Graph construction — Eqs. 4-7
# ---------------------------------------------------------------------------

def build_adjacency(user_events: pd.DataFrame, user_to_idx: dict,
                    resource_to_idx: dict) -> list:
    """
    Build edge list for dynamic interaction graph G_t = (V, E_t, X_t).
    An edge e_{u,v}^t exists whenever user u interacted with resource v
    during window t (Eq. 5).
    Returns list of (user_idx, resource_idx) tuples.
    """
    edges = []
    for _, row in user_events.iterrows():
        u_idx = user_to_idx.get(row["user"], -1)
        r_idx = resource_to_idx.get(row["resource"], -1)
        if u_idx >= 0 and r_idx >= 0:
            edges.append((u_idx, r_idx))
    return list(set(edges))


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(config: dict) -> None:
    raw_dir       = config["dataset"]["raw_dir"]
    processed_dir = config["dataset"]["processed_dir"]
    window_hours  = config["dataset"]["window_hours"]
    train_months  = config["dataset"]["splits"]["train_months"]
    val_months    = config["dataset"]["splits"]["val_months"]
    test_months   = config["dataset"]["splits"]["test_months"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "loio_folds"), exist_ok=True)

    print("Loading raw logs ...")
    logs = load_raw_logs(raw_dir)

    print("Generating windows ...")
    all_events = generate_windows(logs, window_hours)

    # Build vocabulary for resources
    all_resources = all_events["resource"].unique().tolist()
    resource_vocab = {r: i + 1 for i, r in enumerate(all_resources)}  # 0 reserved for UNK
    all_users = all_events["user"].unique().tolist()
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    # Resource nodes start after user nodes
    resource_to_idx = {r: len(all_users) + i for i, r in enumerate(all_resources)}

    # Save metadata
    metadata = {
        "n_users": len(all_users),
        "n_resources": len(all_resources),
        "n_nodes": len(all_users) + len(all_resources),
        "resource_vocab_size": len(resource_vocab) + 1,
    }
    with open(os.path.join(processed_dir, "graph_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Nodes: {metadata['n_nodes']} (users={metadata['n_users']}, "
          f"resources={metadata['n_resources']})")

    # Assign month index to each event
    t_min = all_events["timestamp"].min()
    all_events["month_idx"] = (
        (all_events["timestamp"].dt.year - t_min.year) * 12
        + (all_events["timestamp"].dt.month - t_min.month) + 1
    )

    for split_name, months in [("train", train_months),
                                ("val",   val_months),
                                ("test",  test_months)]:
        print(f"\nBuilding {split_name} partition (months {months[0]}-{months[-1]}) ...")
        split_events = all_events[all_events["month_idx"].isin(months)]
        windows = build_partition_windows(
            split_events, user_to_idx, resource_to_idx, resource_vocab, window_hours
        )
        out_path = os.path.join(processed_dir, f"{split_name}_windows.pt")
        torch.save(windows, out_path)
        n_mal = sum(1 for w in windows if w["label"] == 1)
        print(f"  Saved {len(windows)} windows ({n_mal} malicious) -> {out_path}")

    # LOIO folds
    print("\nBuilding LOIO folds ...")
    train_windows = torch.load(os.path.join(processed_dir, "train_windows.pt"))
    val_windows   = torch.load(os.path.join(processed_dir, "val_windows.pt"))
    test_windows  = torch.load(os.path.join(processed_dir, "test_windows.pt"))
    all_windows   = train_windows + val_windows + test_windows
    folds = loio_folds(all_windows)
    for i, fold in enumerate(folds, 1):
        fold_path = os.path.join(processed_dir, "loio_folds", f"fold_{i}.pt")
        torch.save(fold, fold_path)
        print(f"  Fold {i} saved -> {fold_path}")

    print("\nPreprocessing complete.")


def build_partition_windows(events, user_to_idx, resource_to_idx,
                            resource_vocab, window_hours):
    """Build per-user per-window feature dicts for a given set of events."""
    # Ground-truth malicious windows: loaded from CERT r6.2 annotations
    # In the actual dataset these are labeled in the answers/ directory.
    # Here we construct a placeholder label field; real labels must be
    # joined from the CERT r6.2 ground-truth file before use.
    windows = []
    for (user, w_idx), group in events.groupby(["user", "window_idx"]):
        kt = keystroke_proxy_features(group)
        mt = mouse_proxy_features(group)
        app_seq = application_usage_sequence(group, resource_vocab)
        edges = build_adjacency(group, user_to_idx, resource_to_idx)
        windows.append({
            "user": user,
            "user_idx": user_to_idx.get(user, -1),
            "window_idx": int(w_idx),
            "keystroke_features": torch.tensor(kt, dtype=torch.float32),
            "mouse_features":     torch.tensor(mt, dtype=torch.float32),
            "app_sequence":       torch.tensor(app_seq, dtype=torch.long),
            "edges":              edges,
            "label":              0,   # placeholder: join ground truth before training
        })
    return windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/data_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    preprocess(config)
