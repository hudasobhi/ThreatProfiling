"""
dataset.py — PyTorch Dataset class for ShadowDetectAI window-level data.
"""

from torch.utils.data import Dataset
import torch


class CERTWindowDataset(Dataset):
    """
    Dataset of per-user per-window behavioral feature dicts produced by preprocess.py.

    Each item contains:
        keystroke_features : FloatTensor [8]   — proxy timing statistics (Eq. 13)
        mouse_features     : FloatTensor [6]   — entropy and interval statistics (Eq. 14)
        app_sequence       : LongTensor  [L]   — categorical resource access sequence (Eq. 15)
        user_idx           : int               — node index in the dynamic graph
        window_idx         : int               — temporal window index
        label              : int               — 0 benign / 1 malicious
    """

    def __init__(self, windows: list):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {
            "keystroke_features": w["keystroke_features"],
            "mouse_features":     w["mouse_features"],
            "app_sequence":       w["app_sequence"],
            "user_idx":           w["user_idx"],
            "window_idx":         w["window_idx"],
            "label":              torch.tensor(w["label"], dtype=torch.long),
        }

    def get_malicious_indices(self):
        return [i for i, w in enumerate(self.windows) if w["label"] == 1]

    def get_benign_indices(self):
        return [i for i, w in enumerate(self.windows) if w["label"] == 0]


def collate_fn(batch):
    """
    Custom collate for variable-length app_sequence tensors.
    Pads app sequences to the maximum length in the batch.
    """
    max_len = max(item["app_sequence"].shape[0] for item in batch)
    padded_seqs = torch.zeros(len(batch), max_len, dtype=torch.long)
    seq_lengths = []
    for i, item in enumerate(batch):
        seq = item["app_sequence"]
        padded_seqs[i, :len(seq)] = seq
        seq_lengths.append(len(seq))

    return {
        "keystroke_features": torch.stack([item["keystroke_features"] for item in batch]),
        "mouse_features":     torch.stack([item["mouse_features"]     for item in batch]),
        "app_sequence":       padded_seqs,
        "seq_lengths":        torch.tensor(seq_lengths, dtype=torch.long),
        "user_idx":           torch.tensor([item["user_idx"]   for item in batch]),
        "window_idx":         torch.tensor([item["window_idx"] for item in batch]),
        "label":              torch.stack([item["label"]        for item in batch]),
    }
