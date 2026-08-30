"""
splits.py — Train/val/test chronological split and LOIO fold generation.

Partition design (manuscript Section 6.1):
    Training:   months 1-13  (72.2% of observation period)
    Validation: months 14-15 (11.1%)
    Test:       months 16-18 (16.7%)

Partitions are temporally contiguous. Each malicious identity belongs to
exactly one partition. TGN memory is reset to zero at each boundary.

LOIO cross-validation: 5 folds, one per malicious insider identity in CERT r6.2.
"""

from typing import List, Dict, Tuple


# CERT r6.2 malicious insider identity labels (anonymized)
# Each identity maps to its primary activity partition.
MALICIOUS_IDENTITY_PARTITION = {
    "CERT-M01": "train",
    "CERT-M02": "train",
    "CERT-M03": "train",
    "CERT-M04": "val",
    "CERT-M05": "test",
}


def chronological_split(windows: list,
                         train_months: list,
                         val_months:   list,
                         test_months:  list) -> Tuple[list, list, list]:
    """
    Split a flat list of window dicts by month index.
    Returns (train_windows, val_windows, test_windows).
    """
    train, val, test = [], [], []
    month_to_split = {}
    for m in train_months: month_to_split[m] = "train"
    for m in val_months:   month_to_split[m] = "val"
    for m in test_months:  month_to_split[m] = "test"

    for w in windows:
        month = w.get("month_idx", -1)
        split = month_to_split.get(month, None)
        if split == "train":   train.append(w)
        elif split == "val":   val.append(w)
        elif split == "test":  test.append(w)

    return train, val, test


def loio_folds(all_windows: list) -> List[Dict]:
    """
    Generate 5 leave-one-insider-out folds (Table 10).
    Each fold holds out one malicious identity as the test set;
    the remaining four identities' partitions form the training set.

    Returns a list of 5 dicts, each with keys:
        'fold_id'         : int (1-5)
        'held_out_identity': str (e.g., 'CERT-M01')
        'train_windows'   : list
        'test_windows'    : list
    """
    all_identities = list(MALICIOUS_IDENTITY_PARTITION.keys())
    folds = []

    for fold_id, held_out in enumerate(all_identities, 1):
        train_ws = [w for w in all_windows
                    if w.get("malicious_identity", None) != held_out]
        test_ws  = [w for w in all_windows
                    if w.get("malicious_identity", None) == held_out
                    or w.get("label", 0) == 0]  # benign windows always in test

        folds.append({
            "fold_id":            fold_id,
            "held_out_identity":  held_out,
            "train_windows":      train_ws,
            "test_windows":       test_ws,
        })

    return folds
