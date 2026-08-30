"""
run_triage.py -- Runs the analyst triage simulation (Section 6.8, Figure 14).

Protocol (manuscript Section 6.8):
    - 10 graduate security researcher proxies
    - 20 alerts: 10 TP + 10 FP, stratified random sample from test partition
    - Within-subject counterbalanced design (5 attribution-first, 5 no-attribution-first)
    - 30-minute orientation + 5-alert practice block (validation partition)
    - Practice times NOT recorded
    - Primary outcomes: mean triage time per alert and disposition accuracy

Usage:
    python triage/run_triage.py --alerts_path outputs/triage_alerts.pt
"""

import argparse, os, json, random
import numpy as np
import torch

from triage.interface import TriageInterface, Alert


N_PARTICIPANTS   = 10
N_ALERTS         = 20       # 10 TP + 10 FP
N_PRACTICE       = 5
ATTR_FIRST_IDS   = [1, 2, 3, 4, 5]
NO_ATTR_FIRST_IDS= [6, 7, 8, 9, 10]


def load_alerts(alerts_path: str):
    """Load pre-computed alerts from the test partition evaluation."""
    if not os.path.exists(alerts_path):
        print(f"Alerts file not found at {alerts_path}.")
        print("Generate alerts by running scripts/evaluate.py first.")
        return None
    return torch.load(alerts_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alerts_path", default="outputs/triage_alerts.pt")
    parser.add_argument("--output",      default="outputs/")
    args = parser.parse_args()

    alerts = load_alerts(args.alerts_path)
    if alerts is None:
        return

    all_records = []
    for pid in range(1, N_PARTICIPANTS + 1):
        order = (["attribution", "no_attribution"] if pid in ATTR_FIRST_IDS
                 else ["no_attribution", "attribution"])
        shuffled_alerts = random.sample(alerts, len(alerts))
        for condition in order:
            iface = TriageInterface(pid, condition)
            for alert in shuffled_alerts[:N_ALERTS]:
                record = iface.present_alert(alert)
                all_records.append(record)

    # Save raw records
    os.makedirs(args.output, exist_ok=True)
    import csv
    from dataclasses import asdict
    rows = [asdict(r) for r in all_records]
    with open(os.path.join(args.output, "triage_records.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(all_records)} triage records to {args.output}triage_records.csv")
    print("Run triage/analysis.py to compute statistics and generate Figure 14.")

if __name__ == "__main__": main()
