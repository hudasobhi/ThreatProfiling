"""
interface.py -- Triage user interface for the analyst simulation (Section 6.8).

Presents 20 alerts (10 TP + 10 FP) to participants either with or without
the Shapley attribution panel. Records wall-clock triage time and disposition
for each alert.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Alert:
    alert_id:        int
    window_idx:      int
    user_idx:        int
    anomaly_score:   float
    ground_truth:    int                  # 0=benign, 1=malicious
    shap_values:     Optional[np.ndarray] # None in no-attribution condition
    top_k_features:  Optional[List[str]]  # human-readable feature names


@dataclass
class TriageRecord:
    participant_id:  int
    alert_id:        int
    condition:       str          # "attribution" or "no_attribution"
    triage_time_s:   float
    disposition:     int          # 0=dismiss, 1=escalate
    ground_truth:    int


class TriageInterface:
    """
    Simulated triage interface. In the study, this was a purpose-built web
    UI; here it runs in a terminal for reproducibility.

    Escalation criterion (as given in participant rubric):
        Any flagged session where at least 2 behavioral deviations exceed
        1.5 standard deviations from that user's 30-day baseline.
    """

    def __init__(self, participant_id: int, condition: str):
        assert condition in ("attribution", "no_attribution")
        self.participant_id = participant_id
        self.condition      = condition
        self.records: List[TriageRecord] = []

    def present_alert(self, alert: Alert) -> TriageRecord:
        print(f"\n--- Alert {alert.alert_id} ---")
        print(f"  Anomaly score: {alert.anomaly_score:.4f}")
        if self.condition == "attribution" and alert.shap_values is not None:
            print("  Top contributing features:")
            for i, feat in enumerate(alert.top_k_features or [], 1):
                val = alert.shap_values[i - 1] if alert.shap_values is not None else 0
                print(f"    {i}. {feat:40s}  SHAP={val:+.4f}")
        else:
            print("  [No attribution panel in this condition]")

        t_start = time.perf_counter()
        decision = input("  Escalate (1) or Dismiss (0)? ").strip()
        t_end   = time.perf_counter()

        disposition = int(decision) if decision in ("0", "1") else 0
        record = TriageRecord(
            participant_id = self.participant_id,
            alert_id       = alert.alert_id,
            condition      = self.condition,
            triage_time_s  = t_end - t_start,
            disposition    = disposition,
            ground_truth   = alert.ground_truth,
        )
        self.records.append(record)
        return record
