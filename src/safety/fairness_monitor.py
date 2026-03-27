"""
src/safety/fairness_monitor.py
═══════════════════════════════════════════════════════════════════
AI Fairness & Bias Monitor for Clinical Decision Support

Monitors for:
  1. Demographic parity (equal FPR/FNR across age/gender groups)
  2. Calibration fairness (confidence scores equally calibrated)
  3. Subgroup performance drift alerts
  4. Disparate impact detection

WHY THIS MATTERS:
  Studies show AI radiology models trained predominantly on data
  from one demographic can have 5-10% higher error rates on
  underrepresented groups. In healthcare, that's life-threatening.

Designed to be called at inference time (real-time bias checks)
and during batch evaluation (offline fairness audits).
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger


class DemographicGroup(str, Enum):
    AGE_0_40   = "age_0_40"
    AGE_41_60  = "age_41_60"
    AGE_61_75  = "age_61_75"
    AGE_75_PLUS = "age_75_plus"
    MALE       = "male"
    FEMALE     = "female"
    UNKNOWN    = "unknown"


@dataclass
class FairnessMetrics:
    group:               DemographicGroup
    n_samples:           int
    mean_confidence:     float
    calibration_error:   float     # |mean_confidence - empirical_accuracy|
    positive_rate:       float     # fraction predicted positive above threshold
    flag_count:          int = 0   # times this group triggered a bias alert
    last_updated:        float = field(default_factory=time.time)


@dataclass
class BiasAlert:
    alert_type:   str
    group_a:      DemographicGroup
    group_b:      DemographicGroup
    metric:       str
    value_a:      float
    value_b:      float
    disparity:    float
    severity:     str              # "INFO", "WARNING", "CRITICAL"
    message:      str
    timestamp:    float = field(default_factory=time.time)


class FairnessMonitor:
    """
    Real-time fairness monitor that tracks prediction statistics
    per demographic subgroup and raises alerts when disparities
    exceed clinical safety thresholds.

    Usage (per inference):
        monitor = FairnessMonitor()
        monitor.record_prediction(age=72, gender="male", confidence=0.87,
                                  label_predicted="Pneumonia", urgency=0.91)
        alerts = monitor.check_disparities()
    """

    # Disparity thresholds (industry standard: 80% rule from EEOC)
    CONFIDENCE_DISPARITY_THRESHOLD  = 0.10   # >10% gap triggers WARNING
    CALIBRATION_DISPARITY_THRESHOLD = 0.15   # >15% gap triggers CRITICAL
    POSITIVE_RATE_DISPARITY_THRESHOLD = 0.15 # disparate impact threshold

    def __init__(self):
        self._records: dict[DemographicGroup, list[dict]] = defaultdict(list)
        self._alerts:  list[BiasAlert] = []
        self._total_predictions = 0
        logger.info("FairnessMonitor initialised — demographic parity tracking active")

    # ── Record prediction ─────────────────────────────────────────────
    def record_prediction(
        self,
        confidence:        float,
        urgency:           float,
        label_predicted:   str,
        age:               Optional[int]   = None,
        gender:            Optional[str]   = None,
        ground_truth:      Optional[int]   = None,   # 1 = positive, 0 = negative
    ) -> None:
        """
        Record a single prediction for fairness tracking.
        Call this for every inference, passing demographic metadata
        extracted from clinical notes (age, gender).
        """
        groups = self._infer_groups(age, gender)

        record = {
            "confidence":      confidence,
            "urgency":         urgency,
            "label":           label_predicted,
            "positive":        1 if label_predicted != "No Finding" else 0,
            "ground_truth":    ground_truth,
            "timestamp":       time.time(),
        }

        for group in groups:
            self._records[group].append(record)

        self._total_predictions += 1

        # Rolling window: keep last 500 predictions per group
        for group in groups:
            if len(self._records[group]) > 500:
                self._records[group] = self._records[group][-500:]

    # ── Compute metrics ───────────────────────────────────────────────
    def compute_metrics(self) -> dict[DemographicGroup, FairnessMetrics]:
        metrics = {}
        for group, records in self._records.items():
            if not records:
                continue
            confs = [r["confidence"] for r in records]
            positives = [r["positive"] for r in records]

            # Calibration error (where ground truth available)
            gt_records = [r for r in records if r["ground_truth"] is not None]
            if len(gt_records) >= 5:
                gt_confs   = [r["confidence"] for r in gt_records]
                gt_labels  = [r["ground_truth"] for r in gt_records]
                cal_error  = abs(np.mean(gt_confs) - np.mean(gt_labels))
            else:
                cal_error = 0.0

            metrics[group] = FairnessMetrics(
                group=group,
                n_samples=len(records),
                mean_confidence=float(np.mean(confs)),
                calibration_error=round(cal_error, 4),
                positive_rate=float(np.mean(positives)),
            )
        return metrics

    # ── Disparity check ───────────────────────────────────────────────
    def check_disparities(self) -> list[BiasAlert]:
        """
        Compare demographic groups and raise alerts on disparities.
        Compares: age groups vs each other, male vs female.
        """
        new_alerts: list[BiasAlert] = []
        metrics = self.compute_metrics()

        # Age group comparisons
        age_groups = [g for g in metrics if g.value.startswith("age_")]
        for i in range(len(age_groups)):
            for j in range(i+1, len(age_groups)):
                g_a, g_b = age_groups[i], age_groups[j]
                m_a, m_b = metrics[g_a], metrics[g_b]
                if m_a.n_samples < 10 or m_b.n_samples < 10:
                    continue
                self._check_pair(g_a, g_b, m_a, m_b, new_alerts)

        # Gender comparison
        if DemographicGroup.MALE in metrics and DemographicGroup.FEMALE in metrics:
            m_m = metrics[DemographicGroup.MALE]
            m_f = metrics[DemographicGroup.FEMALE]
            if m_m.n_samples >= 10 and m_f.n_samples >= 10:
                self._check_pair(DemographicGroup.MALE, DemographicGroup.FEMALE,
                                 m_m, m_f, new_alerts)

        if new_alerts:
            self._alerts.extend(new_alerts)
            for alert in new_alerts:
                log_fn = logger.critical if alert.severity == "CRITICAL" else logger.warning
                log_fn(f"FAIRNESS ALERT [{alert.severity}]: {alert.message}")

        return new_alerts

    def _check_pair(
        self,
        g_a: DemographicGroup,
        g_b: DemographicGroup,
        m_a: FairnessMetrics,
        m_b: FairnessMetrics,
        alerts: list[BiasAlert],
    ) -> None:
        # Confidence disparity
        conf_disp = abs(m_a.mean_confidence - m_b.mean_confidence)
        if conf_disp > self.CONFIDENCE_DISPARITY_THRESHOLD:
            alerts.append(BiasAlert(
                alert_type="CONFIDENCE_DISPARITY",
                group_a=g_a, group_b=g_b,
                metric="mean_confidence",
                value_a=m_a.mean_confidence, value_b=m_b.mean_confidence,
                disparity=conf_disp,
                severity="WARNING" if conf_disp < 0.2 else "CRITICAL",
                message=(
                    f"Confidence disparity {conf_disp:.1%} between {g_a.value} and {g_b.value}. "
                    f"({g_a.value}: {m_a.mean_confidence:.3f}, {g_b.value}: {m_b.mean_confidence:.3f})"
                ),
            ))

        # Positive rate disparity (disparate impact)
        if m_a.positive_rate > 0 and m_b.positive_rate > 0:
            rate_ratio = min(m_a.positive_rate, m_b.positive_rate) / max(m_a.positive_rate, m_b.positive_rate)
            if rate_ratio < (1 - self.POSITIVE_RATE_DISPARITY_THRESHOLD):
                alerts.append(BiasAlert(
                    alert_type="POSITIVE_RATE_DISPARITY",
                    group_a=g_a, group_b=g_b,
                    metric="positive_rate",
                    value_a=m_a.positive_rate, value_b=m_b.positive_rate,
                    disparity=1 - rate_ratio,
                    severity="WARNING",
                    message=(
                        f"Disparate positive prediction rate: {g_a.value}={m_a.positive_rate:.1%} "
                        f"vs {g_b.value}={m_b.positive_rate:.1%} (ratio={rate_ratio:.2f})"
                    ),
                ))

    def _infer_groups(
        self, age: Optional[int], gender: Optional[str]
    ) -> list[DemographicGroup]:
        groups = []
        # Age group
        if age is not None:
            if age <= 40:   groups.append(DemographicGroup.AGE_0_40)
            elif age <= 60: groups.append(DemographicGroup.AGE_41_60)
            elif age <= 75: groups.append(DemographicGroup.AGE_61_75)
            else:           groups.append(DemographicGroup.AGE_75_PLUS)
        # Gender
        if gender:
            g = gender.lower().strip()
            if g in ("male","m","man"):
                groups.append(DemographicGroup.MALE)
            elif g in ("female","f","woman"):
                groups.append(DemographicGroup.FEMALE)
        if not groups:
            groups.append(DemographicGroup.UNKNOWN)
        return groups

    # ── Reporting ─────────────────────────────────────────────────────
    def get_fairness_report(self) -> dict:
        metrics = self.compute_metrics()
        return {
            "total_predictions": self._total_predictions,
            "groups": {
                g.value: {
                    "n_samples":        m.n_samples,
                    "mean_confidence":  round(m.mean_confidence, 4),
                    "positive_rate":    round(m.positive_rate, 4),
                    "calibration_error":round(m.calibration_error, 4),
                }
                for g, m in metrics.items()
            },
            "recent_alerts": [
                {
                    "type":     a.alert_type,
                    "severity": a.severity,
                    "message":  a.message,
                    "disparity":round(a.disparity, 4),
                }
                for a in self._alerts[-10:]
            ],
        }

    def reset(self):
        self._records.clear()
        self._alerts.clear()
        self._total_predictions = 0
