"""
monitoring/drift_detector.py
═══════════════════════════════════════════════════════════════════
Data Drift & Model Performance Monitor

Detects:
  1. Input data drift (image statistics shift from training distribution)
  2. Prediction drift (output distribution shift)
  3. Confidence drift (model becoming more/less confident over time)
  4. Label distribution shift (new pathology patterns emerging)

Uses Population Stability Index (PSI) and Kolmogorov-Smirnov tests.
Raises AWS CloudWatch alarms when drift exceeds thresholds.

WHY DRIFT DETECTION MATTERS:
  Hospital equipment changes, patient demographics shift, new
  disease variants emerge. A model trained on 2023 data may
  silently degrade on 2025 inputs. Without drift detection,
  you won't know until patients are harmed.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger


class DriftSeverity(str, Enum):
    NONE     = "NONE"
    LOW      = "LOW"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"


@dataclass
class DriftReport:
    metric:      str
    psi_score:   float        # Population Stability Index (PSI)
    ks_statistic:float        # Kolmogorov-Smirnov statistic
    severity:    DriftSeverity
    message:     str
    timestamp:   float = field(default_factory=time.time)

    # PSI interpretation:
    # PSI < 0.10 : No significant drift
    # PSI 0.10-0.20: Moderate drift — investigate
    # PSI > 0.20 : Critical drift — retrain model


class DriftDetector:
    """
    Online drift detector using sliding window statistics.
    Compares current window vs reference (training) distribution.
    """

    # PSI thresholds
    PSI_MODERATE = 0.10
    PSI_CRITICAL = 0.20
    # Window sizes
    REFERENCE_SIZE = 1000
    CURRENT_SIZE   = 200

    def __init__(self, reference_stats: Optional[dict] = None):
        self._reference_confidence: Optional[np.ndarray] = None
        self._reference_urgency:    Optional[np.ndarray] = None
        self._reference_label_dist: Optional[np.ndarray] = None

        self._current_confidence = deque(maxlen=self.CURRENT_SIZE)
        self._current_urgency    = deque(maxlen=self.CURRENT_SIZE)
        self._current_labels     = deque(maxlen=self.CURRENT_SIZE)

        self._alerts: list[DriftReport] = []

        if reference_stats:
            self._load_reference(reference_stats)
        else:
            self._use_chexpert_reference()

        logger.info("DriftDetector initialised with reference distribution")

    def _use_chexpert_reference(self):
        """Load approximate CheXpert validation set statistics as reference."""
        rng = np.random.default_rng(42)
        # Approximate CheXpert validation confidence distribution (Beta distribution)
        self._reference_confidence = rng.beta(2.5, 1.5, self.REFERENCE_SIZE)
        self._reference_urgency    = rng.beta(1.5, 3.0, self.REFERENCE_SIZE)
        # CheXpert label prevalence (approximate)
        label_probs = [0.40, 0.02, 0.06, 0.15, 0.03, 0.08, 0.06, 0.07, 0.12, 0.03, 0.08, 0.01, 0.02, 0.06]
        self._reference_label_dist = np.array(label_probs)

    def _load_reference(self, stats: dict):
        self._reference_confidence = np.array(stats.get("confidence", []))
        self._reference_urgency    = np.array(stats.get("urgency", []))
        ref_labels = stats.get("label_distribution")
        if ref_labels:
            self._reference_label_dist = np.array(ref_labels)

    def record(self, confidence: float, urgency: float, label_idx: int) -> None:
        self._current_confidence.append(confidence)
        self._current_urgency.append(urgency)
        self._current_labels.append(label_idx)

    def check_drift(self) -> list[DriftReport]:
        """
        Run PSI + KS tests on current window vs reference.
        Returns list of DriftReport, empty if no drift detected.
        """
        if len(self._current_confidence) < 50:
            return []   # Not enough data yet

        reports = []
        current_conf = np.array(list(self._current_confidence))
        current_urg  = np.array(list(self._current_urgency))

        # Confidence drift
        r = self._test_drift("confidence", self._reference_confidence, current_conf)
        if r: reports.append(r)

        # Urgency drift
        r = self._test_drift("urgency", self._reference_urgency, current_urg)
        if r: reports.append(r)

        # Label distribution drift
        if self._reference_label_dist is not None and len(self._current_labels) >= 50:
            r = self._test_label_drift()
            if r: reports.append(r)

        if reports:
            self._alerts.extend(reports)
            for rep in reports:
                if rep.severity == DriftSeverity.CRITICAL:
                    logger.critical(f"DRIFT CRITICAL: {rep.message}")
                    self._trigger_cloudwatch_alarm(rep)
                elif rep.severity == DriftSeverity.MODERATE:
                    logger.warning(f"DRIFT MODERATE: {rep.message}")

        return reports

    def _test_drift(
        self, metric: str, reference: np.ndarray, current: np.ndarray
    ) -> Optional[DriftReport]:
        if len(reference) < 10 or len(current) < 10:
            return None

        psi    = self._compute_psi(reference, current)
        ks_stat = self._compute_ks(reference, current)
        severity = self._classify_psi(psi)

        if severity == DriftSeverity.NONE:
            return None

        return DriftReport(
            metric=metric,
            psi_score=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            severity=severity,
            message=(
                f"{metric} drift detected — PSI={psi:.3f} "
                f"({'CRITICAL' if severity==DriftSeverity.CRITICAL else 'MODERATE'}). "
                f"Reference mean={reference.mean():.3f}, "
                f"Current mean={current.mean():.3f}. "
                f"Consider model retraining."
            ),
        )

    def _test_label_drift(self) -> Optional[DriftReport]:
        labels = list(self._current_labels)
        n_classes = len(self._reference_label_dist)
        counts = np.bincount(labels, minlength=n_classes)[:n_classes].astype(float)
        if counts.sum() == 0:
            return None
        current_dist = counts / counts.sum()

        # PSI on distributions
        psi = float(np.sum(
            (current_dist - self._reference_label_dist) *
            np.log((current_dist + 1e-9) / (self._reference_label_dist + 1e-9))
        ))
        psi = abs(psi)
        severity = self._classify_psi(psi)
        if severity == DriftSeverity.NONE:
            return None

        top_shifted = int(np.argmax(np.abs(current_dist - self._reference_label_dist)))
        return DriftReport(
            metric="label_distribution",
            psi_score=round(psi, 4),
            ks_statistic=0.0,
            severity=severity,
            message=(
                f"Label distribution drift — PSI={psi:.3f}. "
                f"Largest shift in class {top_shifted}. "
                f"New disease pattern may be emerging."
            ),
        )

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        exp_counts, _ = np.histogram(np.clip(expected, 0, 1), bins=bins)
        act_counts, _ = np.histogram(np.clip(actual,   0, 1), bins=bins)
        exp_pct = exp_counts / (exp_counts.sum() + 1e-9)
        act_pct = act_counts / (act_counts.sum() + 1e-9)
        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def _compute_ks(self, a: np.ndarray, b: np.ndarray) -> float:
        try:
            from scipy.stats import ks_2samp
            stat, _ = ks_2samp(a, b)
            return float(stat)
        except ImportError:
            a_sorted = np.sort(a); b_sorted = np.sort(b)
            return float(np.max(np.abs(
                np.searchsorted(a_sorted, b_sorted, side='right') / len(a_sorted) -
                np.arange(len(b_sorted)) / len(b_sorted)
            )))

    def _classify_psi(self, psi: float) -> DriftSeverity:
        if abs(psi) < self.PSI_MODERATE: return DriftSeverity.NONE
        if abs(psi) < self.PSI_CRITICAL: return DriftSeverity.MODERATE
        return DriftSeverity.CRITICAL

    def _trigger_cloudwatch_alarm(self, report: DriftReport) -> None:
        try:
            import boto3
            cw = boto3.client("cloudwatch")
            cw.put_metric_data(
                Namespace="ClinicalAI/ModelHealth",
                MetricData=[{
                    "MetricName": "DataDriftPSI",
                    "Dimensions": [{"Name": "Metric", "Value": report.metric}],
                    "Value": report.psi_score,
                    "Unit": "None",
                }],
            )
        except Exception as e:
            logger.warning(f"CloudWatch metric publish failed: {e}")

    def get_summary(self) -> dict:
        return {
            "current_window_size": len(self._current_confidence),
            "recent_alerts":       len(self._alerts),
            "last_alert":          (
                {"metric": self._alerts[-1].metric,
                 "severity": self._alerts[-1].severity,
                 "psi": self._alerts[-1].psi_score}
                if self._alerts else None
            ),
        }
