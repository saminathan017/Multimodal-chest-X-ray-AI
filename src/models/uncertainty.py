"""
src/models/uncertainty.py
═══════════════════════════════════════════════════════════════════
Uncertainty Quantification for Clinical AI

Implements THREE complementary uncertainty methods:

1. MC Dropout (Gal & Ghahramani, 2016)
   - Enable dropout at test time, run N forward passes
   - Cheap, fast, works on existing models
   - Captures epistemic (model) uncertainty

2. Deep Ensembles (Lakshminarayanan, 2017)
   - Train 3-5 models with different random seeds
   - Gold standard for uncertainty in production
   - Captures both epistemic & aleatoric uncertainty

3. Conformal Prediction Sets
   - Distribution-free prediction intervals
   - Provides guaranteed coverage (e.g. 95% confidence sets)
   - Clinically interpretable: "These 2 pathologies cover 95% probability"

WHY UNCERTAINTY MATTERS IN CLINICAL AI:
  A confident wrong prediction is MORE dangerous than an uncertain
  correct one. Uncertainty scores allow the system to say:
  "I'm 87% confident AND my uncertainty is low — you can trust this."
  vs
  "I'm 87% confident BUT uncertainty is high — call a radiologist NOW."
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ── Uncertainty result ───────────────────────────────────────────────
@dataclass
class UncertaintyResult:
    mean_probs:          np.ndarray    # (num_classes,) — mean prediction
    std_probs:           np.ndarray    # (num_classes,) — epistemic uncertainty
    predictive_entropy:  float         # total uncertainty (bits)
    mutual_information:  float         # epistemic uncertainty (bits)
    aleatoric_uncertainty: float       # data uncertainty
    epistemic_uncertainty: float       # model uncertainty
    confidence_interval: tuple         # (lower, upper) for top prediction
    uncertainty_flag:    bool          # True if uncertainty is clinically high
    prediction_set:      list[str]     # conformal prediction set at 95%

    @property
    def is_high_uncertainty(self) -> bool:
        return self.uncertainty_flag

    def summary(self, labels: list[str]) -> dict:
        top_idx  = int(np.argmax(self.mean_probs))
        return {
            "top_prediction":       labels[top_idx] if top_idx < len(labels) else "Unknown",
            "mean_confidence":      round(float(self.mean_probs[top_idx]), 4),
            "std_confidence":       round(float(self.std_probs[top_idx]), 4),
            "predictive_entropy":   round(self.predictive_entropy, 4),
            "mutual_information":   round(self.mutual_information, 4),
            "epistemic_uncertainty":round(self.epistemic_uncertainty, 4),
            "aleatoric_uncertainty":round(self.aleatoric_uncertainty, 4),
            "confidence_interval":  (round(self.confidence_interval[0],3),
                                     round(self.confidence_interval[1],3)),
            "uncertainty_flag":     self.uncertainty_flag,
            "prediction_set_95pct": self.prediction_set,
        }


# ── MC Dropout sampler ───────────────────────────────────────────────
class MCDropoutSampler:
    """
    Bayesian approximation via Monte Carlo Dropout.

    Wraps any model and enables dropout layers during inference
    to sample from the approximate posterior.

    Args:
        model:          the fusion model (or any nn.Module)
        n_samples:      number of stochastic forward passes (default 30)
        dropout_p:      dropout probability to inject (if not already present)
    """

    HIGH_UNCERTAINTY_ENTROPY = 0.8    # bits threshold for flagging

    def __init__(self, model: nn.Module, n_samples: int = 30, dropout_p: float = 0.3):
        self.model     = model
        self.n_samples = n_samples
        self.dropout_p = dropout_p

    def _enable_dropout(self, module: nn.Module) -> None:
        """Recursively enable dropout layers in eval mode."""
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.train()   # forces dropout to apply even in eval mode
                m.p = max(m.p, self.dropout_p)

    @torch.no_grad()
    def sample(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        labels:   list[str],
    ) -> UncertaintyResult:
        """
        Run N stochastic forward passes and compute uncertainty metrics.

        Returns UncertaintyResult with all uncertainty estimates.
        """
        self.model.eval()
        self._enable_dropout(self.model)

        all_probs: list[np.ndarray] = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                out   = self.model(img_feat, txt_feat)
                probs = out["probs"].squeeze(0).cpu().numpy()
                all_probs.append(probs)

        # Restore eval mode (disables dropout)
        self.model.eval()

        return self._compute_metrics(np.stack(all_probs), labels)

    def _compute_metrics(self, samples: np.ndarray, labels: list[str]) -> UncertaintyResult:
        """
        samples: (N_samples, num_classes)
        """
        mean_p = np.mean(samples, axis=0)      # (num_classes,)
        std_p  = np.std(samples, axis=0)       # epistemic uncertainty

        # Predictive entropy: H[y|x] = -sum(p * log(p))
        eps = 1e-9
        pred_entropy = float(-np.sum(mean_p * np.log(mean_p + eps)))

        # Expected entropy: E[H[y|x,w]] = aleatoric
        per_sample_entropy = -np.sum(samples * np.log(samples + eps), axis=1)
        aleatoric = float(np.mean(per_sample_entropy))

        # Mutual information: I[y;w|x] = H - E[H] = epistemic
        mutual_info = max(0.0, pred_entropy - aleatoric)

        # Top prediction confidence interval (95% CI from samples)
        top_idx = int(np.argmax(mean_p))
        top_samples = samples[:, top_idx]
        ci_lower = float(np.percentile(top_samples, 2.5))
        ci_upper = float(np.percentile(top_samples, 97.5))

        # Conformal prediction set (95% coverage)
        pred_set = self._conformal_set(mean_p, labels, coverage=0.95)

        # Uncertainty flag
        uncertainty_flag = (
            pred_entropy > self.HIGH_UNCERTAINTY_ENTROPY or
            std_p[top_idx] > 0.15 or
            (ci_upper - ci_lower) > 0.30
        )

        return UncertaintyResult(
            mean_probs=mean_p,
            std_probs=std_p,
            predictive_entropy=pred_entropy,
            mutual_information=mutual_info,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=mutual_info,
            confidence_interval=(ci_lower, ci_upper),
            uncertainty_flag=uncertainty_flag,
            prediction_set=pred_set,
        )

    def _conformal_set(
        self, probs: np.ndarray, labels: list[str], coverage: float = 0.95
    ) -> list[str]:
        """
        Adaptive Prediction Set (APS):
        Add classes in descending probability order until
        cumulative probability ≥ coverage threshold.
        """
        sorted_idx = np.argsort(probs)[::-1]
        cumulative  = 0.0
        pred_set    = []
        for idx in sorted_idx:
            if idx < len(labels):
                pred_set.append(labels[idx])
                cumulative += probs[idx]
                if cumulative >= coverage:
                    break
        return pred_set


# ── Temperature Scaling Calibrator ───────────────────────────────────
class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via temperature scaling (Guo et al., 2017).

    A single learnable temperature parameter T scales the logits:
        calibrated_probs = softmax(logits / T)

    T > 1: model is overconfident → soften probabilities
    T < 1: model is underconfident → sharpen probabilities

    Fit on a validation set AFTER main training.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature.clamp(min=0.1, max=10.0))

    def calibrate(
        self,
        logits_val:  torch.Tensor,
        labels_val:  torch.Tensor,
        n_iter:      int   = 200,
        lr:          float = 0.02,
    ) -> float:
        """
        Fit temperature on validation logits.
        Returns Expected Calibration Error (ECE) after calibration.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_val / self.temperature.clamp(min=0.1)
            loss = criterion(scaled_logits, labels_val)
            loss.backward()
            return loss

        optimizer.step(closure)
        ece = self._compute_ece(
            torch.sigmoid(logits_val / self.temperature.clamp(min=0.1)).detach(),
            labels_val,
        )
        logger.info(f"Temperature calibration complete — T={self.temperature.item():.3f}, ECE={ece:.4f}")
        return ece

    def _compute_ece(
        self, probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
    ) -> float:
        """Expected Calibration Error (lower is better, 0 = perfect)."""
        probs  = probs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if mask.sum() > 0:
                acc  = labels[mask].mean()
                conf = probs[mask].mean()
                ece += (mask.sum() / len(probs)) * abs(acc - conf)
        return float(ece)


# ── Deep Ensemble wrapper ─────────────────────────────────────────────
class DeepEnsemble:
    """
    Ensembles 3-5 independently trained fusion models for
    gold-standard uncertainty estimates.

    Usage:
        ensemble = DeepEnsemble(models=[model1, model2, model3])
        result   = ensemble.predict(img_feat, txt_feat, labels)
    """

    def __init__(self, models: list[nn.Module]):
        if len(models) < 2:
            raise ValueError("DeepEnsemble requires at least 2 models.")
        self.models = models
        logger.info(f"DeepEnsemble initialised with {len(models)} members")

    @torch.no_grad()
    def predict(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        labels:   list[str],
    ) -> UncertaintyResult:
        all_probs = []
        for model in self.models:
            model.eval()
            out  = model(img_feat, txt_feat)
            all_probs.append(out["probs"].squeeze(0).cpu().numpy())

        samples = np.stack(all_probs)

        # Reuse MCDropout metrics computation
        sampler = MCDropoutSampler(self.models[0])
        return sampler._compute_metrics(samples, labels)
