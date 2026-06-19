#!/usr/bin/env python3
"""
Step 3 (ML): Hybrid model scaffold for ICF prediction.

Architecture:
- Expert MLP branch for HRV/tabular features
- Transformer-style sensor branch for EDA/IMU with explicit cross-attention
- Fusion head for final ICF score regression (0-100)

Includes theory-informed loss scaffold and functional capacity equation helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertHRVMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"HRV input_dim must be > 0, got {input_dim}")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_hrv: torch.Tensor) -> torch.Tensor:
        return self.network(x_hrv)


class EDACrossIMUEncoder(nn.Module):
    """
    Transformer-like sensor encoder with cross-attention:
    - EDA token attends to IMU token
    - IMU token attends to EDA token
    """

    def __init__(
        self,
        token_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be > 0, got {token_dim}")

        self.eda_projection = nn.Linear(token_dim, model_dim)
        self.imu_projection = nn.Linear(token_dim, model_dim)

        self.eda_cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.imu_cls = nn.Parameter(torch.zeros(1, 1, model_dim))

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "eda_to_imu": nn.MultiheadAttention(
                            embed_dim=model_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "imu_to_eda": nn.MultiheadAttention(
                            embed_dim=model_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ffn_eda": nn.Sequential(
                            nn.Linear(model_dim, model_dim * 2),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(model_dim * 2, model_dim),
                        ),
                        "ffn_imu": nn.Sequential(
                            nn.Linear(model_dim, model_dim * 2),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(model_dim * 2, model_dim),
                        ),
                        "ln_eda_1": nn.LayerNorm(model_dim),
                        "ln_eda_2": nn.LayerNorm(model_dim),
                        "ln_imu_1": nn.LayerNorm(model_dim),
                        "ln_imu_2": nn.LayerNorm(model_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.output = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
        )

    def forward(self, transformer_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transformer_tokens: [B, 2, D] where token 0=EDA, token 1=IMU
        Returns:
            Encoded sensor embedding: [B, model_dim]
        """
        if transformer_tokens.ndim != 3 or transformer_tokens.size(1) != 2:
            raise ValueError(
                f"Expected transformer_tokens shape [B,2,D], got {tuple(transformer_tokens.shape)}"
            )

        eda_raw = transformer_tokens[:, 0, :]
        imu_raw = transformer_tokens[:, 1, :]

        eda = self.eda_projection(eda_raw).unsqueeze(1) + self.eda_cls
        imu = self.imu_projection(imu_raw).unsqueeze(1) + self.imu_cls

        for layer in self.layers:
            eda_attn, _ = layer["eda_to_imu"](query=eda, key=imu, value=imu)
            eda = layer["ln_eda_1"](eda + eda_attn)
            eda_ffn = layer["ffn_eda"](eda)
            eda = layer["ln_eda_2"](eda + eda_ffn)

            imu_attn, _ = layer["imu_to_eda"](query=imu, key=eda, value=eda)
            imu = layer["ln_imu_1"](imu + imu_attn)
            imu_ffn = layer["ffn_imu"](imu)
            imu = layer["ln_imu_2"](imu + imu_ffn)

        fused = torch.cat([eda.squeeze(1), imu.squeeze(1)], dim=-1)
        return self.output(fused)


class HybridICFModel(nn.Module):
    def __init__(
        self,
        hrv_input_dim: int,
        sensor_token_dim: int,
        hrv_hidden_dim: int = 128,
        sensor_model_dim: int = 128,
        fusion_hidden_dim: int = 128,
        num_targets: int = 4,
        num_heads: int = 4,
        num_sensor_layers: int = 2,
        dropout: float = 0.2,
        output_min: float = 0.0,
        output_max: float = 100.0,
    ) -> None:
        super().__init__()

        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.num_targets = int(num_targets)

        self.hrv_branch = ExpertHRVMLP(
            input_dim=hrv_input_dim,
            hidden_dim=hrv_hidden_dim,
            dropout=dropout,
        )
        self.sensor_branch = EDACrossIMUEncoder(
            token_dim=sensor_token_dim,
            model_dim=sensor_model_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_sensor_layers,
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hrv_hidden_dim + sensor_model_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, self.num_targets),
        )

    def forward(
        self,
        x_hrv: torch.Tensor,
        transformer_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hrv_embedding = self.hrv_branch(x_hrv)
        sensor_embedding = self.sensor_branch(transformer_tokens)

        fused = torch.cat([hrv_embedding, sensor_embedding], dim=-1)
        raw_score = self.fusion_head(fused)

        # Constrain score to configured range [output_min, output_max]
        pred_score = torch.sigmoid(raw_score) * (self.output_max - self.output_min) + self.output_min

        return {
            "pred_score": pred_score,
            "raw_score": raw_score,
            "hrv_embedding": hrv_embedding,
            "sensor_embedding": sensor_embedding,
        }


@dataclass
class CapacityConfig:
    omega_f: float = 0.5
    omega_q: float = 0.5
    tau: float = 0.5
    expected_frequency: float = 1.0
    expected_duration: float = 1.0


def compute_functional_capacity(
    probabilities: torch.Tensor,
    durations: torch.Tensor,
    config: CapacityConfig,
) -> torch.Tensor:
    """
    Computes C_a per sample with shape [B, 1].

    probabilities: [B, N] event probabilities p_i
    durations: [B, N] durations d_i
    """
    if probabilities.shape != durations.shape:
        raise ValueError(
            f"probabilities and durations must have same shape, got {probabilities.shape} vs {durations.shape}"
        )

    active = (probabilities > config.tau).float()
    active_count = active.sum(dim=1, keepdim=True)

    freq_term = active_count / max(config.expected_frequency, 1e-6)

    weighted_duration = (active * durations).sum(dim=1, keepdim=True)
    duration_norm = active_count * max(config.expected_duration, 1e-6)
    quality_term = weighted_duration / (duration_norm + 1e-8)

    capacity = (config.omega_f * freq_term) + (config.omega_q * quality_term)
    return capacity


class TheoryInformedICFLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, margin: float = 15.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.base_loss_fn = nn.SmoothL1Loss()

    def forward(
        self,
        pred_score: torch.Tensor,
        clinical_target: torch.Tensor,
        ca_value: torch.Tensor | None,
        icf_class_label: torch.Tensor,
        use_ca_constraint: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l_base = self.base_loss_fn(pred_score, clinical_target)
        if use_ca_constraint and ca_value is not None:
            if ca_value.shape != pred_score.shape:
                if ca_value.ndim == 2 and ca_value.size(1) == 1:
                    ca_value = ca_value.expand_as(pred_score)
                else:
                    raise ValueError(
                        f"ca_value shape {tuple(ca_value.shape)} is incompatible with pred_score shape {tuple(pred_score.shape)}"
                    )
            l_ca = F.mse_loss(pred_score, ca_value)
        else:
            l_ca = torch.zeros((), device=pred_score.device, dtype=pred_score.dtype)
        l_ordinal = self._compute_ordinal_contrastive(pred_score, icf_class_label)

        total_loss = l_base + (self.alpha * l_ca) + (self.beta * l_ordinal)
        return total_loss, l_base, l_ca, l_ordinal

    def _compute_ordinal_contrastive(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 1:
            preds = preds.view(-1, 1)
        if labels.ndim == 1:
            labels = labels.view(-1, 1)

        if preds.shape != labels.shape:
            raise ValueError(f"preds and labels must have same shape, got {preds.shape} vs {labels.shape}")

        per_target_losses = []
        for target_idx in range(preds.size(1)):
            target_preds = preds[:, target_idx].view(-1, 1)
            target_labels = labels[:, target_idx].view(-1)

            pred_diffs = target_preds - target_preds.transpose(0, 1)  # [B, B]
            label_diffs = target_labels.unsqueeze(1) - target_labels.unsqueeze(0)  # [B, B]

            identity_mask = torch.eye(pred_diffs.size(0), device=pred_diffs.device, dtype=torch.bool)

            same_class = (label_diffs == 0) & (~identity_mask)
            ordered_pairs = label_diffs > 0

            same_loss = (pred_diffs.pow(2) * same_class.float()).sum()
            violation = F.relu(self.margin - pred_diffs)
            ordered_loss = (violation.pow(2) * ordered_pairs.float()).sum()

            valid_pairs = same_class.float().sum() + ordered_pairs.float().sum()
            per_target_losses.append((same_loss + ordered_loss) / (valid_pairs + 1e-8))

        if not per_target_losses:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        return torch.stack(per_target_losses).mean()
