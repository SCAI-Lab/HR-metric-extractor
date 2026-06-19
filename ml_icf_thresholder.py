#!/usr/bin/env python3
"""
Post-hoc latent-to-ICF thresholding with interpolation for missing R4 classes.

`num_classes` denotes how many target dimensions are predicted in parallel.
For the current setup this is 4 (Basic Movements, Walking, Oral Care, Grooming).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


ICF_BRACKETS = [
	"0-4% (No difficulty)",
	"5-24% (Mild difficulty)",
	"25-49% (Moderate difficulty)",
	"50-95% (Severe difficulty)",
	"96-100% (Complete difficulty)",
]


@dataclass
class ICFLatentThresholder:
	num_classes: int = 4
	r4_min: int = 1
	r4_max: int = 5
	thresholds_: Dict[int, List[float]] = field(default_factory=dict)

	def fit(self, latent_scores: np.ndarray, r4_labels: np.ndarray) -> "ICFLatentThresholder":
		latent_scores = np.asarray(latent_scores, dtype=float)
		r4_labels = np.asarray(r4_labels, dtype=int)

		if latent_scores.ndim != 2 or r4_labels.ndim != 2:
			raise ValueError("latent_scores and r4_labels must both be 2D arrays [N, num_classes]")
		if latent_scores.shape != r4_labels.shape:
			raise ValueError(
				f"latent_scores and r4_labels must have matching shapes, got {latent_scores.shape} vs {r4_labels.shape}"
			)
		if latent_scores.shape[1] != self.num_classes:
			raise ValueError(
				f"Expected second dimension == num_classes ({self.num_classes}), got {latent_scores.shape[1]}"
			)

		self.thresholds_.clear()

		for target_idx in range(self.num_classes):
			scores = latent_scores[:, target_idx]
			labels = r4_labels[:, target_idx]

			class_scores: Dict[int, np.ndarray] = {}
			for r4 in range(self.r4_min, self.r4_max + 1):
				group = scores[labels == r4]
				class_scores[r4] = group[np.isfinite(group)]

			centroids = self._build_interpolated_centroids(class_scores)

			# Boundaries are midpoints between neighboring class centroids.
			boundaries = [
				float((centroids[r4] + centroids[r4 + 1]) / 2.0)
				for r4 in range(self.r4_min, self.r4_max)
			]
			boundaries.sort()

			# Anchor ends using class-tail percentiles when available.
			if class_scores[1].size > 0:
				boundaries[0] = min(boundaries[0], float(np.percentile(class_scores[1], 95.0)))
			if class_scores[5].size > 0:
				boundaries[-1] = max(boundaries[-1], float(np.percentile(class_scores[5], 5.0)))

			self.thresholds_[target_idx] = boundaries

		return self

	def predict_icf_brackets(self, latent_scores: np.ndarray) -> List[List[str]]:
		latent_scores = np.asarray(latent_scores, dtype=float)
		if latent_scores.ndim != 2 or latent_scores.shape[1] != self.num_classes:
			raise ValueError(f"Expected latent_scores shape [N, {self.num_classes}], got {latent_scores.shape}")

		if len(self.thresholds_) != self.num_classes:
			raise RuntimeError("Thresholder is not fit. Call fit(...) first.")

		outputs: List[List[str]] = []
		for row in latent_scores:
			sample_brackets: List[str] = []
			for target_idx in range(self.num_classes):
				thresholds = self.thresholds_[target_idx]
				bucket = int(np.searchsorted(np.asarray(thresholds, dtype=float), row[target_idx], side="right"))
				# score increases with capacity, so invert bucket to map to ICF difficulty
				icf_bucket = 4 - bucket
				icf_bucket = max(0, min(4, icf_bucket))
				sample_brackets.append(ICF_BRACKETS[icf_bucket])
			outputs.append(sample_brackets)
		return outputs

	def _build_interpolated_centroids(self, class_scores: Dict[int, np.ndarray]) -> Dict[int, float]:
		present = {k: float(np.median(v)) for k, v in class_scores.items() if v.size > 0}
		if not present:
			raise ValueError("Cannot fit thresholds because no valid latent scores were provided")

		centroids: Dict[int, float] = {}
		for r4 in range(self.r4_min, self.r4_max + 1):
			if r4 in present:
				centroids[r4] = present[r4]
				continue

			lower = [k for k in present if k < r4]
			upper = [k for k in present if k > r4]

			if lower and upper:
				k_low = max(lower)
				k_high = min(upper)
				centroids[r4] = (present[k_low] + present[k_high]) / 2.0
			elif lower:
				centroids[r4] = present[max(lower)]
			else:
				centroids[r4] = present[min(upper)]

		return centroids
