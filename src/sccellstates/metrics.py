"""Held-out and technical-confounding metrics for state representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class MetricsError(ValueError):
    """Raised when evaluation inputs are invalid."""


def reconstruction_by_partition(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    partition_labels: Sequence[object],
) -> pd.DataFrame:
    """Summarize reconstruction error separately for each present partition."""
    truth = np.asarray(observed, dtype=np.float64)
    estimate = np.asarray(reconstructed, dtype=np.float64)
    labels = np.asarray(partition_labels, dtype=object)
    if truth.ndim != 2 or truth.shape != estimate.shape:
        raise MetricsError("observed and reconstructed must have the same 2D shape")
    if labels.ndim != 1 or len(labels) != truth.shape[0]:
        raise MetricsError("partition_labels must have one value per row")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise MetricsError("reconstruction inputs must be finite")
    rows: list[dict[str, object]] = []
    for partition in ("train", "validation", "test"):
        mask = labels == partition
        if not mask.any():
            continue
        residual = truth[mask] - estimate[mask]
        denominator = np.square(truth[mask] - truth[mask].mean(axis=0)).sum()
        numerator = np.square(residual).sum()
        rows.append(
            {
                "partition": partition,
                "n_observations": int(mask.sum()),
                "mean_squared_error": float(np.square(residual).mean()),
                "r2_variance_weighted": (
                    float(1 - numerator / denominator) if denominator > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def technical_associations(
    coordinates: np.ndarray,
    covariates: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Compute coordinate-wise Spearman associations with numeric covariates."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise MetricsError("coordinates must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise MetricsError("coordinates must be finite")
    rows: list[dict[str, object]] = []
    for name, raw_covariate in covariates.items():
        covariate = np.asarray(raw_covariate, dtype=np.float64)
        if covariate.ndim != 1 or len(covariate) != values.shape[0]:
            raise MetricsError(f"covariate {name!r} must have one value per row")
        finite = np.isfinite(covariate)
        for index in range(values.shape[1]):
            if finite.sum() < 2 or np.unique(covariate[finite]).size < 2:
                correlation = np.nan
            else:
                correlation = float(spearmanr(values[finite, index], covariate[finite]).statistic)
            rows.append(
                {
                    "coordinate": index,
                    "covariate": str(name),
                    "spearman_r": correlation,
                    "absolute_spearman_r": abs(correlation),
                    "n_observations": int(finite.sum()),
                }
            )
    return pd.DataFrame(rows)


def sample_bootstrap(
    values: np.ndarray,
    sample_ids: Sequence[object],
    *,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    random_state: int,
) -> pd.DataFrame:
    """Estimate sample-level means and bootstrap confidence intervals.

    Values are aggregated within biological sample before samples are
    resampled. A two-dimensional input is treated as cells by coordinates and
    produces one row per coordinate. This keeps cell-rich samples from
    receiving disproportionate weight and makes the uncertainty unit explicit.
    """
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, np.newaxis]
    if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] == 0:
        raise MetricsError("values must be a non-empty one- or two-dimensional array")
    if not np.isfinite(observations).all():
        raise MetricsError("values must be finite")
    labels = np.asarray([str(value).strip() for value in sample_ids], dtype=object)
    if labels.ndim != 1 or len(labels) != observations.shape[0]:
        raise MetricsError("sample_ids must have one value per row")
    if (labels == "").any():
        raise MetricsError("sample_ids must not contain empty values")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    unique_samples = np.asarray(pd.unique(labels), dtype=object)
    sample_means = np.vstack(
        [observations[labels == sample_id].mean(axis=0) for sample_id in unique_samples]
    )
    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, len(unique_samples), size=(n_bootstrap, len(unique_samples)))
    bootstrap_means = sample_means[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    rows = []
    for coordinate in range(observations.shape[1]):
        rows.append(
            {
                "coordinate": coordinate,
                "estimate": float(sample_means[:, coordinate].mean()),
                "lower": float(np.quantile(bootstrap_means[:, coordinate], alpha)),
                "upper": float(np.quantile(bootstrap_means[:, coordinate], 1 - alpha)),
                "n_samples": len(unique_samples),
            }
        )
    return pd.DataFrame(rows)
