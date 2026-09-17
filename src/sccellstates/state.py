"""Leakage-safe program scoring and baseline state representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sccellstates.io import InputError, get_matrix, validate_anndata
from sccellstates.programs import ProgramSet


class StateError(ValueError):
    """Raised when state-representation inputs or fitted state are invalid."""


@dataclass(frozen=True)
class SampleSplit:
    """Disjoint biological-sample partitions for model development."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        partitions = {
            "train": tuple(str(value).strip() for value in self.train),
            "validation": tuple(str(value).strip() for value in self.validation),
            "test": tuple(str(value).strip() for value in self.test),
        }
        for name, values in partitions.items():
            if name == "train" and not values:
                raise StateError("the training partition must contain at least one sample")
            if any(not value for value in values):
                raise StateError(f"the {name} partition contains an empty sample identifier")
            if len(set(values)) != len(values):
                raise StateError(f"the {name} partition contains duplicate samples")
        sets = {name: set(values) for name, values in partitions.items()}
        if sets["train"] & sets["validation"]:
            raise StateError("train and validation samples must be disjoint")
        if sets["train"] & sets["test"]:
            raise StateError("train and test samples must be disjoint")
        if sets["validation"] & sets["test"]:
            raise StateError("validation and test samples must be disjoint")
        for name, values in partitions.items():
            object.__setattr__(self, name, values)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """All sample identifiers in train, validation, test order."""
        return self.train + self.validation + self.test


def make_sample_split(
    adata: ad.AnnData,
    *,
    sample_key: str,
    test_samples: Sequence[str],
    validation_samples: Sequence[str] = (),
) -> SampleSplit:
    """Create an exhaustive split, assigning unheld samples to training."""
    validate_anndata(adata, sample_key=sample_key)
    observed = set(adata.obs[sample_key].astype("string").str.strip().tolist())
    validation = tuple(str(value).strip() for value in validation_samples)
    test = tuple(str(value).strip() for value in test_samples)
    held_out = set(validation) | set(test)
    missing = sorted(held_out - observed)
    if missing:
        raise StateError(f"held-out samples are absent from adata.obs: {missing}")
    train = tuple(sorted(observed - held_out))
    return SampleSplit(train=train, validation=validation, test=test)


def sample_partition_labels(sample_ids: Sequence[object], split: SampleSplit) -> np.ndarray:
    """Map cell-level sample identifiers to split labels."""
    lookup = {
        **dict.fromkeys(split.train, "train"),
        **dict.fromkeys(split.validation, "validation"),
        **dict.fromkeys(split.test, "test"),
    }
    normalized = np.asarray([str(value).strip() for value in sample_ids], dtype=object)
    unknown = sorted(set(normalized) - lookup.keys())
    if unknown:
        raise StateError(f"samples are not assigned to the split: {unknown}")
    return np.asarray([lookup[value] for value in normalized], dtype=object)


def _training_mask(sample_ids: Sequence[object], split: SampleSplit) -> np.ndarray:
    labels = sample_partition_labels(sample_ids, split)
    mask = labels == "train"
    if not mask.any():
        raise StateError("the input contains no training cells")
    return mask


@dataclass(frozen=True)
class ProgramScorer:
    """Score cells against a fixed gene-program vocabulary without refitting."""

    vocabulary: ProgramSet

    def transform(self, adata: ad.AnnData, *, layer: str | None = None) -> np.ndarray:
        """Return cells-by-programs weighted activities for the selected matrix."""
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        result = matrix @ self.vocabulary.weights.T
        return _dense_activities(result)


@dataclass(frozen=True)
class NNLSProjectionResult:
    """Diagnostics and usages from fixed-basis non-negative projection."""

    usages: np.ndarray
    states: np.ndarray
    cell_names: tuple[str, ...]
    sample_id: str
    projection_error: np.ndarray
    relative_reconstruction_error: np.ndarray
    feature_coverage: float
    observed_feature_coverage: np.ndarray
    missing_features: tuple[str, ...]
    extra_features: tuple[str, ...]
    warnings: tuple[str, ...]
    vocabulary_id: str


class NNLSProjector:
    """Project cells onto a frozen program basis using row-wise NNLS."""

    def __init__(self, vocabulary: ProgramSet, *, error_warning_threshold: float = 1.0) -> None:
        if not isinstance(vocabulary, ProgramSet):
            raise TypeError("vocabulary must be a ProgramSet")
        if error_warning_threshold <= 0 or not np.isfinite(error_warning_threshold):
            raise ValueError("error_warning_threshold must be positive and finite")
        self.vocabulary = vocabulary
        self.error_warning_threshold = float(error_warning_threshold)

    def transform(
        self,
        adata: ad.AnnData,
        *,
        layer: str | None = None,
        sample_id: str = "projection",
    ) -> NNLSProjectionResult:
        """Estimate nonnegative usages and normalized states without refitting."""
        if not isinstance(adata, ad.AnnData):
            raise TypeError("adata must be an anndata.AnnData object")
        if not adata.var_names.is_unique:
            raise StateError("adata.var_names must be unique")
        matrix = get_matrix(adata, layer=layer)
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise StateError("selected expression matrix must be finite and numeric")
        if (values < 0).any():
            raise StateError("NNLS projection requires nonnegative expression values")
        lookup = {str(name): index for index, name in enumerate(adata.var_names)}
        present = tuple(name for name in self.vocabulary.feature_names if name in lookup)
        missing = tuple(name for name in self.vocabulary.feature_names if name not in lookup)
        extra = tuple(
            str(name) for name in adata.var_names if name not in self.vocabulary.feature_names
        )
        if not present:
            raise StateError("none of the vocabulary features are present in adata.var_names")
        indices = np.fromiter((lookup[name] for name in present), dtype=np.int64)
        weight_indices = np.fromiter(
            (self.vocabulary.feature_names.index(name) for name in present), dtype=np.int64
        )
        basis = self.vocabulary.weights[:, weight_indices].T
        usages = np.zeros((adata.n_obs, self.vocabulary.n_programs), dtype=np.float64)
        errors = np.zeros(adata.n_obs, dtype=np.float64)
        relative = np.zeros(adata.n_obs, dtype=np.float64)
        coverage = np.zeros(adata.n_obs, dtype=np.float64)
        for row in range(adata.n_obs):
            observed = matrix[row, indices]
            observed = (
                observed.toarray().ravel()
                if sparse.issparse(observed)
                else np.asarray(observed).ravel()
            )
            usages[row], errors[row] = nnls(basis, observed)
            norm = float(np.linalg.norm(observed))
            relative[row] = (
                errors[row] / norm if norm > 0 else (0.0 if errors[row] == 0 else np.inf)
            )
            coverage[row] = float(np.count_nonzero(observed) / len(present))
        totals = usages.sum(axis=1)
        states = np.zeros_like(usages)
        nonzero = totals > 0
        states[nonzero] = usages[nonzero] / totals[nonzero, None]
        warning_messages = []
        if missing:
            warning_messages.append(f"{len(missing)} vocabulary features are missing")
        if np.any(relative > self.error_warning_threshold):
            warning_messages.append("some cells have high relative reconstruction error")
        return NNLSProjectionResult(
            usages=usages,
            states=states,
            cell_names=tuple(map(str, adata.obs_names)),
            sample_id=str(sample_id),
            projection_error=errors,
            relative_reconstruction_error=relative,
            feature_coverage=len(present) / self.vocabulary.n_features,
            observed_feature_coverage=coverage,
            missing_features=missing,
            extra_features=extra,
            warnings=tuple(warning_messages),
            vocabulary_id=self.vocabulary.sample_id,
        )


def _vocabulary_matrix(
    adata: ad.AnnData,
    vocabulary: ProgramSet,
    *,
    layer: str | None,
) -> object:
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if not adata.var_names.is_unique:
        raise StateError("adata.var_names must be unique")
    lookup = {str(name): index for index, name in enumerate(adata.var_names)}
    missing = [name for name in vocabulary.feature_names if name not in lookup]
    if missing:
        raise StateError(f"vocabulary features are absent from adata.var_names: {missing[:5]}")
    indices = np.fromiter((lookup[name] for name in vocabulary.feature_names), dtype=np.int64)
    matrix = get_matrix(adata, layer=layer)
    if matrix.shape != adata.shape:
        raise StateError(
            f"selected matrix shape {matrix.shape} does not match AnnData shape {adata.shape}"
        )
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number):
        raise StateError("selected expression matrix must be numeric")
    if not np.isfinite(values).all():
        raise StateError("selected expression matrix contains NaN or infinite values")
    if (values < 0).any():
        raise StateError("selected expression matrix contains negative values")
    return matrix[:, indices]


def _dense_activities(values: object) -> np.ndarray:
    if sparse.issparse(values):
        values = values.toarray()
    activities = np.asarray(values, dtype=np.float64)
    if activities.ndim == 1:
        activities = activities[:, np.newaxis]
    if not np.isfinite(activities).all():
        raise StateError("program scoring produced non-finite activities")
    return activities


def _feature_mean_variance(matrix: object) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0), dtype=np.float64).ravel()
        squared_means = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        variances = np.maximum(squared_means - means**2, 0.0)
    else:
        values = np.asarray(matrix, dtype=np.float64)
        means = values.mean(axis=0)
        variances = values.var(axis=0)
    return means, variances


class GeneStandardizedProgramScorer:
    """Score a fixed vocabulary after training-fitted gene standardization.

    Gene means and population standard deviations are fitted on training cells
    only. Centering is applied algebraically after sparse multiplication, so a
    cells-by-genes dense matrix is never materialized.
    """

    def __init__(self, vocabulary: ProgramSet) -> None:
        self.vocabulary = vocabulary

    def fit(
        self,
        adata: ad.AnnData,
        *,
        sample_key: str,
        split: SampleSplit,
        layer: str | None = None,
    ) -> GeneStandardizedProgramScorer:
        """Fit per-gene location and scale using training samples only."""
        validate_anndata(adata, sample_key=sample_key, layer=layer)
        mask = _training_mask(adata.obs[sample_key].tolist(), split)
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        means, variances = _feature_mean_variance(matrix[mask, :])
        scales = np.sqrt(variances)
        scales[scales == 0] = 1.0
        self.means_ = means
        self.scales_ = scales
        self.scaled_weights_ = self.vocabulary.weights / scales[np.newaxis, :]
        self.center_offsets_ = self.scaled_weights_ @ means
        self.training_sample_ids_ = split.train
        return self

    def transform(self, adata: ad.AnnData, *, layer: str | None = None) -> np.ndarray:
        """Score cells using the fixed training-gene means and scales."""
        self._require_fitted()
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        result = matrix @ self.scaled_weights_.T
        return _dense_activities(result) - self.center_offsets_[np.newaxis, :]

    def _require_fitted(self) -> None:
        if not hasattr(self, "scaled_weights_"):
            raise StateError("GeneStandardizedProgramScorer must be fitted before use")


class CountResidualProgramScorer:
    """Score programs with training-fitted count rates and depth offsets.

    The expected count for gene ``g`` in cell ``i`` is ``depth_i * rate_g``,
    where rates are fitted from training cells only and ``depth_i`` is the
    selected-gene library size of that cell. Program scores are weighted sums
    of Pearson residuals. The algebra keeps the expression matrix sparse; only
    the cells-by-program result is dense.
    """

    def __init__(self, vocabulary: ProgramSet, *, pseudocount: float = 1e-08) -> None:
        if pseudocount <= 0 or not np.isfinite(pseudocount):
            raise ValueError("pseudocount must be positive and finite")
        self.vocabulary = vocabulary
        self.pseudocount = float(pseudocount)

    def fit(
        self,
        adata: ad.AnnData,
        *,
        sample_key: str,
        split: SampleSplit,
        layer: str | None = None,
    ) -> CountResidualProgramScorer:
        """Fit gene rates using counts from training samples only."""
        validate_anndata(adata, sample_key=sample_key, layer=layer)
        mask = _training_mask(adata.obs[sample_key].tolist(), split)
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        if sparse.issparse(matrix):
            totals = np.asarray(matrix[mask, :].sum(axis=0), dtype=np.float64).ravel()
        else:
            totals = np.asarray(matrix[mask, :], dtype=np.float64).sum(axis=0)
        rates = totals + self.pseudocount
        rates /= rates.sum()
        self.rates_ = rates
        self.training_sample_ids_ = split.train
        return self

    def transform(self, adata: ad.AnnData, *, layer: str | None = None) -> np.ndarray:
        """Score cells using fixed training rates and per-cell count offsets."""
        self._require_fitted()
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        if sparse.issparse(matrix):
            depths = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        else:
            depths = np.asarray(matrix, dtype=np.float64).sum(axis=1)
        safe_depths = np.maximum(depths, 1e-12)
        inverse_sqrt_rates = 1.0 / np.sqrt(self.rates_)
        weighted_counts = matrix @ (self.vocabulary.weights.T * inverse_sqrt_rates[:, None])
        scores = _dense_activities(weighted_counts) / np.sqrt(safe_depths[:, None])
        scores -= np.sqrt(safe_depths[:, None]) * (
            self.vocabulary.weights @ np.sqrt(self.rates_)[:, None]
        ).T
        scores[depths == 0, :] = 0.0
        return _dense_activities(scores)

    def _require_fitted(self) -> None:
        if not hasattr(self, "rates_"):
            raise StateError("CountResidualProgramScorer must be fitted before use")


class TechnicalResidualProgramScorer:
    """Score programs from gene residuals after training-fitted nuisance regression.

    Each selected gene is regressed on an intercept, log selected-gene library
    depth, and log detected-gene count using training cells only. Scores are
    weighted sums of held-out residuals. Sparse matrices remain sparse because
    the residual score is evaluated as a difference of two cells-by-program
    products rather than materializing a cells-by-genes residual matrix.
    """

    def __init__(self, vocabulary: ProgramSet) -> None:
        self.vocabulary = vocabulary

    def fit(
        self,
        adata: ad.AnnData,
        *,
        sample_key: str,
        split: SampleSplit,
        layer: str | None = None,
    ) -> TechnicalResidualProgramScorer:
        """Fit gene-wise nuisance regressions on training cells only."""
        validate_anndata(adata, sample_key=sample_key, layer=layer)
        mask = _training_mask(adata.obs[sample_key].tolist(), split)
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        training = matrix[mask, :]
        design = _technical_design(training)
        log_matrix = _log1p_matrix(training)
        cross_product = np.asarray(log_matrix.T @ design, dtype=np.float64)
        coefficients = np.linalg.lstsq(
            design.T @ design,
            cross_product.T,
            rcond=None,
        )[0]
        self.coefficients_ = coefficients
        self.training_sample_ids_ = split.train
        return self

    def transform(self, adata: ad.AnnData, *, layer: str | None = None) -> np.ndarray:
        """Score cells using fixed training-fitted nuisance regressions."""
        self._require_fitted()
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        design = _technical_design(matrix)
        log_matrix = _log1p_matrix(matrix)
        observed = _dense_activities(log_matrix @ self.vocabulary.weights.T)
        fitted = design @ (self.coefficients_ @ self.vocabulary.weights.T)
        return _dense_activities(observed - fitted)

    def _require_fitted(self) -> None:
        if not hasattr(self, "coefficients_"):
            raise StateError("TechnicalResidualProgramScorer must be fitted before use")


def _technical_design(matrix: object) -> np.ndarray:
    """Return intercept, log selected depth, and log detected-gene covariates."""
    if sparse.issparse(matrix):
        depths = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        detected = np.diff(matrix.tocsr().indptr).astype(np.float64)
    else:
        values = np.asarray(matrix, dtype=np.float64)
        depths = values.sum(axis=1)
        detected = np.count_nonzero(values, axis=1).astype(np.float64)
    return np.column_stack((np.ones(matrix.shape[0]), np.log1p(depths), np.log1p(detected)))


def _log1p_matrix(matrix: object) -> object:
    if sparse.issparse(matrix):
        result = matrix.copy().astype(np.float64)
        result.data = np.log1p(result.data)
        return result
    return np.log1p(np.asarray(matrix, dtype=np.float64))


class MatchedControlProgramScorer:
    """Score top program genes relative to expression-matched control genes.

    Mean-expression bins and control genes are selected from training cells
    only. For each program, its highest-weight genes are L1-renormalized and
    their weighted expression is reduced by the unweighted mean expression of
    deterministic, expression-matched controls.
    """

    def __init__(
        self,
        vocabulary: ProgramSet,
        *,
        n_top_genes: int = 50,
        n_bins: int = 25,
        control_size: int = 50,
        random_state: int,
    ) -> None:
        if not 1 <= n_top_genes < vocabulary.n_features:
            raise ValueError("n_top_genes must be positive and smaller than n_features")
        if not 1 <= n_bins <= vocabulary.n_features:
            raise ValueError("n_bins must be between 1 and n_features")
        if control_size < 1:
            raise ValueError("control_size must be at least 1")
        self.vocabulary = vocabulary
        self.n_top_genes = n_top_genes
        self.n_bins = n_bins
        self.control_size = control_size
        self.random_state = random_state

    def fit(
        self,
        adata: ad.AnnData,
        *,
        sample_key: str,
        split: SampleSplit,
        layer: str | None = None,
    ) -> MatchedControlProgramScorer:
        """Fit expression bins and select control genes from training cells."""
        validate_anndata(adata, sample_key=sample_key, layer=layer)
        mask = _training_mask(adata.obs[sample_key].tolist(), split)
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        means, _ = _feature_mean_variance(matrix[mask, :])

        order = np.lexsort((np.arange(self.vocabulary.n_features), means))
        bins = np.empty(self.vocabulary.n_features, dtype=np.int64)
        bins[order] = (
            np.arange(self.vocabulary.n_features) * self.n_bins // self.vocabulary.n_features
        )
        rng = np.random.default_rng(self.random_state)
        top_indices: list[np.ndarray] = []
        top_weights: list[np.ndarray] = []
        control_indices: list[np.ndarray] = []
        feature_indices = np.arange(self.vocabulary.n_features)
        for weights in self.vocabulary.weights:
            top = np.lexsort((feature_indices, -weights))[: self.n_top_genes]
            selected_controls: list[np.ndarray] = []
            for expression_bin in np.unique(bins[top]):
                candidates = feature_indices[
                    (bins == expression_bin) & ~np.isin(feature_indices, top)
                ]
                if candidates.size == 0:
                    continue
                size = min(self.control_size, candidates.size)
                selected_controls.append(np.sort(rng.choice(candidates, size=size, replace=False)))
            if not selected_controls:
                raise StateError("no matched control genes are available for a program")
            controls = np.unique(np.concatenate(selected_controls))
            selected_weights = weights[top]
            weight_sum = selected_weights.sum()
            if weight_sum <= 0:
                raise StateError("top program genes have zero total weight")
            top_indices.append(top)
            top_weights.append(selected_weights / weight_sum)
            control_indices.append(controls)

        self.top_indices_ = tuple(top_indices)
        self.top_weights_ = tuple(top_weights)
        self.control_indices_ = tuple(control_indices)
        self.control_feature_names_ = tuple(
            tuple(self.vocabulary.feature_names[index] for index in indices)
            for indices in self.control_indices_
        )
        self.training_sample_ids_ = split.train
        return self

    def transform(self, adata: ad.AnnData, *, layer: str | None = None) -> np.ndarray:
        """Score cells using the fixed training-selected program and controls."""
        self._require_fitted()
        matrix = _vocabulary_matrix(adata, self.vocabulary, layer=layer)
        columns = []
        for top, weights, controls in zip(
            self.top_indices_, self.top_weights_, self.control_indices_, strict=True
        ):
            program_score = _dense_activities(matrix[:, top] @ weights)[:, 0]
            control_score = np.asarray(matrix[:, controls].mean(axis=1)).ravel()
            columns.append(program_score - control_score)
        return _dense_activities(np.column_stack(columns))

    def _require_fitted(self) -> None:
        if not hasattr(self, "control_indices_"):
            raise StateError("MatchedControlProgramScorer must be fitted before use")


def store_program_activities(
    adata: ad.AnnData, activities: np.ndarray, *, overwrite: bool = False
) -> None:
    """Store cells-by-program activities in ``.obsm['X_sccs_programs']``."""
    values = _validate_dense_matrix(activities, "activities")
    if values.shape[0] != adata.n_obs:
        raise StateError("activities must have one row per observation")
    if "X_sccs_programs" in adata.obsm and not overwrite:
        raise StateError("adata.obsm['X_sccs_programs'] already exists")
    adata.obsm["X_sccs_programs"] = values.copy()


def _validate_dense_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise StateError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise StateError(f"{name} contains NaN or infinite values")
    return array


def _bounded_dense(matrix: object, *, max_dense_elements: int) -> np.ndarray:
    if max_dense_elements < 1:
        raise ValueError("max_dense_elements must be at least 1")
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise StateError("expression matrix must be two-dimensional")
    elements = int(matrix.shape[0]) * int(matrix.shape[1])
    if elements > max_dense_elements:
        raise StateError(
            f"dense conversion would require {elements} elements; limit is {max_dense_elements}"
        )
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return _validate_dense_matrix(dense, "expression matrix")


class ExpressionPCA:
    """PCA expression baseline fitted only on explicitly training samples."""

    def __init__(
        self,
        n_components: int,
        *,
        random_state: int,
        max_dense_elements: int = 10_000_000,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = n_components
        self.random_state = random_state
        self.max_dense_elements = max_dense_elements

    def fit(
        self,
        adata: ad.AnnData,
        *,
        sample_key: str,
        split: SampleSplit,
        layer: str | None = None,
    ) -> ExpressionPCA:
        """Fit on training cells, retaining exact feature-order provenance."""
        validate_anndata(adata, sample_key=sample_key, layer=layer)
        mask = _training_mask(adata.obs[sample_key].tolist(), split)
        training = _bounded_dense(
            get_matrix(adata, layer=layer)[mask, :],
            max_dense_elements=self.max_dense_elements,
        )
        if self.n_components > min(training.shape):
            raise StateError("n_components cannot exceed training cells or features")
        self.model_ = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            random_state=self.random_state,
        ).fit(training)
        self.feature_names_ = tuple(str(name) for name in adata.var_names)
        self.training_sample_ids_ = split.train
        return self

    def transform(self, adata: ad.AnnData, *, layer: str | None = None) -> np.ndarray:
        """Project cells without updating the training fit."""
        self._require_fitted()
        if tuple(str(name) for name in adata.var_names) != self.feature_names_:
            raise StateError("adata.var_names must exactly match the fitted feature order")
        matrix = _bounded_dense(
            get_matrix(adata, layer=layer), max_dense_elements=self.max_dense_elements
        )
        return np.asarray(self.model_.transform(matrix), dtype=np.float64)

    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Reconstruct expression values from PCA coordinates."""
        self._require_fitted()
        return np.asarray(self.model_.inverse_transform(coordinates), dtype=np.float64)

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise StateError("ExpressionPCA must be fitted before use")


class DirectProgramState:
    """Standardized direct program-activity representation."""

    def fit(
        self,
        activities: np.ndarray,
        *,
        sample_ids: Sequence[object],
        split: SampleSplit,
    ) -> DirectProgramState:
        """Fit scaling parameters on training cells only."""
        values = _validate_dense_matrix(activities, "activities")
        mask = _training_mask(sample_ids, split)
        if len(mask) != values.shape[0]:
            raise StateError("sample_ids length must match activity rows")
        self.scaler_ = StandardScaler().fit(values[mask])
        self.training_sample_ids_ = split.train
        return self

    def transform(self, activities: np.ndarray) -> np.ndarray:
        """Standardize activities using training statistics."""
        self._require_fitted()
        return np.asarray(self.scaler_.transform(activities), dtype=np.float64)

    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Recover activities from standardized coordinates."""
        self._require_fitted()
        return np.asarray(self.scaler_.inverse_transform(coordinates), dtype=np.float64)

    def _require_fitted(self) -> None:
        if not hasattr(self, "scaler_"):
            raise StateError("DirectProgramState must be fitted before use")


class LinearProgramState(DirectProgramState):
    """Linear PCA compression of program activities."""

    def __init__(self, n_components: int, *, random_state: int) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = n_components
        self.random_state = random_state

    def fit(
        self,
        activities: np.ndarray,
        *,
        sample_ids: Sequence[object],
        split: SampleSplit,
    ) -> LinearProgramState:
        values = _validate_dense_matrix(activities, "activities")
        mask = _training_mask(sample_ids, split)
        if len(mask) != values.shape[0]:
            raise StateError("sample_ids length must match activity rows")
        self.scaler_ = StandardScaler().fit(values[mask])
        scaled = self.scaler_.transform(values[mask])
        if self.n_components > min(scaled.shape):
            raise StateError("n_components cannot exceed training cells or programs")
        self.model_ = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            random_state=self.random_state,
        ).fit(scaled)
        self.training_sample_ids_ = split.train
        return self

    def transform(self, activities: np.ndarray) -> np.ndarray:
        self._require_fitted()
        scaled = self.scaler_.transform(_validate_dense_matrix(activities, "activities"))
        return np.asarray(self.model_.transform(scaled), dtype=np.float64)

    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        self._require_fitted()
        scaled = self.model_.inverse_transform(coordinates)
        return np.asarray(self.scaler_.inverse_transform(scaled), dtype=np.float64)

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise StateError("LinearProgramState must be fitted before use")


class NonlinearProgramState(DirectProgramState):
    """Single-bottleneck MLP autoencoder baseline for program activities."""

    def __init__(
        self,
        n_components: int,
        *,
        random_state: int,
        max_iter: int = 500,
        alpha: float = 0.0001,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.alpha = alpha

    def fit(
        self,
        activities: np.ndarray,
        *,
        sample_ids: Sequence[object],
        split: SampleSplit,
    ) -> NonlinearProgramState:
        values = _validate_dense_matrix(activities, "activities")
        mask = _training_mask(sample_ids, split)
        if len(mask) != values.shape[0]:
            raise StateError("sample_ids length must match activity rows")
        self.scaler_ = StandardScaler().fit(values[mask])
        scaled = self.scaler_.transform(values[mask])
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.n_components,),
            activation="tanh",
            solver="lbfgs",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(scaled, scaled)
        self.training_sample_ids_ = split.train
        return self

    def transform(self, activities: np.ndarray) -> np.ndarray:
        """Return the fitted tanh bottleneck activations."""
        self._require_fitted()
        scaled = self.scaler_.transform(_validate_dense_matrix(activities, "activities"))
        return np.tanh(scaled @ self.model_.coefs_[0] + self.model_.intercepts_[0])

    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Decode bottleneck coordinates back to program activities."""
        self._require_fitted()
        values = _validate_dense_matrix(coordinates, "coordinates")
        scaled = values @ self.model_.coefs_[1] + self.model_.intercepts_[1]
        return np.asarray(self.scaler_.inverse_transform(scaled), dtype=np.float64)

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise StateError("NonlinearProgramState must be fitted before use")


def store_state_coordinates(
    adata: ad.AnnData,
    coordinates: np.ndarray,
    *,
    method: str,
    training_sample_ids: Sequence[str],
    parameters: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    """Store chosen state coordinates and versioned fit provenance."""
    values = _validate_dense_matrix(coordinates, "coordinates")
    if values.shape[0] != adata.n_obs:
        raise StateError("coordinates must have one row per observation")
    if not method.strip():
        raise StateError("method must be a non-empty string")
    training = tuple(str(value).strip() for value in training_sample_ids)
    if not training or any(not value for value in training):
        raise StateError("training_sample_ids must be non-empty identifiers")
    existing = adata.uns.get("sccellstates", {})
    if not isinstance(existing, dict):
        raise InputError("adata.uns['sccellstates'] must be a dictionary")
    if "X_sccs_state" in adata.obsm and not overwrite:
        raise StateError("adata.obsm['X_sccs_state'] already exists")
    if "state_representation" in existing and not overwrite:
        raise StateError("state representation provenance already exists")
    provenance = dict(existing)
    provenance.setdefault("schema_version", "1.0")
    provenance["state_representation"] = {
        "method": method,
        "training_sample_ids": list(training),
        "n_components": values.shape[1],
        "parameters": dict(parameters or {}),
    }
    adata.obsm["X_sccs_state"] = values.copy()
    adata.uns["sccellstates"] = provenance
