"""Fail-closed interpolation over measured vLLM batch latency profiles.

The model in this module contains no built-in timing constants.  Every latency
prediction is derived from a versioned JSON profile whose runtime identity,
training envelope, interpolation parameters, and raw-measurement hash are
explicit.  Profiles are parsed without pickle or executable deserialization.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


__all__ = [
    "PROFILE_JSON_SCHEMA",
    "PROFILE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "ALLOWED_FEATURE_TRANSFORMS",
    "BatchDescriptor",
    "BatchTimeProfileIdentity",
    "BatchTimeProfilePoint",
    "BatchTimePrediction",
    "CalibratedBatchTimeModel",
    "BatchTimeModelError",
    "ProfileSchemaError",
    "ProfileIdentityError",
    "PredictionRejectedError",
    "OutOfDistributionError",
    "PredictionUncertaintyError",
]

PROFILE_SCHEMA_VERSION = 2
MAX_PROFILE_INTEGER = (1 << 53) - 1
FEATURE_NAMES = (
    "batch_size",
    "total_tokens",
    "prefill_tokens",
    "decode_tokens",
    "context_tokens",
    "max_context_tokens",
)
ALLOWED_FEATURE_TRANSFORMS = frozenset({"linear", "log1p"})


def _descriptor_schema_properties() -> dict[str, Any]:
    return {
        "batch_size": {"type": "integer", "minimum": 1},
        "total_tokens": {"type": "integer", "minimum": 1},
        "prefill_tokens": {"type": "integer", "minimum": 0},
        "decode_tokens": {"type": "integer", "minimum": 0},
        "context_tokens": {"type": "integer", "minimum": 0},
        "max_context_tokens": {"type": "integer", "minimum": 0},
    }


# This schema is exported for profile producers.  Runtime validation below is
# dependency-free and additionally checks cross-field invariants that JSON
# Schema cannot express concisely (for example total = prefill + decode).
PROFILE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://mlsysbench.local/schemas/vllm-batch-time-profile-v2.json",
    "title": "MLSysBench calibrated vLLM batch time profile",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "identity",
        "measurement",
        "feature_order",
        "training_ranges",
        "interpolation",
        "points",
    ],
    "properties": {
        "schema_version": {"const": PROFILE_SCHEMA_VERSION},
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "profile_id",
                "profile_revision",
                "device",
                "runtime",
                "model",
            ],
            "properties": {
                "profile_id": {"type": "string", "minLength": 1},
                "profile_revision": {"type": "string", "minLength": 1},
                "device": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "vendor",
                        "model",
                        "compute_capability",
                        "gpu_count",
                        "total_memory_bytes",
                    ],
                    "properties": {
                        "vendor": {"type": "string", "minLength": 1},
                        "model": {"type": "string", "minLength": 1},
                        "compute_capability": {"type": "string", "minLength": 1},
                        "gpu_count": {"type": "integer", "minimum": 1},
                        "total_memory_bytes": {"type": "integer", "minimum": 1},
                    },
                },
                "runtime": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "driver_version",
                        "cuda_version",
                        "python_version",
                        "torch_version",
                        "vllm_revision",
                        "attention_backend",
                        "dtype",
                        "enforce_eager",
                    ],
                    "properties": {
                        "driver_version": {"type": "string", "minLength": 1},
                        "cuda_version": {"type": "string", "minLength": 1},
                        "python_version": {"type": "string", "minLength": 1},
                        "torch_version": {"type": "string", "minLength": 1},
                        "vllm_revision": {"type": "string", "minLength": 1},
                        "attention_backend": {"type": "string", "minLength": 1},
                        "dtype": {"type": "string", "minLength": 1},
                        "enforce_eager": {"type": "boolean"},
                    },
                },
                "model": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "revision",
                        "max_model_len",
                        "tensor_parallel_size",
                    ],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "revision": {"type": "string", "minLength": 1},
                        "max_model_len": {"type": "integer", "minimum": 1},
                        "tensor_parallel_size": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
        "measurement": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "latency_unit",
                "latency_statistic",
                "timer",
                "warmup_iterations",
                "minimum_measured_iterations",
                "clock_policy",
                "raw_artifact_sha256",
            ],
            "properties": {
                "latency_unit": {"const": "ms"},
                "latency_statistic": {"const": "mean"},
                "timer": {"type": "string", "minLength": 1},
                "warmup_iterations": {"type": "integer", "minimum": 1},
                "minimum_measured_iterations": {"type": "integer", "minimum": 3},
                "clock_policy": {"type": "string", "minLength": 1},
                "raw_artifact_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
            },
        },
        "feature_order": {
            "type": "array",
            "prefixItems": [{"const": name} for name in FEATURE_NAMES],
            "minItems": len(FEATURE_NAMES),
            "maxItems": len(FEATURE_NAMES),
        },
        "training_ranges": {
            "type": "object",
            "additionalProperties": False,
            "required": list(FEATURE_NAMES),
            "properties": {
                name: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["min", "max"],
                    "properties": {
                        "min": {"type": "integer", "minimum": 0},
                        "max": {"type": "integer", "minimum": 0},
                    },
                }
                for name in FEATURE_NAMES
            },
        },
        "interpolation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "method",
                "neighbors",
                "distance_power",
                "max_normalized_distance",
                "max_relative_uncertainty",
                "feature_weights",
                "feature_transforms",
            ],
            "properties": {
                "method": {"const": "inverse_distance_weighting_v1"},
                "neighbors": {"type": "integer", "minimum": 2},
                "distance_power": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 8,
                },
                "max_normalized_distance": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                "max_relative_uncertainty": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                "feature_weights": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(FEATURE_NAMES),
                    "properties": {
                        name: {"type": "number", "exclusiveMinimum": 0}
                        for name in FEATURE_NAMES
                    },
                },
                "feature_transforms": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(FEATURE_NAMES),
                    "properties": {
                        name: {"enum": sorted(ALLOWED_FEATURE_TRANSFORMS)}
                        for name in FEATURE_NAMES
                    },
                },
            },
        },
        "points": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "point_id",
                    *FEATURE_NAMES,
                    "latency_ms",
                    "latency_stddev_ms",
                    "sample_count",
                ],
                "properties": {
                    "point_id": {"type": "string", "minLength": 1},
                    **_descriptor_schema_properties(),
                    "latency_ms": {"type": "number", "exclusiveMinimum": 0},
                    "latency_stddev_ms": {"type": "number", "minimum": 0},
                    "sample_count": {"type": "integer", "minimum": 3},
                },
            },
        },
    },
}


class BatchTimeModelError(ValueError):
    """Base class for profile and prediction failures."""


class ProfileSchemaError(BatchTimeModelError):
    """Raised when a JSON profile violates the strict profile contract."""


class ProfileIdentityError(BatchTimeModelError):
    """Raised when a profile does not match the requested execution identity."""


class PredictionRejectedError(BatchTimeModelError):
    """Base class for fail-closed prediction rejection with diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class OutOfDistributionError(PredictionRejectedError):
    """Raised when a descriptor is outside the calibrated decision region."""


class PredictionUncertaintyError(PredictionRejectedError):
    """Raised when local profile evidence is too uncertain for prediction."""


@dataclass(frozen=True)
class BatchDescriptor:
    """Features emitted by the shadow vLLM executor for one scheduled batch."""

    batch_size: int
    total_tokens: int
    prefill_tokens: int
    decode_tokens: int
    context_tokens: int
    max_context_tokens: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BatchDescriptor":
        _require_exact_keys(payload, set(FEATURE_NAMES), "batch descriptor")
        values = {
            name: _require_int(payload[name], f"batch descriptor.{name}", minimum=0)
            for name in FEATURE_NAMES
        }
        descriptor = cls(**values)
        descriptor.validate()
        return descriptor

    def validate(self, *, max_model_len: int | None = None) -> None:
        for name in FEATURE_NAMES:
            _require_int(getattr(self, name), f"batch descriptor.{name}", minimum=0)
        for name in ("batch_size", "total_tokens"):
            if getattr(self, name) <= 0:
                raise ProfileSchemaError(f"batch descriptor.{name} must be positive")
        if self.prefill_tokens + self.decode_tokens != self.total_tokens:
            raise ProfileSchemaError(
                "batch descriptor.total_tokens must equal prefill_tokens + decode_tokens"
            )
        if self.batch_size > self.total_tokens:
            raise ProfileSchemaError("batch descriptor.batch_size cannot exceed total_tokens")
        if self.decode_tokens > self.batch_size:
            raise ProfileSchemaError(
                "batch descriptor.decode_tokens cannot exceed batch_size for one scheduler step"
            )
        if self.max_context_tokens > self.context_tokens:
            raise ProfileSchemaError(
                "batch descriptor.max_context_tokens cannot exceed context_tokens"
            )
        if (self.context_tokens == 0) != (self.max_context_tokens == 0):
            raise ProfileSchemaError(
                "batch descriptor.context_tokens and max_context_tokens must both be zero "
                "for an initial prefill"
            )
        if self.context_tokens == 0 and self.decode_tokens != 0:
            raise ProfileSchemaError(
                "batch descriptor.decode_tokens must be zero when no context has been computed"
            )
        if max_model_len is not None:
            if self.max_context_tokens > max_model_len:
                raise ProfileSchemaError(
                    "batch descriptor.max_context_tokens exceeds the profiled max_model_len"
                )
            if self.context_tokens > self.batch_size * max_model_len:
                raise ProfileSchemaError(
                    "batch descriptor.context_tokens exceeds batch_size * max_model_len"
                )

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in FEATURE_NAMES}

    def feature_tuple(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name in FEATURE_NAMES)


@dataclass(frozen=True)
class BatchTimeProfileIdentity:
    """Execution identity that must exactly match a loaded profile."""

    profile_id: str
    profile_revision: str
    device_vendor: str
    device_model: str
    compute_capability: str
    gpu_count: int
    total_memory_bytes: int
    driver_version: str
    cuda_version: str
    python_version: str
    torch_version: str
    vllm_revision: str
    attention_backend: str
    dtype: str
    enforce_eager: bool
    model_name: str
    model_revision: str
    max_model_len: int
    tensor_parallel_size: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BatchTimeProfileIdentity":
        _require_exact_keys(
            payload,
            {"profile_id", "profile_revision", "device", "runtime", "model"},
            "identity",
        )
        device = _require_mapping(payload["device"], "identity.device")
        runtime = _require_mapping(payload["runtime"], "identity.runtime")
        model = _require_mapping(payload["model"], "identity.model")
        _require_exact_keys(
            device,
            {"vendor", "model", "compute_capability", "gpu_count", "total_memory_bytes"},
            "identity.device",
        )
        _require_exact_keys(
            runtime,
            {
                "driver_version",
                "cuda_version",
                "python_version",
                "torch_version",
                "vllm_revision",
                "attention_backend",
                "dtype",
                "enforce_eager",
            },
            "identity.runtime",
        )
        _require_exact_keys(
            model,
            {"name", "revision", "max_model_len", "tensor_parallel_size"},
            "identity.model",
        )
        if not isinstance(runtime["enforce_eager"], bool):
            raise ProfileSchemaError("identity.runtime.enforce_eager must be boolean")
        return cls(
            profile_id=_require_string(payload["profile_id"], "identity.profile_id"),
            profile_revision=_require_string(
                payload["profile_revision"], "identity.profile_revision"
            ),
            device_vendor=_require_string(device["vendor"], "identity.device.vendor"),
            device_model=_require_string(device["model"], "identity.device.model"),
            compute_capability=_require_string(
                device["compute_capability"], "identity.device.compute_capability"
            ),
            gpu_count=_require_int(device["gpu_count"], "identity.device.gpu_count", minimum=1),
            total_memory_bytes=_require_int(
                device["total_memory_bytes"],
                "identity.device.total_memory_bytes",
                minimum=1,
            ),
            driver_version=_require_string(
                runtime["driver_version"], "identity.runtime.driver_version"
            ),
            cuda_version=_require_string(
                runtime["cuda_version"], "identity.runtime.cuda_version"
            ),
            python_version=_require_string(
                runtime["python_version"], "identity.runtime.python_version"
            ),
            torch_version=_require_string(
                runtime["torch_version"], "identity.runtime.torch_version"
            ),
            vllm_revision=_require_string(
                runtime["vllm_revision"], "identity.runtime.vllm_revision"
            ),
            attention_backend=_require_string(
                runtime["attention_backend"], "identity.runtime.attention_backend"
            ),
            dtype=_require_string(runtime["dtype"], "identity.runtime.dtype"),
            enforce_eager=runtime["enforce_eager"],
            model_name=_require_string(model["name"], "identity.model.name"),
            model_revision=_require_string(model["revision"], "identity.model.revision"),
            max_model_len=_require_int(
                model["max_model_len"], "identity.model.max_model_len", minimum=1
            ),
            tensor_parallel_size=_require_int(
                model["tensor_parallel_size"],
                "identity.model.tensor_parallel_size",
                minimum=1,
            ),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class BatchTimeProfilePoint:
    """One aggregated hardware measurement at an exact batch descriptor."""

    point_id: str
    descriptor: BatchDescriptor
    latency_ms: float
    latency_stddev_ms: float
    sample_count: int


@dataclass(frozen=True)
class BatchTimePrediction:
    """Predicted latency and the evidence used to accept the prediction."""

    latency_ms: float
    nearest_normalized_distance: float
    weighted_normalized_distance: float
    uncertainty_ms: float
    relative_uncertainty: float
    exact_match: bool
    neighbor_point_ids: tuple[str, ...]
    neighbor_distances: tuple[float, ...]
    neighbor_weights: tuple[float, ...]

    @property
    def normalized_distance(self) -> float:
        """Nearest-point distance used by the fail-closed OOD gate."""

        return self.nearest_normalized_distance

    def diagnostics(self) -> dict[str, Any]:
        return {
            "normalized_distance": self.normalized_distance,
            "nearest_normalized_distance": self.nearest_normalized_distance,
            "weighted_normalized_distance": self.weighted_normalized_distance,
            "uncertainty_ms": self.uncertainty_ms,
            "relative_uncertainty": self.relative_uncertainty,
            "exact_match": self.exact_match,
            "neighbor_point_ids": list(self.neighbor_point_ids),
            "neighbor_distances": list(self.neighbor_distances),
            "neighbor_weights": list(self.neighbor_weights),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"latency_ms": self.latency_ms, **self.diagnostics()}


@dataclass(frozen=True)
class _TrainingRange:
    minimum: int
    maximum: int

    @property
    def span(self) -> int:
        return self.maximum - self.minimum


class CalibratedBatchTimeModel:
    """Deterministic inverse-distance interpolation over hardware profile points.

    Use :meth:`from_json` or :meth:`from_dict`; both require the caller's exact
    runtime identity.  ``predict`` rejects descriptors outside the observed
    feature envelope, descriptors farther than the profile's declared nearest
    neighbor threshold, and locally uncertain predictions.
    """

    def __init__(
        self,
        *,
        identity: BatchTimeProfileIdentity,
        points: tuple[BatchTimeProfilePoint, ...],
        training_ranges: Mapping[str, _TrainingRange],
        feature_weights: Mapping[str, float],
        feature_transforms: Mapping[str, str],
        neighbors: int,
        distance_power: float,
        max_normalized_distance: float,
        max_relative_uncertainty: float,
        measurement: Mapping[str, Any],
    ):
        self.identity = identity
        self.points = points
        self.training_ranges = dict(training_ranges)
        self.feature_weights = dict(feature_weights)
        self.feature_transforms = dict(feature_transforms)
        self.neighbors = neighbors
        self.distance_power = distance_power
        self.max_normalized_distance = max_normalized_distance
        self.max_relative_uncertainty = max_relative_uncertainty
        self.measurement = dict(measurement)
        varying_weight = sum(
            self.feature_weights[name]
            for name in FEATURE_NAMES
            if self.training_ranges[name].span > 0
        )
        if varying_weight <= 0:
            raise ProfileSchemaError("profile must vary at least one batch feature")
        self._varying_feature_weight = varying_weight
        self._transformed_training_ranges: dict[str, tuple[float, float]] = {}
        for name in FEATURE_NAMES:
            transformed_minimum = _apply_feature_transform(
                self.training_ranges[name].minimum,
                self.feature_transforms[name],
            )
            transformed_maximum = _apply_feature_transform(
                self.training_ranges[name].maximum,
                self.feature_transforms[name],
            )
            if (
                self.training_ranges[name].span > 0
                and transformed_maximum <= transformed_minimum
            ):
                raise ProfileSchemaError(
                    f"profile.interpolation.feature_transforms.{name} loses "
                    "resolution across its training range"
                )
            self._transformed_training_ranges[name] = (
                transformed_minimum,
                transformed_maximum,
            )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_identity: BatchTimeProfileIdentity | None = None,
    ) -> "CalibratedBatchTimeModel":
        """Load a strict JSON profile, optionally matching a runtime identity.

        The no-argument compatibility form still validates every identity field
        in the profile.  Production evaluators should pass ``expected_identity``
        from their pinned runtime manifest to reject a valid profile for the
        wrong device, model, or software revision.
        """

        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ProfileSchemaError(f"batch time profile must be JSON: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(
                    handle,
                    object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=_reject_nonstandard_json_number,
                )
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileSchemaError(f"cannot load batch time profile {path}: {exc}") from exc
        if expected_identity is None:
            profile = _require_mapping(payload, "profile")
            identity_payload = _require_mapping(
                profile.get("identity"), "profile.identity"
            )
            expected_identity = BatchTimeProfileIdentity.from_mapping(identity_payload)
        return cls.from_dict(payload, expected_identity=expected_identity)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_identity: BatchTimeProfileIdentity,
    ) -> "CalibratedBatchTimeModel":
        """Load JSON while requiring an exact caller-supplied runtime identity."""

        return cls.from_path(path, expected_identity=expected_identity)

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        expected_identity: BatchTimeProfileIdentity,
    ) -> "CalibratedBatchTimeModel":
        """Validate an already-decoded profile without accepting compatibility defaults."""

        if not isinstance(expected_identity, BatchTimeProfileIdentity):
            raise TypeError("expected_identity must be a BatchTimeProfileIdentity")
        profile = _require_mapping(payload, "profile")
        _require_exact_keys(
            profile,
            {
                "schema_version",
                "identity",
                "measurement",
                "feature_order",
                "training_ranges",
                "interpolation",
                "points",
            },
            "profile",
        )
        schema_version = _require_int(
            profile["schema_version"], "profile.schema_version", minimum=1
        )
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileSchemaError(
                f"unsupported profile.schema_version {schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        feature_order = profile["feature_order"]
        if not isinstance(feature_order, list) or tuple(feature_order) != FEATURE_NAMES:
            raise ProfileSchemaError(
                f"profile.feature_order must be exactly {list(FEATURE_NAMES)!r}"
            )

        identity = BatchTimeProfileIdentity.from_mapping(
            _require_mapping(profile["identity"], "profile.identity")
        )
        mismatches = _identity_mismatches(expected_identity, identity)
        if mismatches:
            raise ProfileIdentityError(
                "batch time profile identity mismatch: " + "; ".join(mismatches)
            )

        measurement = _parse_measurement(profile["measurement"])
        ranges = _parse_training_ranges(profile["training_ranges"])
        interpolation = _parse_interpolation(profile["interpolation"])
        points = _parse_points(
            profile["points"],
            minimum_samples=measurement["minimum_measured_iterations"],
            max_model_len=identity.max_model_len,
        )
        if interpolation["neighbors"] > len(points):
            raise ProfileSchemaError(
                "profile.interpolation.neighbors cannot exceed the number of profile points"
            )
        _validate_ranges_against_points(ranges, points)

        return cls(
            identity=identity,
            points=points,
            training_ranges=ranges,
            feature_weights=interpolation["feature_weights"],
            feature_transforms=interpolation["feature_transforms"],
            neighbors=interpolation["neighbors"],
            distance_power=interpolation["distance_power"],
            max_normalized_distance=interpolation["max_normalized_distance"],
            max_relative_uncertainty=interpolation["max_relative_uncertainty"],
            measurement=measurement,
        )

    def predict(
        self,
        batch: BatchDescriptor | Mapping[str, Any],
    ) -> BatchTimePrediction:
        """Predict latency or reject the descriptor with auditable diagnostics."""

        descriptor = (
            batch if isinstance(batch, BatchDescriptor) else BatchDescriptor.from_mapping(batch)
        )
        descriptor.validate(max_model_len=self.identity.max_model_len)
        distances = sorted(
            (
                (self._normalized_distance(descriptor, point.descriptor), point)
                for point in self.points
            ),
            key=lambda item: (item[0], item[1].descriptor.feature_tuple(), item[1].point_id),
        )
        nearest_distance = distances[0][0]
        range_violations = self._range_violations(descriptor)
        if range_violations:
            raise OutOfDistributionError(
                "batch descriptor is outside the measured training range: "
                + "; ".join(range_violations),
                {
                    "nearest_normalized_distance": nearest_distance,
                    "range_violations": range_violations,
                    "max_normalized_distance": self.max_normalized_distance,
                },
            )
        if nearest_distance > self.max_normalized_distance:
            raise OutOfDistributionError(
                "batch descriptor is too far from measured profile points: "
                f"nearest normalized distance {nearest_distance:.6g} exceeds "
                f"{self.max_normalized_distance:.6g}",
                {
                    "nearest_normalized_distance": nearest_distance,
                    "max_normalized_distance": self.max_normalized_distance,
                    "nearest_point_id": distances[0][1].point_id,
                },
            )

        exact = next(
            (
                point
                for distance, point in distances
                if distance == 0.0 and point.descriptor == descriptor
            ),
            None,
        )
        if exact is not None:
            standard_error = exact.latency_stddev_ms / math.sqrt(exact.sample_count)
            prediction = BatchTimePrediction(
                latency_ms=exact.latency_ms,
                nearest_normalized_distance=0.0,
                weighted_normalized_distance=0.0,
                uncertainty_ms=standard_error,
                relative_uncertainty=standard_error / exact.latency_ms,
                exact_match=True,
                neighbor_point_ids=(exact.point_id,),
                neighbor_distances=(0.0,),
                neighbor_weights=(1.0,),
            )
            self._validate_prediction_uncertainty(prediction)
            return prediction

        selected = distances[: self.neighbors]
        raw_weights = [distance ** (-self.distance_power) for distance, _ in selected]
        weight_total = sum(raw_weights)
        if not math.isfinite(weight_total) or weight_total <= 0:
            raise PredictionUncertaintyError(
                "non-finite interpolation weights",
                {"nearest_normalized_distance": nearest_distance},
            )
        weights = [weight / weight_total for weight in raw_weights]
        latency_ms = sum(
            weight * point.latency_ms
            for weight, (_, point) in zip(weights, selected)
        )
        weighted_distance = sum(
            weight * distance
            for weight, (distance, _) in zip(weights, selected)
        )
        local_spread = math.sqrt(
            sum(
                weight * (point.latency_ms - latency_ms) ** 2
                for weight, (_, point) in zip(weights, selected)
            )
        )
        measurement_error = math.sqrt(
            sum(
                (
                    weight
                    * point.latency_stddev_ms
                    / math.sqrt(point.sample_count)
                )
                ** 2
                for weight, (_, point) in zip(weights, selected)
            )
        )
        uncertainty_ms = math.hypot(local_spread, measurement_error)
        if not all(
            math.isfinite(value) and value >= 0
            for value in (latency_ms, weighted_distance, uncertainty_ms)
        ) or latency_ms <= 0:
            raise PredictionUncertaintyError(
                "interpolation produced non-finite diagnostics",
                {"nearest_normalized_distance": nearest_distance},
            )
        prediction = BatchTimePrediction(
            latency_ms=latency_ms,
            nearest_normalized_distance=nearest_distance,
            weighted_normalized_distance=weighted_distance,
            uncertainty_ms=uncertainty_ms,
            relative_uncertainty=uncertainty_ms / latency_ms,
            exact_match=False,
            neighbor_point_ids=tuple(point.point_id for _, point in selected),
            neighbor_distances=tuple(distance for distance, _ in selected),
            neighbor_weights=tuple(weights),
        )
        self._validate_prediction_uncertainty(prediction)
        return prediction

    def validate_identity(self, expected_identity: BatchTimeProfileIdentity) -> None:
        """Reject use under a device, runtime, or model other than the profiled one."""

        if not isinstance(expected_identity, BatchTimeProfileIdentity):
            raise TypeError("expected_identity must be a BatchTimeProfileIdentity")
        mismatches = _identity_mismatches(expected_identity, self.identity)
        if mismatches:
            raise ProfileIdentityError(
                "batch time profile identity mismatch: " + "; ".join(mismatches)
            )

    def _normalized_distance(
        self,
        left: BatchDescriptor,
        right: BatchDescriptor,
    ) -> float:
        squared = 0.0
        for name in FEATURE_NAMES:
            training_range = self.training_ranges[name]
            if training_range.span == 0:
                continue
            transformed_minimum, transformed_maximum = (
                self._transformed_training_ranges[name]
            )
            transformed_span = transformed_maximum - transformed_minimum
            left_value = _apply_feature_transform(
                getattr(left, name), self.feature_transforms[name]
            )
            right_value = _apply_feature_transform(
                getattr(right, name), self.feature_transforms[name]
            )
            delta = (left_value - right_value) / transformed_span
            squared += self.feature_weights[name] * delta * delta
        return math.sqrt(squared / self._varying_feature_weight)

    def _range_violations(self, descriptor: BatchDescriptor) -> list[str]:
        violations: list[str] = []
        for name in FEATURE_NAMES:
            value = getattr(descriptor, name)
            training_range = self.training_ranges[name]
            if value < training_range.minimum or value > training_range.maximum:
                violations.append(
                    f"{name}={value} not in "
                    f"[{training_range.minimum}, {training_range.maximum}]"
                )
        return violations

    def _validate_prediction_uncertainty(self, prediction: BatchTimePrediction) -> None:
        if (
            not math.isfinite(prediction.relative_uncertainty)
            or prediction.relative_uncertainty > self.max_relative_uncertainty
        ):
            raise PredictionUncertaintyError(
                "predicted latency is insufficiently supported: relative uncertainty "
                f"{prediction.relative_uncertainty:.6g} exceeds "
                f"{self.max_relative_uncertainty:.6g}",
                {
                    **prediction.diagnostics(),
                    "latency_ms": prediction.latency_ms,
                    "max_relative_uncertainty": self.max_relative_uncertainty,
                },
            )


def _apply_feature_transform(value: int, transform: str) -> float:
    if transform == "linear":
        return float(value)
    if transform == "log1p":
        return math.log1p(value)
    raise ProfileSchemaError(f"unsupported feature transform {transform!r}")


def _parse_measurement(payload: Any) -> dict[str, Any]:
    measurement = _require_mapping(payload, "profile.measurement")
    keys = {
        "latency_unit",
        "latency_statistic",
        "timer",
        "warmup_iterations",
        "minimum_measured_iterations",
        "clock_policy",
        "raw_artifact_sha256",
    }
    _require_exact_keys(measurement, keys, "profile.measurement")
    if measurement["latency_unit"] != "ms":
        raise ProfileSchemaError("profile.measurement.latency_unit must be 'ms'")
    if measurement["latency_statistic"] != "mean":
        raise ProfileSchemaError("profile.measurement.latency_statistic must be 'mean'")
    raw_hash = _require_string(
        measurement["raw_artifact_sha256"],
        "profile.measurement.raw_artifact_sha256",
    )
    if len(raw_hash) != 64 or any(character not in "0123456789abcdef" for character in raw_hash):
        raise ProfileSchemaError(
            "profile.measurement.raw_artifact_sha256 must be 64 lowercase hex characters"
        )
    return {
        "latency_unit": "ms",
        "latency_statistic": "mean",
        "timer": _require_string(measurement["timer"], "profile.measurement.timer"),
        "warmup_iterations": _require_int(
            measurement["warmup_iterations"],
            "profile.measurement.warmup_iterations",
            minimum=1,
        ),
        "minimum_measured_iterations": _require_int(
            measurement["minimum_measured_iterations"],
            "profile.measurement.minimum_measured_iterations",
            minimum=3,
        ),
        "clock_policy": _require_string(
            measurement["clock_policy"], "profile.measurement.clock_policy"
        ),
        "raw_artifact_sha256": raw_hash,
    }


def _parse_training_ranges(payload: Any) -> dict[str, _TrainingRange]:
    ranges = _require_mapping(payload, "profile.training_ranges")
    _require_exact_keys(ranges, set(FEATURE_NAMES), "profile.training_ranges")
    result: dict[str, _TrainingRange] = {}
    for name in FEATURE_NAMES:
        spec = _require_mapping(ranges[name], f"profile.training_ranges.{name}")
        _require_exact_keys(spec, {"min", "max"}, f"profile.training_ranges.{name}")
        minimum = _require_int(
            spec["min"], f"profile.training_ranges.{name}.min", minimum=0
        )
        maximum = _require_int(
            spec["max"], f"profile.training_ranges.{name}.max", minimum=0
        )
        if minimum > maximum:
            raise ProfileSchemaError(
                f"profile.training_ranges.{name}.min cannot exceed max"
            )
        if name in {"batch_size", "total_tokens"}:
            if minimum <= 0:
                raise ProfileSchemaError(
                    f"profile.training_ranges.{name}.min must be positive"
                )
        result[name] = _TrainingRange(minimum, maximum)
    return result


def _parse_interpolation(payload: Any) -> dict[str, Any]:
    interpolation = _require_mapping(payload, "profile.interpolation")
    keys = {
        "method",
        "neighbors",
        "distance_power",
        "max_normalized_distance",
        "max_relative_uncertainty",
        "feature_weights",
        "feature_transforms",
    }
    _require_exact_keys(interpolation, keys, "profile.interpolation")
    if interpolation["method"] != "inverse_distance_weighting_v1":
        raise ProfileSchemaError(
            "profile.interpolation.method must be 'inverse_distance_weighting_v1'"
        )
    weights_payload = _require_mapping(
        interpolation["feature_weights"], "profile.interpolation.feature_weights"
    )
    _require_exact_keys(
        weights_payload,
        set(FEATURE_NAMES),
        "profile.interpolation.feature_weights",
    )
    weights = {
        name: _require_finite_number(
            weights_payload[name],
            f"profile.interpolation.feature_weights.{name}",
            minimum_exclusive=0.0,
        )
        for name in FEATURE_NAMES
    }
    transforms_payload = _require_mapping(
        interpolation["feature_transforms"],
        "profile.interpolation.feature_transforms",
    )
    _require_exact_keys(
        transforms_payload,
        set(FEATURE_NAMES),
        "profile.interpolation.feature_transforms",
    )
    transforms: dict[str, str] = {}
    for name in FEATURE_NAMES:
        transform = _require_string(
            transforms_payload[name],
            f"profile.interpolation.feature_transforms.{name}",
        )
        if transform not in ALLOWED_FEATURE_TRANSFORMS:
            allowed = ", ".join(sorted(ALLOWED_FEATURE_TRANSFORMS))
            raise ProfileSchemaError(
                f"profile.interpolation.feature_transforms.{name} must be one of: "
                f"{allowed}"
            )
        transforms[name] = transform
    distance_power = _require_finite_number(
        interpolation["distance_power"],
        "profile.interpolation.distance_power",
        minimum_exclusive=0.0,
    )
    if distance_power > 8:
        raise ProfileSchemaError("profile.interpolation.distance_power cannot exceed 8")
    max_distance = _require_finite_number(
        interpolation["max_normalized_distance"],
        "profile.interpolation.max_normalized_distance",
        minimum_exclusive=0.0,
    )
    if max_distance > 1:
        raise ProfileSchemaError(
            "profile.interpolation.max_normalized_distance cannot exceed 1"
        )
    max_uncertainty = _require_finite_number(
        interpolation["max_relative_uncertainty"],
        "profile.interpolation.max_relative_uncertainty",
        minimum_exclusive=0.0,
    )
    if max_uncertainty > 1:
        raise ProfileSchemaError(
            "profile.interpolation.max_relative_uncertainty cannot exceed 1"
        )
    return {
        "neighbors": _require_int(
            interpolation["neighbors"], "profile.interpolation.neighbors", minimum=2
        ),
        "distance_power": distance_power,
        "max_normalized_distance": max_distance,
        "max_relative_uncertainty": max_uncertainty,
        "feature_weights": weights,
        "feature_transforms": transforms,
    }


def _parse_points(
    payload: Any,
    *,
    minimum_samples: int,
    max_model_len: int,
) -> tuple[BatchTimeProfilePoint, ...]:
    if not isinstance(payload, list) or len(payload) < 2:
        raise ProfileSchemaError("profile.points must contain at least two points")
    expected_keys = {
        "point_id",
        *FEATURE_NAMES,
        "latency_ms",
        "latency_stddev_ms",
        "sample_count",
    }
    points: list[BatchTimeProfilePoint] = []
    point_ids: set[str] = set()
    descriptors: set[tuple[int, ...]] = set()
    for index, item in enumerate(payload):
        label = f"profile.points[{index}]"
        point = _require_mapping(item, label)
        _require_exact_keys(point, expected_keys, label)
        point_id = _require_string(point["point_id"], f"{label}.point_id")
        if point_id in point_ids:
            raise ProfileSchemaError(f"duplicate profile point_id {point_id!r}")
        descriptor = BatchDescriptor.from_mapping(
            {name: point[name] for name in FEATURE_NAMES}
        )
        descriptor.validate(max_model_len=max_model_len)
        feature_tuple = descriptor.feature_tuple()
        if feature_tuple in descriptors:
            raise ProfileSchemaError(
                f"profile points must aggregate repeated descriptors; duplicate at {label}"
            )
        sample_count = _require_int(
            point["sample_count"], f"{label}.sample_count", minimum=minimum_samples
        )
        points.append(
            BatchTimeProfilePoint(
                point_id=point_id,
                descriptor=descriptor,
                latency_ms=_require_finite_number(
                    point["latency_ms"], f"{label}.latency_ms", minimum_exclusive=0.0
                ),
                latency_stddev_ms=_require_finite_number(
                    point["latency_stddev_ms"],
                    f"{label}.latency_stddev_ms",
                    minimum_inclusive=0.0,
                ),
                sample_count=sample_count,
            )
        )
        point_ids.add(point_id)
        descriptors.add(feature_tuple)
    return tuple(points)


def _validate_ranges_against_points(
    ranges: Mapping[str, _TrainingRange],
    points: tuple[BatchTimeProfilePoint, ...],
) -> None:
    for name in FEATURE_NAMES:
        observed = [getattr(point.descriptor, name) for point in points]
        declared = ranges[name]
        observed_min = min(observed)
        observed_max = max(observed)
        if declared.minimum != observed_min or declared.maximum != observed_max:
            raise ProfileSchemaError(
                f"profile.training_ranges.{name} must equal observed point range "
                f"[{observed_min}, {observed_max}], got "
                f"[{declared.minimum}, {declared.maximum}]"
            )


def _identity_mismatches(
    expected: BatchTimeProfileIdentity,
    actual: BatchTimeProfileIdentity,
) -> list[str]:
    expected_values = expected.to_flat_dict()
    actual_values = actual.to_flat_dict()
    return [
        f"{name}: expected {expected_values[name]!r}, profile has {actual_values[name]!r}"
        for name in expected_values
        if expected_values[name] != actual_values[name]
    ]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProfileSchemaError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ProfileSchemaError(f"{label} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ProfileSchemaError(f"{label} has invalid fields: {'; '.join(details)}")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProfileSchemaError(f"{label} must be a non-empty trimmed string")
    return value


def _require_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileSchemaError(f"{label} must be an integer")
    if value < minimum:
        raise ProfileSchemaError(f"{label} must be at least {minimum}")
    if value > MAX_PROFILE_INTEGER:
        raise ProfileSchemaError(
            f"{label} exceeds the largest exactly represented JSON integer "
            f"{MAX_PROFILE_INTEGER}"
        )
    return value


def _require_finite_number(
    value: Any,
    label: str,
    *,
    minimum_exclusive: float | None = None,
    minimum_inclusive: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileSchemaError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileSchemaError(f"{label} must be finite")
    if minimum_exclusive is not None and result <= minimum_exclusive:
        raise ProfileSchemaError(f"{label} must be greater than {minimum_exclusive}")
    if minimum_inclusive is not None and result < minimum_inclusive:
        raise ProfileSchemaError(f"{label} must be at least {minimum_inclusive}")
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileSchemaError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_number(value: str) -> None:
    raise ProfileSchemaError(f"non-standard JSON number {value!r} is not allowed")
