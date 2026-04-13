"""Validated configuration loading for Valerie."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ValerieBaseModel(BaseModel):
    """Base config model with strict defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class QuantizationConfig(ValerieBaseModel):
    mode: Literal["none", "4bit", "8bit"] = "none"


class ModelConfig(ValerieBaseModel):
    name: str
    backend: Literal["transformer_lens", "dummy"] = "transformer_lens"
    model_name: str
    device_preference: list[Literal["mps", "cuda", "cpu"]] = Field(
        default_factory=lambda: ["mps", "cuda", "cpu"]
    )
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    default_prepend_bos: bool = True
    trust_remote_code: bool = False
    n_ctx: int | None = None
    first_n_layers: int | None = None
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    cache_dir: str | None = None
    dummy_d_model: int = 16
    dummy_n_layers: int = 4
    seed: int = 0

    @field_validator("device_preference")
    @classmethod
    def validate_device_preference(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("device_preference must include at least one device")
        return value


class ExtractionPositionConfig(ValerieBaseModel):
    strategy: Literal["all", "last", "index", "mean_pool"] = "last"
    index: int | None = None

    @model_validator(mode="after")
    def validate_index_requirement(self) -> ExtractionPositionConfig:
        if self.strategy == "index" and self.index is None:
            raise ValueError("position.index is required when strategy='index'")
        return self


class ExtractionConfig(ValerieBaseModel):
    components: list[
        Literal[
            "resid_pre",
            "resid_mid",
            "resid_post",
            "mlp_pre",
            "mlp_post",
            "attn_pattern",
            "head_result",
        ]
    ] = Field(default_factory=lambda: ["resid_post"])
    layers: list[int] | Literal["all"] = "all"
    position: ExtractionPositionConfig = Field(default_factory=ExtractionPositionConfig)
    save_logits: bool = True


class ConditionConfig(ValerieBaseModel):
    description: str
    target: Literal["self", "narrative", "neutral"] = "neutral"
    variants: list[str]

    @field_validator("variants")
    @classmethod
    def validate_variants(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("each condition requires at least one prompt variant")
        return value


class TaskConfig(ValerieBaseModel):
    id: str
    prompt: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ExperimentControlConfig(ValerieBaseModel):
    require_matched_variant_counts: bool = True
    warn_on_char_length_delta_over: int = 24
    deterministic_seed: int = 0


class ExperimentConfig(ValerieBaseModel):
    name: str
    description: str
    extraction: ExtractionConfig
    conditions: dict[str, ConditionConfig]
    tasks: list[TaskConfig]
    controls: ExperimentControlConfig = Field(default_factory=ExperimentControlConfig)

    @model_validator(mode="after")
    def validate_conditions(self) -> ExperimentConfig:
        if "neutral" not in self.conditions:
            raise ValueError("experiments must define a neutral condition")
        if len(self.tasks) == 0:
            raise ValueError("experiments must define at least one task")
        if self.controls.require_matched_variant_counts:
            counts = {name: len(condition.variants) for name, condition in self.conditions.items()}
            if len(set(counts.values())) != 1:
                raise ValueError(
                    "all conditions must have the same number of variants when "
                    "require_matched_variant_counts=true"
                )
        return self


class ProbeConfig(ValerieBaseModel):
    name: str
    task: Literal["binary_classification", "regression", "multiclass_classification"] = (
        "multiclass_classification"
    )
    input_component: str = "resid_post"
    framing_conditions: list[str] = Field(default_factory=lambda: ["threat", "care", "neutral"])
    narrative_reference_condition: str = "threat"
    narrative_control_condition: str = "narrative_threat_control"
    regularization: float = 1.0
    max_iter: int = 1000
    solver: Literal["lbfgs", "liblinear", "saga"] = "lbfgs"
    random_seed: int = 0
    num_permutations: int = 10
    pca_components: int = 3
    clustering_method: Literal["kmeans", "agglomerative"] = "kmeans"
    save_models: bool = True

    @field_validator("framing_conditions")
    @classmethod
    def validate_framing_conditions(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("framing_conditions must include at least two labels")
        return value


def _load_yaml_file(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at top level of {path}")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml_file(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_yaml_file(path))


def load_probe_config(path: str | Path) -> ProbeConfig:
    return ProbeConfig.model_validate(_load_yaml_file(path))
