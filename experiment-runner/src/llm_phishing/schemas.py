from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PhishingType(str, Enum):
    none = "none"
    classic = "classic"
    spear = "spear"
    unknown = "unknown"


class GenerationOrigin(str, Enum):
    human = "human"
    ai = "ai"
    unknown = "unknown"


class InjectionGoal(str, Enum):
    label_manipulation = "label_manipulation"
    task_override = "task_override"
    format_manipulation = "format_manipulation"
    prompt_extraction = "prompt_extraction"
    warning_suppression = "warning_suppression"
    analysis_termination = "analysis_termination"
    other = "other"


class GroundTruth(BaseModel):
    # External injection/intent corpora may not annotate phishing at all.
    is_phishing: bool | None
    phishing_type: PhishingType
    generation_origin: GenerationOrigin
    # "unknown" is allowed only here: the dataset annotation (P0/P1) may be missing.
    personalization_level: int | Literal["unknown"] = "unknown"
    has_prompt_injection: bool | None = None
    injection_goals: list[InjectionGoal] = Field(default_factory=list)

    # Validate the instance personalization level.
    @model_validator(mode="after")
    def validate_personalization(self) -> GroundTruth:
        if isinstance(self.personalization_level, int) and self.personalization_level not in (0, 1):
            raise ValueError("personalization_level must be 0, 1 or unknown")
        return self


class RawEmail(BaseModel):
    instance_id: str
    raw_source: str | None = None
    sender_name: str = ""
    sender_address: str = ""
    reply_to: str = ""
    subject: str = ""
    text_body: str = ""
    html_body: str = ""
    date: str = ""
    language: str = "en"
    source_dataset: str = "unknown"
    split: str = "unspecified"
    base_id: str | None = None
    campaign_id: str | None = None
    template_id: str | None = None
    profile_id: str | None = None
    generator_model: str | None = None
    ground_truth: GroundTruth | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedURL(BaseModel):
    displayed_text: str = ""
    target_url: str
    domain: str = ""
    display_domain: str = ""
    domain_mismatch: bool = False


class ProcessedEmail(BaseModel):
    instance_id: str
    sender_name: str
    sender_address: str
    reply_to: str
    subject: str
    text: str
    date: str
    language: str
    urls: list[ExtractedURL] = Field(default_factory=list)
    features: dict[str, float | int | bool | str] = Field(default_factory=dict)
    transformations: list[dict[str, Any]] = Field(default_factory=list)
    source_dataset: str = "unknown"
    split: str = "unspecified"
    base_id: str | None = None
    ground_truth: GroundTruth | None = None

    # Whitelist email content, excluding all evaluation metadata.
    def prompt_view(self) -> dict[str, Any]:
        return {
            "sender_name": self.sender_name,
            "sender_address": self.sender_address,
            "reply_to": self.reply_to,
            "subject": self.subject,
            "text": self.text,
            "urls": [url.model_dump() for url in self.urls],
            "date": self.date,
        }


class GenerationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    human: float = Field(ge=0, le=1)
    ai: float = Field(ge=0, le=1)
    unknown: float = Field(ge=0, le=1)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["sender", "reply_to", "subject", "body", "url", "other"]
    category: str = Field(min_length=1, max_length=80)
    excerpt: str = Field(min_length=1, max_length=240)


class LLMPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_phishing: bool
    phishing_type: PhishingType
    generation_origin: GenerationOrigin
    generation_scores: GenerationScores
    personalization_level: int = Field(ge=0, le=2)
    prompt_injection_detected: bool
    injection_goals: list[InjectionGoal] = Field(default_factory=list)
    phishing_risk_score: int = Field(ge=0, le=100)
    prompt_injection_risk_score: int = Field(ge=0, le=100)
    evidence: list[Evidence] = Field(default_factory=list, max_length=8)
    explanation: str = Field(min_length=1, max_length=800)


class PartialPrediction(BaseModel):
    """Task adapters retain the full result shape without inventing unsupported values."""
    model_config = ConfigDict(extra="forbid")
    is_phishing: bool | None = None
    phishing_type: PhishingType | None = None
    generation_origin: GenerationOrigin | None = None
    generation_scores: GenerationScores | None = None
    personalization_level: int | None = Field(default=None, ge=0, le=2)
    prompt_injection_detected: bool | None = None
    injection_goals: list[InjectionGoal] | None = None
    phishing_risk_score: int | None = Field(default=None, ge=0, le=100)
    prompt_injection_risk_score: int | None = Field(default=None, ge=0, le=100)
    evidence: list[Evidence] | None = None
    explanation: str | None = None


class ValidationReport(BaseModel):
    json_syntax_valid: bool = False
    schema_valid: bool = False
    score_sum_valid: bool = False
    semantic_consistency_valid: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Combine syntax, schema, score-sum and semantic validation flags.
    # Select the configured standalone or hybrid baseline.
    # Check whether the configured method needs a model.
    # Combine syntax, schema, score-sum and semantic validation flags.
    # Select the configured standalone or hybrid baseline.
    # Check whether the configured method needs a model.
    @property
    def fully_valid(self) -> bool:
        return all(
            (
                self.json_syntax_valid,
                self.schema_valid,
                self.score_sum_valid,
                self.semantic_consistency_valid,
            )
        )


class TechnicalMeasurements(BaseModel):
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    time_to_first_token_ms: float | None = None
    prompt_duration_ms: float | None = None
    generation_duration_ms: float | None = None
    inference_duration_ms: float | None = None
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    process_rss_mb_before: float | None = None
    process_rss_mb_after: float | None = None
    backend_version: str | None = None


class ComponentResult(BaseModel):
    method: str
    prediction: LLMPrediction | PartialPrediction | None = None
    raw_response: str | None = None
    repaired_response: str | None = None
    validation_initial: ValidationReport | None = None
    validation_repaired: ValidationReport | None = None
    measurements: TechnicalMeasurements = Field(default_factory=TechnicalMeasurements)
    status: Literal["success", "error", "timeout", "not_applicable"] = "success"
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] | None = None


class ExperimentRecord(BaseModel):
    run_id: str
    instance_id: str
    config_id: str
    model_id: str
    prompt_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_dataset: str
    split: str
    base_id: str | None = None
    ground_truth: GroundTruth | None = None
    raw_email: RawEmail
    processed_email: ProcessedEmail
    components: dict[str, ComponentResult]
    final_result: ComponentResult
    reproducibility: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    config_id: str
    task: Literal["phishing", "injection"] = "phishing"
    output_contract: Literal["comparison_v1"] = "comparison_v1"
    response_schema: Literal["full_email_v1"] = "full_email_v1"
    method: Literal[
        "rules_only",
        "tfidf_logreg",
    ]
    model: str = "Qwen3-4B-4bit"
    backend: Literal["mlx"] = "mlx"
    mlx_hf_repo: str | None = None
    mlx_model_id: str | None = None
    constrained_json: bool = True
    allow_repair: bool = True
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = 42
    timeout_seconds: int = 180
    max_input_tokens: int | None = Field(default=4096, ge=1)
    max_output_tokens: int = Field(default=1024, ge=128, le=4096)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    prompt_name: Literal["basic", "security"] = "security"
    dataset_path: str
    personalization_annotations: str | None = None
    split: str | None = "test"
    limit: int | None = Field(default=None, ge=1)
    resume: bool = True
    tfidf_model_path: str = "models/tfidf_logreg.joblib"

    @model_validator(mode="after")
    def supported_baseline_task(self) -> ExperimentConfig:
        if self.method == "tfidf_logreg" and self.task != "phishing":
            raise ValueError("The saved TF-IDF classifier supports phishing detection only.")
        return self

    @property
    def baseline_component(self) -> str | None:
        if self.method in {"rules_only", "rules_plus_llm"}:
            return "rules"
        if self.method in {"tfidf_logreg", "tfidf_plus_llm"}:
            return "tfidf"
        return None

    @property
    def uses_llm(self) -> bool:
        return self.method not in {"rules_only", "tfidf_logreg"}
