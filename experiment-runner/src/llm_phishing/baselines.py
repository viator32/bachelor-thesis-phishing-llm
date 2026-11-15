from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .feature_extractor import CREDENTIALS, INJECTION, PAYMENT, URGENCY
from .schemas import (
    ComponentResult,
    Evidence,
    PartialPrediction,
    PhishingType,
    ProcessedEmail,
)


# Combine subject, sender, reply-to and body.
def email_text(email: ProcessedEmail) -> str:
    return f"Subject: {email.subject}\nFrom: {email.sender_address}\nReply-To: {email.reply_to}\n{email.text}"


# Apply fixed, uncalibrated rule scores to an email.
def rules_predict(email: ProcessedEmail) -> ComponentResult:
    text = email_text(email)
    score = 0
    evidence: list[Evidence] = []

    # Add points and evidence when a pattern matches the enclosing text.
    def add(pattern: re.Pattern, points: int, category: str) -> None:
        nonlocal score
        match = pattern.search(text)
        if match:
            score += points
            evidence.append(Evidence(field="body", category=category, excerpt=match.group(0)[:200]))

    add(URGENCY, 15, "urgency")
    add(CREDENTIALS, 30, "credential_request")
    add(PAYMENT, 25, "financial_request")
    add(INJECTION, 35, "instruction_override")
    mismatch_count = int(email.features.get("domain_mismatch_count", 0))
    reply_mismatch = bool(email.features.get("sender_reply_to_mismatch", False))
    if mismatch_count:
        score += min(25, mismatch_count * 15)
        evidence.append(
            Evidence(field="url", category="domain_mismatch", excerpt="offline URL mismatch")
        )
    if reply_mismatch:
        score += 15
        evidence.append(
            Evidence(field="reply_to", category="sender_mismatch", excerpt=email.reply_to[:200])
        )
    score = min(100, score)
    injection = bool(INJECTION.search(text))
    is_phishing = score >= 40
    personalization = min(1, int(email.features.get("personalization_hits", 0)))
    phishing_type = (
        PhishingType.spear
        if is_phishing and personalization
        else (PhishingType.classic if is_phishing else PhishingType.none)
    )
    prediction = PartialPrediction(
        is_phishing=is_phishing,
        phishing_type=phishing_type,
        personalization_level=personalization,
        prompt_injection_detected=injection,
        injection_goals=["task_override"] if injection else [],
        phishing_risk_score=score,
        prompt_injection_risk_score=min(100, int(email.features.get("injection_hits", 0)) * 60),  # heuristic, uncalibrated
        evidence=evidence[:8],
        explanation="Deterministic offline rule score; AI-generation attribution is not applicable.",
    )
    return ComponentResult(
        method="rules_only",
        prediction=prediction,
        details={
            "supported_tasks": ["is_phishing", "phishing_type", "personalization_level", "prompt_injection_detected"],
            "unsupported_tasks": ["generation_origin"],
            "threshold": 40,
        },
    )


class TfidfLogRegBaseline:
    """Train, load and evaluate the TF-IDF baseline."""

    # Use the supplied pipeline or create the TF-IDF classifier.
    def __init__(self, model: Pipeline | None = None, seed: int = 42) -> None:
        self.model = model or Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50_000)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed),
                ),
            ]
        )

    # Fit the classifier on labeled emails containing both classes.
    def train(self, emails: Iterable[ProcessedEmail]) -> TfidfLogRegBaseline:
        rows = [
            email for email in emails
            if email.ground_truth is not None and email.ground_truth.is_phishing is not None
        ]
        if len({email.ground_truth.is_phishing for email in rows}) < 2:
            raise ValueError(
                "TF-IDF training requires at least one phishing and one legitimate email"
            )
        self.model.fit(
            [email_text(email) for email in rows],
            [email.ground_truth.is_phishing for email in rows],
        )
        return self

    # Save the pipeline, creating its parent directory.
    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target)

    # Load a saved classifier pipeline.
    @classmethod
    def load(cls, path: str | Path) -> TfidfLogRegBaseline:
        return cls(joblib.load(path))

    # Classify an email with the fitted TF-IDF pipeline.
    def predict(self, email: ProcessedEmail) -> ComponentResult:
        probability = float(self.model.predict_proba([email_text(email)])[0][1])
        is_phishing = probability >= 0.5
        prediction = PartialPrediction(
            is_phishing=is_phishing,
            phishing_risk_score=max(0, min(100, math.floor(probability * 100))),
            explanation="TF-IDF logistic-regression phishing baseline.",
        )
        return ComponentResult(
            method="tfidf_logreg",
            prediction=prediction,
            details={
                "phishing_probability": probability,
                "supported_tasks": ["is_phishing"],
                "unsupported_tasks": [
                    "phishing_type",
                    "generation_origin",
                    "personalization_level",
                    "prompt_injection_detected",
                ],
            },
        )
