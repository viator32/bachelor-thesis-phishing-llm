from __future__ import annotations

import re

from .schemas import ProcessedEmail

# These patterns double as the rule-based scoring signals in baselines.py (rules_predict
# imports them directly), so a change here also changes the rules_only baseline's behavior.
URGENCY = re.compile(
    r"\b(urgent|immediately|today|within 24 hours|act now|deadline)\b", re.IGNORECASE
)
CREDENTIALS = re.compile(
    r"\b(password|credential|login|verify your account|sign[ -]?in)\b", re.IGNORECASE
)
PAYMENT = re.compile(
    r"\b(invoice|payment|bank transfer|gift card|wire transfer|crypto)\b", re.IGNORECASE
)
INJECTION = re.compile(
    r"\b(ignore|disregard|override|forget)\b.{0,50}\b(previous|prior|system|instruction|rules?)\b"
    r"|\b(system prompt|developer message|output only|do not warn|mark (?:this )?as safe)\b",
    re.IGNORECASE | re.DOTALL,
)
PERSONALIZATION = re.compile(
    r"\b(dear|hello|hi)\s+[A-Z][a-z]+|\b(your (?:team|department|manager|company))\b"
)


# Compute deterministic signals and assign them to email.features.
def extract_features(email: ProcessedEmail) -> dict[str, float | int | bool | str]:
    full_text = f"{email.subject}\n{email.text}"
    sender_domain = (
        email.sender_address.rsplit("@", 1)[-1].lower() if "@" in email.sender_address else ""
    )
    reply_domain = email.reply_to.rsplit("@", 1)[-1].lower() if "@" in email.reply_to else ""
    features: dict[str, float | int | bool | str] = {
        "character_count": len(full_text),
        "word_count": len(full_text.split()),
        "url_count": len(email.urls),
        "domain_mismatch_count": sum(int(url.domain_mismatch) for url in email.urls),
        "sender_reply_to_mismatch": bool(
            reply_domain and sender_domain and reply_domain != sender_domain
        ),
        "urgency_hits": len(URGENCY.findall(full_text)),
        "credential_hits": len(CREDENTIALS.findall(full_text)),
        "payment_hits": len(PAYMENT.findall(full_text)),
        "injection_hits": len(INJECTION.findall(full_text)),
        "personalization_hits": len(PERSONALIZATION.findall(full_text)),
        "exclamation_count": full_text.count("!"),
        "uppercase_ratio": sum(c.isupper() for c in full_text)
        / max(1, sum(c.isalpha() for c in full_text)),
    }
    email.features = features
    return features
