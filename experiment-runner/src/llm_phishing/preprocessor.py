from __future__ import annotations

import re
from typing import Any

from .email_parser import extract_plain_urls, html_to_safe_text
from .schemas import ProcessedEmail, RawEmail

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# Approximate tokens deterministically; not the model tokenizer.
def lexical_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


# Retain both ends when text exceeds the lexical token budget.
def truncate_middle(text: str, max_tokens: int | None) -> tuple[str, dict[str, Any]]:
    tokens = lexical_tokens(text)
    if max_tokens is None or len(tokens) <= max_tokens:
        return text, {
            "operation": "truncate_middle",
            "applied": False,
            "tokenizer": "offline_lexical_v1",
            "tokens_before": len(tokens),
            "tokens_after": len(tokens),
        }
    # Keep both ends rather than just the head: sender/greeting cues live at the start and the
    # call-to-action or links (the strongest phishing signal) usually sit at the end.
    head_count = max_tokens // 2
    tail_count = max_tokens - head_count
    kept = tokens[:head_count] + ["[...TRUNCATED...]"] + tokens[-tail_count:]
    truncated = " ".join(kept)
    return truncated, {
        "operation": "truncate_middle",
        "applied": True,
        "tokenizer": "offline_lexical_v1",
        "tokens_before": len(tokens),
        "tokens_after": len(kept),
        "head_tokens": head_count,
        "tail_tokens": tail_count,
    }


# Normalize email text, extract URLs and apply the token budget.
def preprocess_email(raw: RawEmail, max_input_tokens: int | None = 4096) -> ProcessedEmail:
    html_text, html_urls = html_to_safe_text(raw.html_body)
    combined = raw.text_body.strip()
    # Some datasets duplicate the same content into text_body and html_body; only append the
    # HTML-derived text when it adds something new, to avoid doubling signal counts downstream.
    if html_text and html_text not in combined:
        combined = f"{combined}\n{html_text}".strip()
    normalized = combined.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized).strip()
    truncated, truncation = truncate_middle(normalized, max_input_tokens)

    urls_by_target = {url.target_url: url for url in html_urls}
    for url in extract_plain_urls(normalized):
        urls_by_target.setdefault(url.target_url, url)

    return ProcessedEmail(
        instance_id=raw.instance_id,
        sender_name=raw.sender_name.strip(),
        sender_address=raw.sender_address.strip(),
        reply_to=raw.reply_to.strip(),
        subject=raw.subject.strip(),
        text=truncated,
        date=raw.date.strip(),
        language=raw.language,
        urls=list(urls_by_target.values()),
        transformations=[
            {"operation": "decode_mime", "applied": bool(raw.raw_source)},
            {"operation": "html_visible_text", "applied": bool(raw.html_body)},
            {"operation": "normalize_newlines", "applied": True},
            truncation,
        ],
        source_dataset=raw.source_dataset,
        split=raw.split,
        base_id=raw.base_id,
        ground_truth=raw.ground_truth,
    )
