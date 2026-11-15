from __future__ import annotations

import re
from email import policy
from email.message import Message
from email.parser import BytesParser
from email.utils import parseaddr
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .schemas import ExtractedURL, RawEmail

URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)


# Decode a MIME text part with a payload fallback.
def _decode_part(part: Message) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except (LookupError, UnicodeDecodeError, AttributeError):
        pass  # unknown/broken charset in a real-world email; fall back to manual decoding below
    payload = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


# Extract visible HTML text and links without executing or loading content.
def html_to_safe_text(html: str) -> tuple[str, list[ExtractedURL]]:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "iframe", "form", "object", "embed", "svg"]):
        tag.decompose()

    links: list[ExtractedURL] = []
    for anchor in soup.find_all("a"):
        target = str(anchor.get("href") or "").strip()
        if not target.lower().startswith(("http://", "https://")):
            continue
        displayed = anchor.get_text(" ", strip=True)
        target_domain = (urlparse(target).hostname or "").lower()
        display_match = URL_PATTERN.search(displayed)
        display_domain = (
            (urlparse(display_match.group(0)).hostname or "").lower() if display_match else ""
        )
        links.append(
            ExtractedURL(
                displayed_text=displayed[:300],
                target_url=target[:2000],
                domain=target_domain,
                display_domain=display_domain,
                # link text showing one domain while href points elsewhere is a classic
                # phishing/masking signal; feature_extractor and baselines.py score on this.
                domain_mismatch=bool(display_domain and display_domain != target_domain),
            )
        )
    text = unescape(soup.get_text("\n", strip=True))
    return text, links


# Find and deduplicate HTTP(S) URLs in plain text.
def extract_plain_urls(text: str) -> list[ExtractedURL]:
    seen: set[str] = set()
    result: list[ExtractedURL] = []
    for match in URL_PATTERN.findall(text or ""):
        target = match.rstrip(".,);>")
        if target in seen:
            continue
        seen.add(target)
        result.append(
            ExtractedURL(
                displayed_text=target,
                target_url=target,
                domain=(urlparse(target).hostname or "").lower(),
            )
        )
    return result


# Read MIME headers and non-attachment text from an EML file.
def parse_eml(path: str | Path, instance_id: str | None = None) -> RawEmail:
    source_path = Path(path)
    message = BytesParser(policy=policy.default).parsebytes(source_path.read_bytes())
    sender_name, sender_address = parseaddr(str(message.get("From", "")))
    _, reply_to = parseaddr(str(message.get("Reply-To", "")))
    text_parts: list[str] = []
    html_parts: list[str] = []

    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        disposition = str(part.get("Content-Disposition", "")).lower()
        if "attachment" in disposition:
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            text_parts.append(_decode_part(part))
        elif content_type == "text/html":
            html_parts.append(_decode_part(part))

    return RawEmail(
        instance_id=instance_id or source_path.stem,
        raw_source=message.as_string(),
        sender_name=sender_name,
        sender_address=sender_address,
        reply_to=reply_to,
        subject=str(message.get("Subject", "")),
        text_body="\n".join(text_parts),
        html_body="\n".join(html_parts),
        date=str(message.get("Date", "")),
        metadata={"source_file": source_path.name},
    )
