"""Independent personalization annotations; never model instructions or predictions."""

import csv
import json
import re
from pathlib import Path
from typing import Any


# Observable text markers, deliberately separate from reviewed P0/P1 labels.
SALUTATION = re.compile(
    r"(?:^|\n)\s*(?:dear|hello|hi|hallo|liebe[r]?|guten tag|gentile|car[oa]|ciao)"
    r"\s+(?:(?:mr|mrs|ms|dr|herr|frau|sig|sigra)\.?\s+)?([^\W\d_][\w'’-]*)",
    re.IGNORECASE,
)
GENERIC_RECIPIENTS = {
    "customer", "client", "user", "member", "subscriber", "team", "all", "everyone",
    "colleague", "colleagues", "sir", "madam", "friend", "partner", "valued", "dear",
    "taxpayer", "recipient", "employee", "employees", "account", "support", "hr",
    "kunde", "kundin", "kunden", "mitglieder", "mitglied", "kollegen", "kolleginnen",
    "damen", "herren", "nutzende", "nutzer", "klient", "cliente", "clienti", "utente",
    "utenti", "collega", "colleghi", "signore", "signora", "staff", "manager",
}
RECIPIENT_CONTEXT = re.compile(
    r"\b(?:your\s+(?:team|department|company|project|organization|organisation|role)"
    r"|ihr(?:e|en|em|er|es)?\s+(?:team|abteilung|unternehmen|projekt|organisation|position)"
    r"|(?:tu[oa]|vostr[oa]|su[oa])\s+(?:team|reparto|azienda|progetto|organizzazione|ruolo))\b",
    re.IGNORECASE,
)


def personalization_indicators(text: str) -> dict[str, bool]:
    """Return exploratory markers; absence does not establish an unpersonalized email."""
    named = any(match.group(1)[0].isupper() and match.group(1).casefold() not in GENERIC_RECIPIENTS
                for match in SALUTATION.finditer(text))
    return {"named_salutation": named, "recipient_context_phrase": bool(RECIPIENT_CONTEXT.search(text))}


# Validate an independently annotated level without inferring one from email content.
def personalization_level(value: Any) -> int | None:
    if value in (None, "", "unknown"):
        return None
    if type(value) is int and value in (0, 1):
        return value
    raise ValueError("personalization_level must be the integer 0, 1 or unknown")


# Read a partial CSV annotation overlay, checking IDs and levels against the full dataset.
def load_personalization_annotations(path: Path, instance_ids: set[str]) -> dict[str, dict]:
    annotations = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "personalization_level"}.issubset(reader.fieldnames or []):
            raise ValueError("Annotation CSV requires id and personalization_level columns")
        for row in reader:
            instance_id = row["id"].strip()
            if instance_id not in instance_ids or instance_id in annotations:
                raise ValueError(f"Unknown or duplicate annotation ID: {instance_id}")
            value = row["personalization_level"].strip()
            if value not in {"", "unknown", "0", "1"}:
                raise ValueError(f"Invalid personalization level for {instance_id}: {value}")
            annotations[instance_id] = {
                "personalization_level": int(value) if value in {"0", "1"} else None,
                "annotation_note": row.get("annotation_note", "") or "",
            }
    return annotations


# Export blank annotation rows from a task or prototype JSONL dataset without overwriting files.
def write_personalization_template(dataset: Path, output: Path) -> int:
    with dataset.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [str(row.get("id") or row["instance_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "personalization_level", "annotation_note"])
        writer.writerows((instance_id, "", "") for instance_id in ids)
    return len(ids)
