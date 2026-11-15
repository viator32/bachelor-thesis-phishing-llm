"""Provisional rule-assisted personalization assessment of every saved test input.

No detector predictions are read. These suggestions are NOT human reference labels.
Run from the repository root: python3 thesis/assets/analysis/personalisierung_vorschlaege.py
"""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT = Path(__file__).resolve().parent
DATA = ROOT / "experiment-runner/data"
LOCAL = DATA / "evaluation/personalization"
METHOD = "assistant_rules_v1"
DATASETS = {
    "phishfuzzer": "PhishFuzzer", "ephishllm": "E-PhishLLM", "crossmodel": "Cross-Model",
    "contextual": "Contextual Paired Email Injection", "llmail": "LLMail-Inject",
}
GENERIC = set("customer client user member subscriber team all everyone colleague colleagues sir madam "
              "friend partner valued dear taxpayer recipient employee employees account support hr staff "
              "manager developer participant student researcher business administrator admin colleague "
              "kunde kundin kunden mitglieder mitglied kollegen kolleginnen damen herren nutzer "
              "cliente clienti utente utenti collega colleghi signore signora tutti professionista "
              "sir/madam recipient_name firstname name nome cognome vorname nachname username".split())
NAME = r"([^\W\d_][\w'’.-]*(?:\s+[^\W\d_][\w'’.-]*){0,2})"
GREETING = re.compile(
    r"(?:^|\n)\s*(?:dear|hello|hi|hallo|liebe[r]?|guten tag|gentile|car[oa]|ciao|"
    r"sehr geehrte[r]?)\s+(?:(?:mr|mrs|ms|dr|herr|frau|sig|sigra|signor|signora)\.?\s+)?"
    + NAME + r"(?=\s*[,!:\n]|\s*$)", re.I)
EMAIL_GREETING = re.compile(r"(?:^|\n)\s*(?:dear|hello|hi|hallo)\s+([\w.+-]+@[\w.-]+)", re.I)
RECIPIENT_HEADER = re.compile(r"(?:^|\n)(?:To|An|A):\s*([^\n]{3,120})", re.I)
PLACEHOLDER = re.compile(r"\[(?:your |recipient.?s? )?(?:name|first.?name|last.?name|nome|cognome|vorname|nachname)\]|\{\{?\s*(?:recipient|first_name|name)\s*\}?\}", re.I)
SIGNATURE = re.compile(r"\n\s*(?:best regards|kind regards|sincerely|regards|mit freundlichen grüßen|"
                       r"cordiali saluti|distinti saluti|un caro saluto)\s*[,!]?\s*\n", re.I)
RECIPIENT_LINK = re.compile(r"\b(?:you|your|yours|our|we|unsere?[nmsr]?|wir|ihre?[nmsr]?|sie|deine?[nmsr]?|"
                            r"tu[oaie]|vostr[oaie]|suo|sua|lei|nostr[oaie])\b", re.I)
DETAILS = {
    "role": re.compile(r"\b(?:in your role as|given your role as|as (?:our|the|a)|als|in qualità di)\s+"
                       r"(?:(?:senior|lead|chief|head|responsabile|leitende[rn]?|technical|project)\s+){0,2}"
                       r"(?:engineer|manager|director|analyst|developer|designer|coordinator|researcher|"
                       r"professor|consultant|officer|leiter\w*|entwickler\w*|ingenieur\w*|"
                       r"direttore|responsabile|progettista|ricercatore)\b", re.I),
    "work_artifact": re.compile(r"\b(?:your|our|ihre?[nmsr]?|unsere?[nmsr]?|deine?[nmsr]?|"
                                r"(?:il |la |i |le )?(?:tu[oaie]|vostr[oaie]|nostr[oaie]))\s+"
                                r"(?:(?:recent|preliminary|ongoing|updated|latest|current|final|"
                                r"letzte[nmrs]?|aktuelle[nmrs]?|neue[nmrs]?|recente|nuovo|nuova)\s+){0,2}"
                                r"(?:report|project|proposal|presentation|analysis|research|submission|"
                                r"application|work|bericht\w*|projekt\w*|präsentation|analyse|forschung|"
                                r"progetto|relazione|rapporto|presentazione|analisi|ricerca)\s+"
                                r"(?:on|for|of|regarding|about|to|in|zu[mr]?|über|für|sul|sulla|su|di|per)\s+[^\n.!?]{8,100}", re.I),
    "follow_up": re.compile(r"\b(?:following up on|follow-up (?:on|to)|regarding your|in response to your|"
                            r"as (?:we|you) discussed|our (?:recent|last|previous) (?:meeting|discussion|conversation)|"
                            r"wie (?:besprochen|vereinbart)|bezug auf (?:ihr|unser)|"
                            r"in seguito (?:a|alla)|come (?:discusso|concordato)|in merito (?:al|alla))\s+[^\n.!?]{8,110}", re.I),
    "reference_id": re.compile(r"\b(?:ticket|case|ref(?:erence)?|order|invoice|account number|"
                               r"kundennummer|auftragsnummer|referenz|vorgang|fattura|ordine|riferimento)"
                               r"\b\s*(?:no\.?\b|number\b|nr\.?\b|numero\b|:|#)?\s*"
                               r"(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{2,25}\b", re.I),
    "specific_progress": re.compile(r"\b(?:your (?:department|team|submission|report|proposal|application) "
                                    r"(?:has|have|completed|was|is|includes|contains)|"
                                    r"(?:section|chapter|phase|milestone) (?:one|two|three|four|\d+) (?:of|in|needs|requires)|"
                                    r"(?:ihr|unser)\w* (?:team|abteilung|bericht) (?:hat|haben|enthält)|"
                                    r"(?:il tuo|la tua|il vostro|la vostra) (?:team|relazione|rapporto|proposta) "
                                    r"(?:ha|è|contiene))[^\n.!?]{5,100}", re.I),
    "recipient_organization": re.compile(r"\b(?:your (?:role|position|work|team|department) (?:at|with)|"
                                         r"(?:ihre?|deine?) (?:arbeit|position|rolle) bei|"
                                         r"(?:tuo|vostro) (?:ruolo|lavoro) (?:presso|a))\s+[^\n,.!?]{3,70}", re.I),
    "individual_feedback": re.compile(r"\b(?:section|chapter|abschnitt|kapitel|sezione|capitolo)\s+"
                                      r"(?:one|two|three|four|five|eins|zwei|drei|uno|due|tre|\d+)"
                                      r"[^\n.!?]{0,80}", re.I),
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def assessment_text(text):
    """Exclude trusted task, transport fields, appended URL lists and sender signature."""
    match = re.search(r"<UNTRUSTED_EMAIL>(.*?)</UNTRUSTED_EMAIL>", text, re.S)
    scope = "untrusted_email_block" if match else "message_body"
    body = match.group(1) if match else re.split(r"(?:^|\n)Body:\s*|\bBody:\s*", text, maxsplit=1)[-1]
    body = re.split(r"\nURLs:\s*", body, maxsplit=1)[0]
    body = SIGNATURE.split(body, maxsplit=1)[0]
    return body.strip(), scope


def assess(text, language):
    body, scope = assessment_text(text)
    evidence = []

    def add(kind, match):
        evidence.append(dict(kind=kind, quote=match.group(0), start=match.start(), end=match.end()))

    for match in GREETING.finditer(body):
        recipient = match.group(1).strip()
        words = re.findall(r"[\w'’.-]+", recipient)
        if words and not any(word.casefold() in GENERIC for word in words) and words[0][0].isupper():
            add("named_recipient", match)
            break
    for match in EMAIL_GREETING.finditer(body):
        add("recipient_email", match)
        break
    for match in RECIPIENT_HEADER.finditer(body[:400]):
        target = match.group(1)
        if "@" in target and not re.search(r"\b(?:undisclosed|list|all|team|customers)\b", target, re.I):
            add("recipient_header", match)
            break
    addressed = bool(evidence)
    for kind, pattern in DETAILS.items():
        for match in pattern.finditer(body):
            context = body[max(0, match.start()-140):match.end()+140]
            if not (addressed or RECIPIENT_LINK.search(context)):
                continue
            # The role of the software assistant is not the email recipient's occupation.
            if kind == "role" and re.search(r"\b(?:assistant|language model|AI|system prompt)\b", match.group(), re.I):
                continue
            if kind == "reference_id" and re.search(r"XXX|0000|12345", match.group(), re.I):
                continue
            add(kind, match)
            break
    details = [item for item in evidence if item["kind"] in DETAILS]
    # Two distinct, non-overlapping contextual clues near one another; greeting does not count.
    linked_pair = next(((a, b) for i, a in enumerate(details) for b in details[i+1:]
                        if a["kind"] != b["kind"] and abs(a["start"]-b["start"]) <= 700
                        and (a["end"] <= b["start"] or b["end"] <= a["start"])), None)
    reasons = []
    level = 2 if linked_pair else 1 if evidence else 0
    if level == 2:
        reasons.append("two_nonoverlapping_linked_context_clues")
    elif level == 1:
        reasons.append("recipient_or_single_context_clue")
    else:
        reasons.append("no_recipient_clue_detected_by_rules_not_proof_of_absence")
    uncertain = []
    if language not in {"en", "de", "it"}:
        uncertain.append("language_outside_rule_coverage")
    if len(re.findall(r"[A-Za-z]\d[A-Za-z]", body)) > 15:
        uncertain.append("obfuscated_text")
    if sum("\u0400" <= c <= "\u052f" or "\u0e00" <= c <= "\u0e7f" or "\u3040" <= c <= "\u9fff" for c in body) > 100:
        uncertain.append("substantial_uncovered_script")
    if PLACEHOLDER.search(body):
        uncertain.append("unresolved_recipient_placeholder")
    if not body:
        uncertain.append("empty_assessable_message")
    return dict(suggested_level=None if uncertain else level, annotation_method=METHOD,
                annotation_status="provisional_not_human_verified", human_verified=False,
                annotation_note="; ".join(reasons + uncertain), evidence=evidence,
                assessment_scope=scope, assessed_text_sha256=sha_bytes(body.encode()),
                input_sha256=sha_bytes(text.encode()), language=language,
                limitation="Finite lexical rules; no semantic completeness, causal effect or independent ground truth.")


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    snapshots = LOCAL / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    detailed, public, provenance = [], [], []
    for name, dataset in DATASETS.items():
        path = DATA / f"processed/comparison_{name}_test.jsonl"
        original = path.read_bytes()
        rows = [json.loads(line) for line in original.decode().splitlines() if line.strip()]
        snapshot = snapshots / path.name
        if not snapshot.exists():
            snapshot.write_bytes(original)
        for row in rows:
            identifier = row["instance_id"]
            proposal = assess(row["text_body"], row.get("language") or "en")
            row["personalization_assessment"] = proposal
            base = dict(id=identifier, dataset=dataset, **proposal)
            detailed.append(dict(base, input_text=row["text_body"]))
            public.append({k: v for k, v in base.items() if k not in {"evidence", "limitation"}} |
                          {"evidence_kinds": ";".join(item["kind"] for item in proposal["evidence"]),
                           "evidence_count": len(proposal["evidence"])})
        # Add suggestions only; original labels, texts, IDs and order are untouched.
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        assert all({k: v for k, v in new.items() if k != "personalization_assessment"} ==
                   {k: v for k, v in old.items() if k != "personalization_assessment"}
                   for new, old in zip(rows, (json.loads(line) for line in original.decode().splitlines())))
        provenance.append(dict(dataset=dataset, path=str(path.relative_to(ROOT)), rows=len(rows),
                               original_snapshot_sha256=sha_bytes(snapshot.read_bytes()),
                               enriched_sha256=sha_bytes(path.read_bytes())))
    assert len(public) == 4296 and len({r["id"] for r in public}) == 4296
    with (LOCAL / "assistant_proposals.jsonl").open("w") as stream:
        for row in detailed:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "personalisierung_vorschlaege.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)
    review_path = LOCAL / "assistant_review.csv"
    existing_reviews = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            existing_reviews = {row["id"]: row for row in csv.DictReader(stream)}
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["id", "dataset", "suggested_level", "suggestion_note", "evidence", "personalization_level", "annotation_note", "reviewer", "input_text"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"id": r["id"], "dataset": r["dataset"], "suggested_level": r["suggested_level"],
                          "suggestion_note": r["annotation_note"], "evidence": json.dumps(r["evidence"], ensure_ascii=False), "input_text": r["input_text"],
                          "annotation_note": existing_reviews.get(r["id"], {}).get(
                              "annotation_note" if "suggestion_note" in existing_reviews.get(r["id"], {}) else "human_note", ""),
                          **{key: existing_reviews.get(r["id"], {}).get(key, "")
                             for key in ["personalization_level", "reviewer"]}} for r in detailed)
    manifest = dict(date="2026-10-27", annotation_method=METHOD, human_verified=False,
                    script_sha256=sha_bytes(Path(__file__).read_bytes()), selected_inputs=len(public),
                    detector_predictions_read=False, dataset_enrichment="personalization_assessment only; existing ground truth preserved",
                    sources=provenance)
    (OUT / "personalisierung_vorschlaege_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    catalog_path = DATA / "evaluation/baseline_comparison.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["note"] = ("Test only; no new detector calls or training. Original input text and reference labels "
                       "preserved; additional provisional rule-assisted personalization metadata, not human reference labels.")
    for entry in catalog["datasets"]:
        p = ROOT / "experiment-runner" / entry["path"]
        entry.setdefault("unannotated_sha256", entry["sha256"])
        entry["sha256"] = sha_bytes(p.read_bytes())
        entry["personalization_assessment"] = METHOD + "; provisional metadata, not reference labels"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+"\n")
    for dataset in DATASETS.values():
        print(dataset, dict(Counter("offen" if r["suggested_level"] is None else f"P{r['suggested_level']}*"
                                    for r in public if r["dataset"] == dataset)))


if __name__ == "__main__":
    main()
