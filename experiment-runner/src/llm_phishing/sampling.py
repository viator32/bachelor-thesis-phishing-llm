from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Any


def _label_key(row: dict[str, Any]) -> str:
    return json.dumps(row["target"], sort_keys=True)


def label_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"labels": json.loads(key), "count": count}
        for key, count in sorted(Counter(_label_key(row) for row in rows).items())
    ]


def _quotas(groups: dict[str, list], maximum: int, cover_all: bool) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    quotas = {key: maximum * len(items) // total for key, items in groups.items()}
    remainder = sorted(groups, key=lambda key: (-(maximum * len(groups[key]) % total), key))
    for key in remainder[:maximum - sum(quotas.values())]:
        quotas[key] += 1
    if cover_all:
        quotas = {key: max(1, count) for key, count in quotas.items()}
        while sum(quotas.values()) > maximum:
            donor = max(
                (key for key in quotas if quotas[key] > 1),
                key=lambda key: (quotas[key] - maximum * len(groups[key]) / total, key),
            )
            quotas[donor] -= 1
    return quotas


def stratified_sample(
    rows: list[dict[str, Any]], maximum: int | None, seed: int,
) -> list[dict[str, Any]]:
    """Preserve label proportions and cover every available label combination."""
    if maximum is not None and maximum < 1:
        raise ValueError("The maximum email count must be positive")
    if not rows:
        return []
    wanted = min(maximum, len(rows)) if maximum is not None else len(rows)
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels[_label_key(row)].append(row)
    if wanted < len(labels):
        raise ValueError(
            f"Maximum emails must be at least {len(labels)} to include every available label combination"
        )
    selected = []
    for label, count in sorted(_quotas(labels, wanted, cover_all=True).items()):
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in labels[label]:
            strata[row["stratum"]].append(row)
        for stratum, size in sorted(_quotas(strata, count, cover_all=count >= len(strata)).items()):
            items = sorted(strata[stratum], key=lambda row: row["id"])
            random.Random(f"{seed}:labels:{label}:stratum:{stratum}").shuffle(items)
            selected.extend(items[:size])
    random.Random(f"{seed}:labels:order").shuffle(selected)
    return selected
