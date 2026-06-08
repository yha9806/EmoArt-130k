from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ControlledBCandidate:
    sample_id: str
    candidate_image: str
    source: str
    priority: int
    score: float = 0.0
    tier: str = ""
    caption: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sample_id": self.sample_id,
            "decision": "accept",
            "candidate_image": self.candidate_image,
            "source": self.source,
            "priority": self.priority,
            "score": self.score,
        }
        if self.tier:
            row["tier"] = self.tier
        if self.caption:
            row["caption"] = self.caption
        if self.metadata:
            row["metadata"] = self.metadata
        return row


def select_controlled_ladder(
    candidates: Iterable[ControlledBCandidate], *, target_count: int
) -> list[ControlledBCandidate]:
    best_by_sample: dict[str, ControlledBCandidate] = {}
    for candidate in candidates:
        if not candidate.sample_id or not candidate.candidate_image:
            continue
        existing = best_by_sample.get(candidate.sample_id)
        if existing is None or _candidate_sort_key(candidate) > _candidate_sort_key(existing):
            best_by_sample[candidate.sample_id] = candidate

    ordered = sorted(best_by_sample.values(), key=_candidate_sort_key, reverse=True)
    if len(ordered) < target_count:
        raise ValueError(f"not enough unique candidates for target {target_count}: {len(ordered)} available")
    return ordered[:target_count]


def filter_current_identical_candidates(
    candidates: Iterable[ControlledBCandidate], current_image_dir: str | Path
) -> tuple[list[ControlledBCandidate], list[ControlledBCandidate]]:
    current_image_dir = Path(current_image_dir)
    filtered: list[ControlledBCandidate] = []
    blocked: list[ControlledBCandidate] = []
    current_hashes: dict[str, str] = {}
    for candidate in candidates:
        current_path = current_image_dir / f"{candidate.sample_id}.jpg"
        candidate_path = Path(candidate.candidate_image)
        if current_path.exists() and candidate_path.exists():
            current_hash = current_hashes.setdefault(str(current_path), _sha256(current_path))
            if _sha256(candidate_path) == current_hash:
                blocked.append(candidate)
                continue
        filtered.append(candidate)
    return filtered, blocked


def _candidate_sort_key(candidate: ControlledBCandidate) -> tuple[int, float, str]:
    return (candidate.priority, candidate.score, candidate.sample_id)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
