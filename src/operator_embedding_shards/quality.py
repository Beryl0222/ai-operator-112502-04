"""Data-quality reporting for blocked and skipped token positions.

The training batch path must be able to hand disabled ids to the data-quality
pipeline *together with the reason*, and to distinguish genuine unknown
tokens from padding and from holes in the shard layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .updates import UpdateBatch
from .vocab import TokenStatus


@dataclass(frozen=True)
class DataQualityReport:
    """Counts and details for every non-embeddable token position."""

    micro_batch_id: str
    step: int
    unknown: tuple[int, ...]
    disabled: tuple[int, ...]
    missing: tuple[int, ...]
    padding: tuple[int, ...]
    reasons: tuple[str, ...]
    affected_versions: tuple[str, ...]

    @classmethod
    def from_updates(cls, batch: UpdateBatch) -> "DataQualityReport":
        unknown: list[int] = []
        missing: list[int] = []
        padding: list[int] = []
        reasons: list[str] = []
        versions: list[str] = []
        for skipped in batch.skipped:
            if skipped.status is TokenStatus.UNKNOWN:
                unknown.append(skipped.token_id)
            elif skipped.status is TokenStatus.MISSING:
                missing.append(skipped.token_id)
            elif skipped.status is TokenStatus.PADDING:
                padding.append(skipped.token_id)
            else:
                continue
            if skipped.version not in versions:
                versions.append(skipped.version)
            if skipped.reason:
                reasons.append(f"{skipped.status.value}:{skipped.token_id}:{skipped.reason}")
        disabled = [entry.token_id for entry in batch.blocked]
        for entry in batch.blocked:
            if entry.version not in versions:
                versions.append(entry.version)
            reasons.append(f"disabled:{entry.token_id}:{entry.reason}")
        return cls(
            micro_batch_id=batch.micro_batch_id,
            step=batch.step,
            unknown=tuple(unknown),
            disabled=tuple(disabled),
            missing=tuple(missing),
            padding=tuple(padding),
            reasons=tuple(reasons),
            affected_versions=tuple(versions),
        )

    @property
    def counts(self) -> Mapping[str, int]:
        return {
            TokenStatus.UNKNOWN.value: len(self.unknown),
            TokenStatus.DISABLED.value: len(self.disabled),
            TokenStatus.MISSING.value: len(self.missing),
            TokenStatus.PADDING.value: len(self.padding),
        }

    @property
    def has_issues(self) -> bool:
        return bool(self.unknown or self.disabled or self.missing or self.padding)

    @property
    def blocks_updates(self) -> bool:
        return bool(self.disabled)

    def as_dict(self) -> dict[str, object]:
        return {
            "micro_batch_id": self.micro_batch_id,
            "step": self.step,
            "unknown": list(self.unknown),
            "disabled": list(self.disabled),
            "missing": list(self.missing),
            "padding": list(self.padding),
            "reasons": list(self.reasons),
            "affected_versions": list(self.affected_versions),
            "counts": dict(self.counts),
        }


__all__ = ["DataQualityReport"]
