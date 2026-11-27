"""Optimiser hand-off records.

A forward pass knows which *shard* and which *vocabulary version* produced
each vector.  :class:`UpdateBatch` carries that provenance to the optimiser so
it never has to infer a shard from a flat token id, and so repeated ids can be
merged with an explicitly declared rule instead of being overwritten once per
occurrence.

Every record also carries the layout and row fingerprints captured during the
forward pass.  A store applying the record later can therefore decide whether
the update is still attributable to the same row (:meth:`ParameterStore.apply`
in :mod:`operator_embedding_shards.store`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .errors import AggregationError
from .hashing import stable_digest
from .vocab import TokenStatus


class AggregationRule(str, Enum):
    """How gradients of repeated ids within one micro-batch are merged."""

    SUM = "sum"
    MEAN = "mean"
    FIRST = "first"
    LAST = "last"
    ERROR = "error"

    @classmethod
    def parse(cls, value: "AggregationRule | str") -> "AggregationRule":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            allowed = ", ".join(rule.value for rule in cls)
            raise AggregationError(
                f"unsupported aggregation rule {value!r}; expected one of: {allowed}"
            ) from None


@dataclass(frozen=True)
class UpdateTarget:
    """Exact embedding row an update belongs to, including its shard."""

    version: str
    token_id: int
    shard_id: str
    shard_start: int
    local_index: int

    @property
    def global_index(self) -> int:
        return self.shard_start + self.local_index

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "token_id": self.token_id,
            "shard_id": self.shard_id,
            "shard_start": self.shard_start,
            "local_index": self.local_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UpdateTarget":
        return cls(
            version=str(payload["version"]),
            token_id=int(payload["token_id"]),
            shard_id=str(payload["shard_id"]),
            shard_start=int(payload["shard_start"]),
            local_index=int(payload["local_index"]),
        )


@dataclass(frozen=True)
class UpdateRecord:
    """One aggregated, attributable gradient update ready for the optimiser."""

    target: UpdateTarget
    gradient: tuple[float, ...]
    occurrences: tuple[int, ...]
    aggregation: AggregationRule
    layout_fingerprint: str
    row_fingerprint: str
    micro_batch_id: str
    step: int
    update_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.gradient:
            raise AggregationError("update gradient must not be empty")
        if not self.occurrences:
            raise AggregationError("update must record at least one occurrence")
        object.__setattr__(
            self,
            "update_id",
            stable_digest(
                "update",
                self.micro_batch_id,
                int(self.step),
                self.target.version,
                self.target.token_id,
                self.target.shard_id,
                self.target.shard_start,
                self.target.local_index,
                self.layout_fingerprint,
                self.row_fingerprint,
                self.aggregation.value,
                [float(value) for value in self.gradient],
                list(self.occurrences),
            ),
        )

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)

    @property
    def dim(self) -> int:
        return len(self.gradient)

    def as_dict(self) -> dict[str, object]:
        return {
            "update_id": self.update_id,
            "target": self.target.as_dict(),
            "gradient": list(self.gradient),
            "occurrences": list(self.occurrences),
            "occurrence_count": self.occurrence_count,
            "aggregation": self.aggregation.value,
            "layout_fingerprint": self.layout_fingerprint,
            "row_fingerprint": self.row_fingerprint,
            "micro_batch_id": self.micro_batch_id,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UpdateRecord":
        record = cls(
            target=UpdateTarget.from_dict(payload["target"]),  # type: ignore[arg-type]
            gradient=tuple(float(value) for value in payload["gradient"]),  # type: ignore[union-attr]
            occurrences=tuple(int(value) for value in payload["occurrences"]),  # type: ignore[union-attr]
            aggregation=AggregationRule.parse(str(payload["aggregation"])),
            layout_fingerprint=str(payload["layout_fingerprint"]),
            row_fingerprint=str(payload["row_fingerprint"]),
            micro_batch_id=str(payload["micro_batch_id"]),
            step=int(payload["step"]),
        )
        expected = payload.get("update_id")
        if expected is not None and expected != record.update_id:
            raise AggregationError("serialised update_id does not match its content")
        return record


@dataclass(frozen=True)
class BlockedUpdate:
    """A disabled id that must not update its row, with a data-quality reason."""

    token_id: int
    version: str
    occurrences: tuple[int, ...]
    shard_id: str | None
    shard_start: int | None
    local_index: int | None
    reason: str
    layout_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "version": self.version,
            "occurrences": list(self.occurrences),
            "shard_id": self.shard_id,
            "shard_start": self.shard_start,
            "local_index": self.local_index,
            "reason": self.reason,
            "layout_fingerprint": self.layout_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BlockedUpdate":
        return cls(
            token_id=int(payload["token_id"]),
            version=str(payload["version"]),
            occurrences=tuple(int(value) for value in payload["occurrences"]),  # type: ignore[union-attr]
            shard_id=payload.get("shard_id"),  # type: ignore[arg-type]
            shard_start=payload.get("shard_start"),  # type: ignore[arg-type]
            local_index=payload.get("local_index"),  # type: ignore[arg-type]
            reason=str(payload["reason"]),
            layout_fingerprint=str(payload["layout_fingerprint"]),
        )

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)


@dataclass(frozen=True)
class SkippedToken:
    """A token position that produced no row update (unknown/missing/padding)."""

    token_id: int
    version: str
    occurrence: int
    status: TokenStatus
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "version": self.version,
            "occurrence": self.occurrence,
            "status": self.status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SkippedToken":
        return cls(
            token_id=int(payload["token_id"]),
            version=str(payload["version"]),
            occurrence=int(payload["occurrence"]),
            status=TokenStatus(str(payload["status"])),
            reason=payload.get("reason"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class UpdateBatch:
    """The complete hand-off from one backward pass to the optimiser."""

    micro_batch_id: str
    step: int
    aggregation: AggregationRule
    records: tuple[UpdateRecord, ...]
    blocked: tuple[BlockedUpdate, ...]
    skipped: tuple[SkippedToken, ...]
    versions: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def update_ids(self) -> tuple[str, ...]:
        return tuple(record.update_id for record in self.records)

    def records_for(self, version: str) -> tuple[UpdateRecord, ...]:
        return tuple(record for record in self.records if record.target.version == version)

    def quality_report(self):
        from .quality import DataQualityReport

        return DataQualityReport.from_updates(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "micro_batch_id": self.micro_batch_id,
            "step": self.step,
            "aggregation": self.aggregation.value,
            "versions": list(self.versions),
            "records": [record.as_dict() for record in self.records],
            "blocked": [blocked.as_dict() for blocked in self.blocked],
            "skipped": [skipped.as_dict() for skipped in self.skipped],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UpdateBatch":
        return cls(
            micro_batch_id=str(payload["micro_batch_id"]),
            step=int(payload["step"]),
            aggregation=AggregationRule.parse(str(payload["aggregation"])),
            records=tuple(
                UpdateRecord.from_dict(item)
                for item in payload["records"]  # type: ignore[union-attr]
            ),
            blocked=tuple(
                BlockedUpdate.from_dict(item)
                for item in payload.get("blocked", ())  # type: ignore[union-attr]
            ),
            skipped=tuple(
                SkippedToken.from_dict(item)
                for item in payload.get("skipped", ())  # type: ignore[union-attr]
            ),
            versions=tuple(str(item) for item in payload.get("versions", ())),  # type: ignore[union-attr]
        )


def aggregate_gradients(
    gradients: Sequence[Sequence[float]],
    rule: AggregationRule,
    *,
    dim: int,
) -> tuple[float, ...]:
    """Merge gradients of one id according to ``rule``.

    ``gradients`` keeps the micro-batch order so ``first``/``last`` are
    deterministic and reproducible.
    """

    if not gradients:
        raise AggregationError("cannot aggregate an empty gradient list")
    for index, gradient in enumerate(gradients):
        if len(gradient) != dim:
            raise AggregationError(
                f"gradient {index} has dim {len(gradient)}, expected {dim}"
            )
    if rule is AggregationRule.SUM:
        merged = [0.0] * dim
        for gradient in gradients:
            for axis, value in enumerate(gradient):
                merged[axis] += value
        return tuple(merged)
    if rule is AggregationRule.MEAN:
        merged = [0.0] * dim
        for gradient in gradients:
            for axis, value in enumerate(gradient):
                merged[axis] += value
        count = len(gradients)
        return tuple(value / count for value in merged)
    if rule is AggregationRule.FIRST:
        return tuple(gradients[0])
    if rule is AggregationRule.LAST:
        return tuple(gradients[-1])
    if rule is AggregationRule.ERROR:
        # The caller (backward) already rejected repeated ids; a single
        # occurrence is the only case that reaches here.
        if len(gradients) == 1:
            return tuple(gradients[0])
        raise AggregationError("rule 'error' forbids merging repeated ids")
    raise AggregationError(
        f"rule {rule.value!r} cannot be used to merge repeated ids"
    )


__all__ = [
    "AggregationRule",
    "UpdateTarget",
    "UpdateRecord",
    "BlockedUpdate",
    "SkippedToken",
    "UpdateBatch",
    "aggregate_gradients",
]
