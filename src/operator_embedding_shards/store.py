"""Parameter store that applies update records and audits attribution.

The store never takes a flat token id alone.  :meth:`ParameterStore.apply`
checks the record's layout fingerprint and row fingerprint before touching a
row, which is what makes two guarantees possible:

* an update computed for a vector produced under vocabulary version *A* can
  never overwrite that row after the batch switched to version *B*;
* resuming from a checkpoint can separate updates that are still attributable
  (:attr:`ApplyOutcome.APPLIED` / ``DUPLICATE``) from updates whose row moved
  or whose layout disappeared (:attr:`ApplyOutcome.STALE` /
  ``UNATTRIBUTED``).

The default step function is ``parameter += learning_rate * gradient``; the
store intentionally does not embed optimiser state, so the same records can be
handed to an external optimiser instead of being applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Mapping

from .errors import CheckpointError, UnknownVocabularyVersion
from .hashing import row_fingerprint
from .updates import UpdateBatch, UpdateRecord
from .vocab import VocabularyRegistry, VocabularyVersion


class ApplyOutcome(str, Enum):
    """Why an update record was or was not applied."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE = "stale"
    UNATTRIBUTED = "unattributed"
    DISABLED = "disabled"
    STEP_MISMATCH = "step_mismatch"
    SHARD_MISMATCH = "shard_mismatch"

    @property
    def is_replayable(self) -> bool:
        """True when the record was applied or is a benign replay."""

        return self in (ApplyOutcome.APPLIED, ApplyOutcome.DUPLICATE)


@dataclass(frozen=True)
class ApplyResult:
    """Per-record decision returned by :meth:`ParameterStore.apply`."""

    update_id: str
    token_id: int
    version: str
    shard_id: str
    outcome: ApplyOutcome
    reason: str | None = None

    @property
    def is_replayable(self) -> bool:
        return self.outcome.is_replayable

    def as_dict(self) -> dict[str, object]:
        return {
            "update_id": self.update_id,
            "token_id": self.token_id,
            "version": self.version,
            "shard_id": self.shard_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }


@dataclass
class ApplyReport:
    """Aggregate result of applying one :class:`UpdateBatch`."""

    results: tuple[ApplyResult, ...]
    step: int
    learning_rate: float

    def __iter__(self) -> Iterator[ApplyResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def applied(self) -> tuple[ApplyResult, ...]:
        return tuple(item for item in self.results if item.outcome is ApplyOutcome.APPLIED)

    @property
    def duplicates(self) -> tuple[ApplyResult, ...]:
        return tuple(item for item in self.results if item.outcome is ApplyOutcome.DUPLICATE)

    @property
    def rejected(self) -> tuple[ApplyResult, ...]:
        return tuple(item for item in self.results if not item.is_replayable)

    @property
    def replayable_update_ids(self) -> tuple[str, ...]:
        return tuple(item.update_id for item in self.results if item.is_replayable)

    @property
    def lost_update_ids(self) -> tuple[str, ...]:
        return tuple(item.update_id for item in self.results if not item.is_replayable)

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "learning_rate": self.learning_rate,
            "results": [item.as_dict() for item in self.results],
            "replayable": list(self.replayable_update_ids),
            "lost": list(self.lost_update_ids),
        }


@dataclass(frozen=True)
class ReplayabilityReport:
    """Resume-time split between replayable and lost updates.

    ``replayable`` updates can still be handed to the optimiser: either they
    were never applied and their row is unchanged, or they are already in the
    applied ledger.  ``lost`` updates can no longer be attributed to the row
    they were produced from (layout switched, row advanced, row removed); they
    must be dropped rather than replayed onto a different vector.
    """

    step: int
    replayable: tuple[str, ...]
    lost: tuple[str, ...]
    details: tuple[ApplyResult, ...]
    records: Mapping[str, UpdateRecord] = field(repr=False, default_factory=dict)

    @classmethod
    def from_apply(
        cls,
        report: ApplyReport,
        records: Mapping[str, UpdateRecord],
    ) -> "ReplayabilityReport":
        replayable: list[str] = []
        lost: list[str] = []
        for result in report:
            (replayable if result.is_replayable else lost).append(result.update_id)
        return cls(
            step=report.step,
            replayable=tuple(replayable),
            lost=tuple(lost),
            details=report.results,
            records=records,
        )

    @property
    def lost_details(self) -> tuple[ApplyResult, ...]:
        return tuple(item for item in self.details if not item.is_replayable)

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "replayable": list(self.replayable),
            "lost": list(self.lost),
            "details": [item.as_dict() for item in self.details],
        }


@dataclass
class _VersionState:
    version: VocabularyVersion
    layout_fingerprint: str
    rows: dict[int, list[float]] = field(default_factory=dict)

    @classmethod
    def from_vocabulary(cls, version: VocabularyVersion) -> "_VersionState":
        return cls(
            version=version,
            layout_fingerprint=version.layout_fingerprint,
            rows={token_id: list(values) for token_id, _, _, values in version.iter_rows()},
        )

    def vocabulary(self) -> VocabularyVersion:
        return self.version.with_values(
            {token_id: tuple(values) for token_id, values in self.rows.items()}
        )


class ParameterStore:
    """Holds mutable row values per vocabulary version and applies updates.

    Mutations are published back into the shared :class:`VocabularyRegistry`
    after each successful apply, so a forward pass that runs after the
    optimiser step sees exactly the rows the store just wrote.  This keeps the
    operator and the store consistent without duplicating tables: the registry
    is the single source of truth for row values.
    """

    def __init__(
        self,
        registry: VocabularyRegistry,
        *,
        learning_rate: float = 1.0,
        publish: bool = True,
    ) -> None:
        if not isinstance(registry, VocabularyRegistry):
            raise TypeError("registry must be a VocabularyRegistry")
        self.registry = registry
        self.learning_rate = float(learning_rate)
        self.publish = bool(publish)
        self._states: dict[str, _VersionState] = {
            version.version: _VersionState.from_vocabulary(version) for version in registry
        }
        self._applied: set[str] = set()

    # -- inspection ---------------------------------------------------
    def __contains__(self, version: object) -> bool:
        return version in self._states

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(self._states)

    def vocabulary(self, version: str) -> VocabularyVersion:
        try:
            state = self._states[version]
        except KeyError:
            raise UnknownVocabularyVersion(f"version {version!r} is not tracked") from None
        return state.vocabulary()

    def values(self, version: str, token_id: int) -> tuple[float, ...]:
        state = self._states.get(version)
        if state is None:
            raise UnknownVocabularyVersion(f"version {version!r} is not tracked")
        try:
            return tuple(state.rows[token_id])
        except KeyError:
            raise KeyError(f"token id {token_id} has no row in version {version!r}") from None

    def applied_update_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._applied))

    # -- mutation -----------------------------------------------------
    def _classify(self, record: UpdateRecord) -> tuple[ApplyOutcome, str | None, _VersionState | None]:
        if record.update_id in self._applied:
            return ApplyOutcome.DUPLICATE, "update_id was already applied", None
        state = self._states.get(record.target.version)
        if state is None:
            return (
                ApplyOutcome.UNATTRIBUTED,
                f"vocabulary version {record.target.version!r} is no longer loaded",
                None,
            )
        if record.target.token_id in state.version.disabled:
            return (
                ApplyOutcome.DISABLED,
                "token id is disabled in this vocabulary version; updates are blocked",
                None,
            )
        if state.layout_fingerprint != record.layout_fingerprint:
            return (
                ApplyOutcome.STALE,
                "layout fingerprint changed since the forward pass",
                None,
            )
        shard = None
        for candidate in state.version.shards:
            if candidate.shard_id == record.target.shard_id:
                shard = candidate
                break
        if shard is None:
            return (
                ApplyOutcome.UNATTRIBUTED,
                f"shard {record.target.shard_id!r} is no longer part of the layout",
                None,
            )
        if (
            record.target.shard_start != shard.start
            or record.target.local_index >= shard.length
            or record.target.global_index != record.target.token_id
            or record.target.token_id >= shard.stop
        ):
            return (
                ApplyOutcome.SHARD_MISMATCH,
                "record target no longer points at the same row of the shard",
                None,
            )
        current = state.rows.get(record.target.token_id)
        if current is None:
            return (
                ApplyOutcome.UNATTRIBUTED,
                "row was removed from the shard",
                None,
            )
        if row_fingerprint(current) != record.row_fingerprint:
            return (
                ApplyOutcome.STALE,
                "row values changed since the forward pass; update is no longer attributable",
                None,
            )
        return ApplyOutcome.APPLIED, None, state

    def apply(
        self,
        batch: UpdateBatch | Iterable[UpdateRecord],
        *,
        learning_rate: float | None = None,
        step: int | None = None,
        record_results: bool = True,
        dry_run: bool = False,
    ) -> ApplyReport:
        """Apply every record, skipping ones that lost their attribution.

        ``dry_run`` classifies records exactly as a real apply would but does
        not mutate rows or the applied-update ledger.  It is the tool used when
        resuming: it tells the caller which updates are still replayable and
        which have lost their attribution.
        """

        records = batch.records if isinstance(batch, UpdateBatch) else tuple(batch)
        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        if step is not None:
            batch_step = int(step)
        elif isinstance(batch, UpdateBatch):
            batch_step = batch.step
        else:
            batch_step = records[0].step if records else 0
        results: list[ApplyResult] = []
        touched: set[str] = set()
        for record in records:
            if record.step != batch_step:
                results.append(
                    ApplyResult(
                        update_id=record.update_id,
                        token_id=record.target.token_id,
                        version=record.target.version,
                        shard_id=record.target.shard_id,
                        outcome=ApplyOutcome.STEP_MISMATCH,
                        reason=(
                            f"record step {record.step} does not match batch step {batch_step}"
                        ),
                    )
                )
                continue
            outcome, reason, state = self._classify(record)
            if outcome is ApplyOutcome.APPLIED and state is not None:
                if dry_run:
                    # Report a would-be apply without touching the row; classify
                    # it as DUPLICATE-free by leaving state untouched.
                    results.append(
                        ApplyResult(
                            update_id=record.update_id,
                            token_id=record.target.token_id,
                            version=record.target.version,
                            shard_id=record.target.shard_id,
                            outcome=ApplyOutcome.APPLIED,
                            reason="dry_run",
                        )
                    )
                    continue
                row = state.rows[record.target.token_id]
                for axis, value in enumerate(record.gradient):
                    row[axis] += lr * value
                touched.add(record.target.version)
                if record_results:
                    self._applied.add(record.update_id)
            results.append(
                ApplyResult(
                    update_id=record.update_id,
                    token_id=record.target.token_id,
                    version=record.target.version,
                    shard_id=record.target.shard_id,
                    outcome=outcome,
                    reason=reason,
                )
            )
        if touched and not dry_run and self.publish:
            for version_name in touched:
                self._states[version_name].version = self.registry.replace(
                    self._states[version_name].vocabulary()
                )
        return ApplyReport(results=tuple(results), step=batch_step, learning_rate=lr)

    def replayability(
        self,
        batch: UpdateBatch | Iterable[UpdateRecord],
        *,
        step: int | None = None,
    ) -> "ReplayabilityReport":
        """Classify a batch without mutating anything.

        Convenience wrapper around :meth:`apply` with ``dry_run=True`` that
        splits update ids into replayable and lost groups for resume logic.
        """

        report = self.apply(batch, step=step, dry_run=True)
        records = (
            {record.update_id: record for record in batch.records}
            if isinstance(batch, UpdateBatch)
            else {record.update_id: record for record in batch}
        )
        return ReplayabilityReport.from_apply(report, records)

    # -- checkpoints --------------------------------------------------
    def checkpoint(self) -> dict[str, object]:
        """Serialise rows and the set of applied update ids."""

        return {
            "format": "embedding-shards-checkpoint",
            "format_version": 1,
            "versions": {
                version: {
                    "layout_fingerprint": state.layout_fingerprint,
                    "rows": {
                        str(token_id): list(values)
                        for token_id, values in sorted(state.rows.items())
                    },
                }
                for version, state in self._states.items()
            },
            "applied_update_ids": sorted(self._applied),
        }

    @classmethod
    def from_checkpoint(
        cls,
        registry: VocabularyRegistry,
        payload: Mapping[str, object],
        *,
        learning_rate: float = 1.0,
    ) -> "ParameterStore":
        """Rebuild a store, validating every layout against the registry."""

        if payload.get("format") != "embedding-shards-checkpoint":
            raise CheckpointError("unrecognised checkpoint format")
        if int(payload.get("format_version", 0)) != 1:
            raise CheckpointError("unsupported checkpoint format_version")
        versions = payload.get("versions")
        if not isinstance(versions, Mapping) or not versions:
            raise CheckpointError("checkpoint contains no versions")
        store = cls(registry, learning_rate=learning_rate)
        for version_name, entry in versions.items():
            if not isinstance(entry, Mapping):
                raise CheckpointError(f"checkpoint entry for {version_name!r} is malformed")
            state = store._states.get(str(version_name))
            if state is None:
                # A checkpointed version not in the registry cannot be restored;
                # keep it detached so updates for it report UNATTRIBUTED.
                raise UnknownVocabularyVersion(
                    f"checkpoint references unknown version {version_name!r}"
                )
            layout = str(entry.get("layout_fingerprint", ""))
            if layout != state.layout_fingerprint:
                raise CheckpointError(
                    f"checkpoint layout for {version_name!r} does not match the registry"
                )
            rows = entry.get("rows")
            if not isinstance(rows, Mapping):
                raise CheckpointError(f"checkpoint rows for {version_name!r} are malformed")
            for token_id, values in rows.items():
                token = int(token_id)
                if token not in state.rows:
                    raise CheckpointError(
                        f"checkpoint row {token} is not part of version {version_name!r}"
                    )
                row = [float(value) for value in values]  # type: ignore[union-attr]
                if len(row) != state.version.dim:
                    raise CheckpointError(
                        f"checkpoint row {token} has dim {len(row)}, "
                        f"expected {state.version.dim}"
                    )
                state.rows[token] = row
        applied = payload.get("applied_update_ids", ())
        if not isinstance(applied, (list, tuple)):
            raise CheckpointError("applied_update_ids must be a list")
        store._applied = {str(item) for item in applied}
        if store.publish:
            for version_name, state in store._states.items():
                state.version = registry.replace(state.vocabulary())
        return store

    def shard_view(self, version: str) -> dict[str, Mapping[int, tuple[float, ...]]]:
        """Return ``{shard_id: {token_id: values}}`` for optimiser sharding."""

        state = self._states.get(version)
        if state is None:
            raise UnknownVocabularyVersion(f"version {version!r} is not tracked")
        view: dict[str, dict[int, tuple[float, ...]]] = {}
        for shard in state.version.shards:
            view[shard.shard_id] = {
                token_id: tuple(state.rows[token_id])
                for token_id in range(shard.start, shard.stop)
            }
        return view


__all__ = [
    "ApplyOutcome",
    "ApplyResult",
    "ApplyReport",
    "ReplayabilityReport",
    "ParameterStore",
]
