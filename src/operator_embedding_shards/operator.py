"""Shard-aware embedding gather and its backward pass.

Forward
    ``EmbeddingShards.forward(ids, versions=...)`` resolves every token id
    against the vocabulary version declared for that micro-batch position,
    emits the vector in the original token order, and records the shard each
    vector belongs to together with its status (ok / unknown / disabled /
    missing / padding).  Repeated ids inside one micro-batch share the same
    vector snapshot; the later positions record ``reuse_of`` pointing at the
    first position.

Backward
    ``EmbeddingShards.backward(forward, token_gradients, aggregation=...)``
    groups occurrences by ``(version, token_id)`` and merges them with the
    declared rule (sum / mean / first / last / error).  Only ``ok`` targets
    produce rows; disabled ids become :class:`BlockedUpdate` entries carrying
    a data-quality reason and never touch a row.  Unknown, missing and padding
    positions are recorded as :class:`SkippedToken` entries.

Because every emitted vector remembers the *version* and *layout fingerprint*
it was produced under, a version switch inside a batch cannot make a later
version's gradient overwrite a vector that an earlier version already
produced.  The store enforces this on apply; this module merely preserves the
provenance in the update records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .errors import (
    AggregationError,
    BatchShapeError,
    GradientShapeError,
    MissingTokenError,
)
from .hashing import stable_digest
from .updates import (
    AggregationRule,
    BlockedUpdate,
    SkippedToken,
    UpdateBatch,
    UpdateRecord,
    UpdateTarget,
    aggregate_gradients,
)
from .vocab import Resolution, TokenStatus, VocabularyRegistry, VocabularyVersion


@dataclass(frozen=True)
class TokenVector:
    """One emitted vector plus everything needed to attribute it later."""

    position: int
    token_id: int
    version: str
    values: tuple[float, ...]
    status: TokenStatus
    shard_id: str | None
    shard_start: int | None
    local_index: int | None
    layout_fingerprint: str
    row_fingerprint: str | None
    reason: str | None = None
    # Position of the first occurrence when the same (version, token_id)
    # already appeared earlier in this micro-batch; ``None`` otherwise.
    reuse_of: int | None = None

    @property
    def is_embeddable(self) -> bool:
        return self.status is TokenStatus.OK

    @property
    def target(self) -> UpdateTarget | None:
        if self.status is not TokenStatus.OK:
            return None
        assert self.shard_id is not None
        assert self.shard_start is not None
        assert self.local_index is not None
        return UpdateTarget(
            version=self.version,
            token_id=self.token_id,
            shard_id=self.shard_id,
            shard_start=self.shard_start,
            local_index=self.local_index,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "token_id": self.token_id,
            "version": self.version,
            "values": list(self.values),
            "status": self.status.value,
            "shard_id": self.shard_id,
            "shard_start": self.shard_start,
            "local_index": self.local_index,
            "layout_fingerprint": self.layout_fingerprint,
            "row_fingerprint": self.row_fingerprint,
            "reason": self.reason,
            "reuse_of": self.reuse_of,
        }


@dataclass(frozen=True)
class GatherResult:
    """Result of one forward gather; the backward pass consumes it."""

    micro_batch_id: str
    step: int
    vectors: tuple[TokenVector, ...]
    aggregation: AggregationRule
    versions: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self) -> Iterator[TokenVector]:
        return iter(self.vectors)

    def __getitem__(self, index: int) -> TokenVector:
        return self.vectors[index]

    @property
    def values(self) -> tuple[tuple[float, ...], ...]:
        """Vectors in the original token order."""

        return tuple(vector.values for vector in self.vectors)

    @property
    def dim(self) -> int:
        return len(self.vectors[0].values) if self.vectors else 0

    def statuses(self) -> tuple[TokenStatus, ...]:
        return tuple(vector.status for vector in self.vectors)

    def shard_attribution(self) -> tuple[str | None, ...]:
        return tuple(vector.shard_id for vector in self.vectors)

    def non_embeddable(self) -> tuple[TokenVector, ...]:
        return tuple(vector for vector in self.vectors if vector.status is not TokenStatus.OK)

    def as_dict(self) -> dict[str, object]:
        return {
            "micro_batch_id": self.micro_batch_id,
            "step": self.step,
            "aggregation": self.aggregation.value,
            "versions": list(self.versions),
            "vectors": [vector.as_dict() for vector in self.vectors],
        }


def _normalise_versions(
    versions: str | Sequence[str] | None,
    ids_length: int,
    default: str,
) -> tuple[str, ...]:
    if versions is None:
        return (default,) * ids_length
    if isinstance(versions, str):
        return (versions,) * ids_length
    resolved = tuple(str(item) for item in versions)
    if len(resolved) != ids_length:
        raise BatchShapeError(
            f"ids has {ids_length} entries but versions has {len(resolved)}; "
            "provide one version per token position"
        )
    return resolved


def _normalise_gradients(
    gradients: Iterable[Iterable[float] | None],
    ids_length: int,
    dim: int,
    vectors: Sequence[TokenVector],
) -> tuple[tuple[float, ...] | None, ...]:
    """Validate one gradient row per position.

    An embeddable position must supply a gradient of the embedding dimension.
    A non-embeddable position (unknown / disabled / missing / padding) may pass
    ``None`` because no row update will be produced for it; a supplied row is
    still checked and ignored.
    """

    collected: list[tuple[float, ...] | None] = []
    for row in gradients:
        if row is None:
            collected.append(None)
        else:
            collected.append(tuple(float(value) for value in row))
    if len(collected) != ids_length:
        raise GradientShapeError(
            f"expected {ids_length} token gradients, received {len(collected)}"
        )
    for position, (gradient, vector) in enumerate(zip(collected, vectors)):
        if gradient is None:
            if vector.is_embeddable:
                raise GradientShapeError(
                    f"gradient at position {position} is required: token id "
                    f"{vector.token_id} is embeddable under version {vector.version!r}"
                )
            continue
        if len(gradient) != dim:
            raise GradientShapeError(
                f"gradient at position {position} has dim {len(gradient)}, expected {dim}"
            )
    return tuple(collected)


class EmbeddingShards:
    """Forward gather over a :class:`VocabularyRegistry` and its backward pass."""

    def __init__(
        self,
        registry: VocabularyRegistry,
        *,
        default_version: str | None = None,
        aggregation: AggregationRule | str = AggregationRule.SUM,
        padding_id: int | None = None,
        padding_vector: Sequence[float] | None = None,
        reject_missing: bool = False,
    ) -> None:
        if not isinstance(registry, VocabularyRegistry):
            raise TypeError("registry must be a VocabularyRegistry")
        self.registry = registry
        if default_version is None:
            known = registry.versions
            if len(known) != 1:
                raise ValueError(
                    "default_version is required when the registry holds more than one version"
                )
            default_version = known[0]
        # Resolve eagerly so a bad default fails before the first batch.
        registry.get(default_version)
        self.default_version = default_version
        self.aggregation = AggregationRule.parse(aggregation)
        self.padding_id = padding_id
        self.padding_vector = (
            tuple(float(value) for value in padding_vector)
            if padding_vector is not None
            else None
        )
        if self.padding_vector is not None and len(self.padding_vector) != registry.get(
            default_version
        ).dim:
            raise ValueError("padding_vector has the wrong embedding dimension")
        self.reject_missing = reject_missing

    # -- resolution ---------------------------------------------------
    def resolve(
        self,
        ids: Sequence[int],
        *,
        versions: str | Sequence[str] | None = None,
    ) -> tuple[Resolution, ...]:
        """Resolve ids without computing vectors.

        Useful for the data-quality path and for pre-flight validation of a
        batch that mixes vocabulary versions.
        """

        ids = tuple(ids)
        self._validate_ids(ids)
        resolved_versions = _normalise_versions(versions, len(ids), self.default_version)
        cache: dict[str, VocabularyVersion] = {}
        resolutions: list[Resolution] = []
        for token_id, version_name in zip(ids, resolved_versions):
            vocab = cache.get(version_name)
            if vocab is None:
                vocab = self.registry.get(version_name)
                cache[version_name] = vocab
            if self.padding_id is not None and token_id == self.padding_id:
                resolutions.append(
                    Resolution(
                        token_id=token_id,
                        status=TokenStatus.PADDING,
                        reason="token id is the configured padding id",
                    )
                )
                continue
            resolutions.append(vocab.resolve(token_id))
        return tuple(resolutions)

    # -- forward ------------------------------------------------------
    def forward(
        self,
        ids: Sequence[int],
        *,
        versions: str | Sequence[str] | None = None,
        micro_batch_id: str | None = None,
        step: int = 0,
        aggregation: AggregationRule | str | None = None,
    ) -> GatherResult:
        """Resolve ``ids`` in order and return vectors plus shard attribution.

        ``versions`` may be one version for the whole batch or one entry per
        token position.  Positions that switch version mid-batch are resolved
        against the version declared *for that position*; a later version never
        affects vectors already produced.  Within a single forward call every
        version name is snapshotted once, so even a concurrent registry swap
        cannot change the vectors of positions that come later in the same
        call.
        """

        ids = tuple(ids)
        self._validate_ids(ids)
        resolved_versions = _normalise_versions(versions, len(ids), self.default_version)
        rule = (
            AggregationRule.parse(aggregation)
            if aggregation is not None
            else self.aggregation
        )

        vocab_cache: dict[str, VocabularyVersion] = {}
        first_seen: dict[tuple[str, int], int] = {}
        vectors: list[TokenVector] = []
        seen_versions: list[str] = []

        for position, (token_id, version_name) in enumerate(zip(ids, resolved_versions)):
            vocab = vocab_cache.get(version_name)
            if vocab is None:
                vocab = self.registry.get(version_name)
                vocab_cache[version_name] = vocab
            if version_name not in seen_versions:
                seen_versions.append(version_name)

            if self.padding_id is not None and token_id == self.padding_id:
                vectors.append(
                    TokenVector(
                        position=position,
                        token_id=token_id,
                        version=version_name,
                        values=(
                            self.padding_vector
                            if self.padding_vector is not None
                            else vocab.unknown_vector
                        ),
                        status=TokenStatus.PADDING,
                        shard_id=None,
                        shard_start=None,
                        local_index=None,
                        layout_fingerprint=vocab.layout_fingerprint,
                        row_fingerprint=None,
                        reason="token id is the configured padding id",
                    )
                )
                continue

            values, resolution = vocab.vector_for(token_id)
            if self.reject_missing and resolution.status is TokenStatus.MISSING:
                raise MissingTokenError(
                    f"token id {token_id} at position {position} is missing from "
                    f"vocabulary {version_name!r}: {resolution.reason}"
                )

            key = (version_name, token_id)
            reuse_of = first_seen.get(key)
            if reuse_of is None:
                first_seen[key] = position
            vectors.append(
                TokenVector(
                    position=position,
                    token_id=token_id,
                    version=version_name,
                    values=values,
                    status=resolution.status,
                    shard_id=resolution.shard_id,
                    shard_start=resolution.shard_start,
                    local_index=resolution.local_index,
                    layout_fingerprint=vocab.layout_fingerprint,
                    row_fingerprint=vocab.row_fingerprint(token_id),
                    reason=resolution.reason,
                    reuse_of=reuse_of,
                )
            )

        return GatherResult(
            micro_batch_id=micro_batch_id or _default_micro_batch_id(ids, resolved_versions),
            step=int(step),
            vectors=tuple(vectors),
            aggregation=rule,
            versions=tuple(seen_versions),
        )

    def embed(
        self,
        ids: Sequence[int],
        *,
        versions: str | Sequence[str] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """Return only the gathered vectors, in original token order."""

        return self.forward(ids, versions=versions).values

    # -- backward -----------------------------------------------------
    def backward(
        self,
        forward: GatherResult,
        gradients: Iterable[Iterable[float] | None],
        *,
        aggregation: AggregationRule | str | None = None,
        step: int | None = None,
        micro_batch_id: str | None = None,
    ) -> UpdateBatch:
        """Turn per-position gradients into attributable update records.

        Gradient rows are expected in the same order as the forward vectors.
        Repeated ids are merged with ``aggregation``; the resulting record
        records every contributing position so the optimiser can audit how a
        row update was formed.  The merge is never a silent overwrite.
        """

        if not isinstance(forward, GatherResult):
            raise TypeError("backward expects the GatherResult returned by forward")
        rule = (
            AggregationRule.parse(aggregation)
            if aggregation is not None
            else forward.aggregation
        )
        if rule is AggregationRule.ERROR:
            repeated = self._repeated_keys(forward)
            if repeated:
                preview = ", ".join(
                    f"{version}:{token_id}" for version, token_id in repeated[:5]
                )
                raise AggregationError(
                    "aggregation rule 'error' rejects repeated ids in one micro-batch: "
                    f"{preview}"
                )

        grad_rows = _normalise_gradients(
            gradients, len(forward.vectors), forward.dim, forward.vectors
        )

        grouped: dict[tuple[str, int], list[tuple[int, tuple[float, ...]]]] = {}
        order: list[tuple[str, int]] = []
        representatives: dict[tuple[str, int], TokenVector] = {}
        for vector, gradient in zip(forward.vectors, grad_rows):
            if vector.status is not TokenStatus.OK:
                continue
            assert gradient is not None
            key = (vector.version, vector.token_id)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
                representatives[key] = vector
            grouped[key].append((vector.position, gradient))

        blocked = self._collect_blocked(forward)
        skipped = tuple(
            SkippedToken(
                token_id=vector.token_id,
                version=vector.version,
                occurrence=vector.position,
                status=vector.status,
                reason=vector.reason,
            )
            for vector in forward.vectors
            if vector.status not in (TokenStatus.OK, TokenStatus.DISABLED)
        )

        records: list[UpdateRecord] = []
        for key in order:
            representative = representatives[key]
            contributions = grouped[key]
            merged = aggregate_gradients(
                [gradient for _, gradient in contributions], rule, dim=forward.dim
            )
            target = representative.target
            assert target is not None
            assert representative.row_fingerprint is not None
            records.append(
                UpdateRecord(
                    target=target,
                    gradient=merged,
                    occurrences=tuple(position for position, _ in contributions),
                    aggregation=rule,
                    layout_fingerprint=representative.layout_fingerprint,
                    row_fingerprint=representative.row_fingerprint,
                    micro_batch_id=micro_batch_id or forward.micro_batch_id,
                    step=forward.step if step is None else int(step),
                )
            )

        return UpdateBatch(
            micro_batch_id=micro_batch_id or forward.micro_batch_id,
            step=forward.step if step is None else int(step),
            aggregation=rule,
            records=tuple(records),
            blocked=blocked,
            skipped=skipped,
            versions=forward.versions,
        )

    @staticmethod
    def _collect_blocked(forward: GatherResult) -> tuple[BlockedUpdate, ...]:
        merged: dict[tuple[str, int], BlockedUpdate] = {}
        for vector in forward.vectors:
            if vector.status is not TokenStatus.DISABLED:
                continue
            key = (vector.version, vector.token_id)
            previous = merged.get(key)
            if previous is None:
                merged[key] = BlockedUpdate(
                    token_id=vector.token_id,
                    version=vector.version,
                    occurrences=(vector.position,),
                    shard_id=vector.shard_id,
                    shard_start=vector.shard_start,
                    local_index=vector.local_index,
                    reason=vector.reason
                    or "disabled ids must not receive gradient updates",
                    layout_fingerprint=vector.layout_fingerprint,
                )
            else:
                merged[key] = BlockedUpdate(
                    token_id=previous.token_id,
                    version=previous.version,
                    occurrences=previous.occurrences + (vector.position,),
                    shard_id=previous.shard_id,
                    shard_start=previous.shard_start,
                    local_index=previous.local_index,
                    reason=previous.reason,
                    layout_fingerprint=previous.layout_fingerprint,
                )
        return tuple(merged.values())

    # -- convenience --------------------------------------------------
    def __call__(
        self,
        ids: Sequence[int],
        gradients: Iterable[Iterable[float] | None],
        *,
        versions: str | Sequence[str] | None = None,
        aggregation: AggregationRule | str | None = None,
        micro_batch_id: str | None = None,
        step: int = 0,
    ) -> tuple[GatherResult, UpdateBatch]:
        """Forward then backward in one call."""

        forward = self.forward(
            ids,
            versions=versions,
            micro_batch_id=micro_batch_id,
            step=step,
            aggregation=aggregation,
        )
        return forward, self.backward(forward, gradients)

    @staticmethod
    def _validate_ids(ids: Sequence[int]) -> None:
        for position, token_id in enumerate(ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise BatchShapeError(
                    f"ids[{position}] must be an int, got {type(token_id).__name__}"
                )

    @staticmethod
    def _repeated_keys(forward: GatherResult) -> list[tuple[str, int]]:
        counts: dict[tuple[str, int], int] = {}
        for vector in forward.vectors:
            if vector.status is TokenStatus.OK:
                key = (vector.version, vector.token_id)
                counts[key] = counts.get(key, 0) + 1
        return [key for key, count in counts.items() if count > 1]


# The gather-facing name is an alias: one class implements forward *and*
# backward so the two halves can never drift apart.
EmbeddingGather = EmbeddingShards


def _default_micro_batch_id(ids: Sequence[int], versions: Sequence[str]) -> str:
    return stable_digest("micro-batch", list(ids), list(versions))[:24]


__all__ = [
    "TokenVector",
    "GatherResult",
    "EmbeddingShards",
    "EmbeddingGather",
]
