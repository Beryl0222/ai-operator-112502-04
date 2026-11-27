"""Vocabulary versions, shard geometry and per-token resolution.

A :class:`VocabularyVersion` is an immutable snapshot: it owns the shard
layout (which shard, if any, holds a token id) and the embedding row values
for that version.  Resolving an id returns both the vector and the shard the
vector came from, so downstream optimiser records can be attributed to a
shard instead of a single flat table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence

from .errors import (
    LayoutConflictError,
    ShardLayoutError,
    UnknownVocabularyVersion,
    VocabularyError,
)
from .hashing import layout_fingerprint, row_fingerprint, stable_digest


class TokenStatus(str, Enum):
    """Outcome of resolving one token id against a vocabulary version."""

    OK = "ok"
    UNKNOWN = "unknown"
    DISABLED = "disabled"
    MISSING = "missing"
    PADDING = "padding"

    @property
    def is_embeddable(self) -> bool:
        return self is TokenStatus.OK

    @property
    def allows_update(self) -> bool:
        return self is TokenStatus.OK


@dataclass(frozen=True)
class ShardSpec:
    """One contiguous shard of the embedding table.

    ``start`` is the first token id owned by the shard; the shard owns
    ``start .. start + len(values) - 1``.  Gaps between shards are legal and
    resolve to :attr:`TokenStatus.MISSING`.
    """

    shard_id: str
    start: int
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shard_id, str) or not self.shard_id:
            raise ShardLayoutError("shard_id must be a non-empty string")
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise ShardLayoutError(f"shard {self.shard_id!r}: start must be an int")
        if self.start < 0:
            raise ShardLayoutError(f"shard {self.shard_id!r}: start must be >= 0")
        if not self.values:
            raise ShardLayoutError(f"shard {self.shard_id!r}: needs at least one row")
        dim = len(self.values[0])
        if dim == 0:
            raise ShardLayoutError(f"shard {self.shard_id!r}: embedding dim must be > 0")
        normalised: list[tuple[float, ...]] = []
        for index, row in enumerate(self.values):
            if len(row) != dim:
                raise ShardLayoutError(
                    f"shard {self.shard_id!r}: row {index} has dim {len(row)}, expected {dim}"
                )
            normalised.append(tuple(float(value) for value in row))
        object.__setattr__(self, "values", tuple(normalised))

    @property
    def dim(self) -> int:
        return len(self.values[0])

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def stop(self) -> int:
        """Exclusive upper token id bound for this shard."""

        return self.start + self.length

    def contains(self, token_id: int) -> bool:
        return self.start <= token_id < self.stop

    def row(self, token_id: int) -> tuple[float, ...]:
        if not self.contains(token_id):
            raise IndexError(f"token id {token_id} is outside shard {self.shard_id!r}")
        return self.values[token_id - self.start]

    def __len__(self) -> int:
        return self.length


@dataclass(frozen=True)
class Resolution:
    """How one token id resolved against a vocabulary version."""

    token_id: int
    status: TokenStatus
    shard_id: str | None = None
    shard_start: int | None = None
    local_index: int | None = None
    reason: str | None = None

    @property
    def is_embeddable(self) -> bool:
        return self.status is TokenStatus.OK

    @property
    def allows_update(self) -> bool:
        return self.status is TokenStatus.OK

    def as_dict(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "status": self.status.value,
            "shard_id": self.shard_id,
            "shard_start": self.shard_start,
            "local_index": self.local_index,
            "reason": self.reason,
        }


def _default_vector(dim: int, value: float) -> tuple[float, ...]:
    return tuple([float(value)] * dim)


@dataclass(frozen=True)
class VocabularyVersion:
    """An immutable vocabulary layout plus its embedding rows.

    ``unknown_id`` names the sentinel id that always resolves as
    :attr:`TokenStatus.UNKNOWN`, even if a shard range happens to cover it.
    ``disabled`` lists ids that must resolve without a shard attribution and
    must never accept a gradient update.
    """

    version: str
    shards: tuple[ShardSpec, ...]
    unknown_id: int = -1
    disabled: frozenset[int] = frozenset()
    unknown_vector: tuple[float, ...] | None = None
    disabled_vector: tuple[float, ...] | None = None
    _index: Mapping[int, tuple[int, int]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise VocabularyError("version must be a non-empty string")
        if not self.shards:
            raise ShardLayoutError(f"vocabulary {self.version!r}: at least one shard required")
        if isinstance(self.unknown_id, bool) or not isinstance(self.unknown_id, int):
            raise VocabularyError("unknown_id must be an int")

        seen_ids: dict[str, ShardSpec] = {}
        seen_tokens: dict[int, str] = {}
        dim: int | None = None
        index: dict[int, tuple[int, int]] = {}
        for shard_index, shard in enumerate(self.shards):
            if not isinstance(shard, ShardSpec):
                raise ShardLayoutError("shards must be ShardSpec instances")
            if shard.shard_id in seen_ids:
                raise ShardLayoutError(f"duplicate shard_id {shard.shard_id!r}")
            if dim is None:
                dim = shard.dim
            elif shard.dim != dim:
                raise ShardLayoutError(
                    f"shard {shard.shard_id!r} has dim {shard.dim}, expected {dim}"
                )
            for token_id in range(shard.start, shard.stop):
                other = seen_tokens.get(token_id)
                if other is not None:
                    raise ShardLayoutError(
                        f"token id {token_id} appears in shards {other!r} and "
                        f"{shard.shard_id!r}"
                    )
                seen_tokens[token_id] = shard.shard_id
                index[token_id] = (shard_index, token_id - shard.start)
            seen_ids[shard.shard_id] = shard

        disabled = frozenset(int(value) for value in self.disabled)
        # A disabled id that lives in a shard still blocks updates; a disabled
        # id outside every shard is simply never embeddable.
        for token_id in disabled:
            if token_id == self.unknown_id:
                raise VocabularyError(
                    "unknown_id cannot also be disabled; it must stay a fallback"
                )

        if self.unknown_vector is not None and len(self.unknown_vector) != dim:
            raise VocabularyError(
                f"unknown_vector has dim {len(self.unknown_vector)}, expected {dim}"
            )
        if self.disabled_vector is not None and len(self.disabled_vector) != dim:
            raise VocabularyError(
                f"disabled_vector has dim {len(self.disabled_vector)}, expected {dim}"
            )

        object.__setattr__(self, "disabled", disabled)
        object.__setattr__(
            self, "unknown_vector", self._normalised_vector(self.unknown_vector, dim, 0.0)
        )
        object.__setattr__(
            self, "disabled_vector", self._normalised_vector(self.disabled_vector, dim, 0.0)
        )
        object.__setattr__(self, "_index", MappingProxyType(index))
        object.__setattr__(self, "dim", dim)
        object.__setattr__(
            self,
            "layout_fingerprint",
            layout_fingerprint(
                self.version,
                int(dim),
                [(shard.shard_id, shard.start, shard.length) for shard in self.shards],
            ),
        )
        # Row fingerprints let optimiser hand-off records detect stale snapshots.
        row_fps: dict[int, str] = {}
        for shard in self.shards:
            for offset, values in enumerate(shard.values):
                row_fps[shard.start + offset] = row_fingerprint(values)
        object.__setattr__(self, "_row_fingerprints", MappingProxyType(row_fps))
        object.__setattr__(
            self,
            "content_fingerprint",
            stable_digest(
                "content",
                self.layout_fingerprint,
                [[shard.shard_id, list(shard.values)] for shard in self.shards],
                list(self.unknown_vector),
                list(self.disabled_vector),
                sorted(self.disabled),
                int(self.unknown_id),
            ),
        )

    dim: int = field(init=False)
    layout_fingerprint: str = field(init=False)
    content_fingerprint: str = field(init=False)
    _row_fingerprints: Mapping[int, str] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    @staticmethod
    def _normalised_vector(
        values: Sequence[float] | None, dim: int | None, default: float
    ) -> tuple[float, ...]:
        if values is None:
            assert dim is not None
            return _default_vector(dim, default)
        return tuple(float(value) for value in values)

    # -- geometry -----------------------------------------------------
    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(shard.shard_id for shard in self.shards)

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)

    def token_range(self) -> tuple[int, int]:
        """Inclusive lower and exclusive upper bounds across all shards."""

        return min(shard.start for shard in self.shards), max(
            shard.stop for shard in self.shards
        )

    def geometry(self) -> tuple[tuple[str, int, int], ...]:
        return tuple((shard.shard_id, shard.start, shard.length) for shard in self.shards)

    # -- resolution ---------------------------------------------------
    def resolve(self, token_id: int) -> Resolution:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise VocabularyError(f"token id must be an int, got {type(token_id).__name__}")
        if token_id == self.unknown_id:
            return Resolution(
                token_id=token_id,
                status=TokenStatus.UNKNOWN,
                reason="token id is the configured unknown sentinel",
            )
        if token_id in self.disabled:
            location = self._index.get(token_id)
            return Resolution(
                token_id=token_id,
                status=TokenStatus.DISABLED,
                shard_id=self.shards[location[0]].shard_id if location else None,
                shard_start=self.shards[location[0]].start if location else None,
                local_index=location[1] if location else None,
                reason="token id is disabled by the vocabulary version",
            )
        location = self._index.get(token_id)
        if location is not None:
            shard_index, local_index = location
            shard = self.shards[shard_index]
            return Resolution(
                token_id=token_id,
                status=TokenStatus.OK,
                shard_id=shard.shard_id,
                shard_start=shard.start,
                local_index=local_index,
            )
        lower, upper = self.token_range()
        if lower <= token_id < upper:
            reason = "token id falls in a gap between shards"
        else:
            reason = "token id is outside every shard range"
        return Resolution(token_id=token_id, status=TokenStatus.MISSING, reason=reason)

    def lookup(self, token_id: int) -> tuple[float, ...]:
        """Return the raw row for an embeddable id, rejecting everything else."""

        resolution = self.resolve(token_id)
        if resolution.status is not TokenStatus.OK:
            raise KeyError(
                f"token id {token_id} resolved as {resolution.status.value}"
            )
        assert resolution.shard_id is not None and resolution.local_index is not None
        return self.shard(resolution.shard_id).values[resolution.local_index]

    def vector_for(self, token_id: int) -> tuple[tuple[float, ...], Resolution]:
        """Return the vector a forward pass should emit plus its resolution.

        Unknown ids receive ``unknown_vector``; disabled ids receive
        ``disabled_vector``; missing ids receive the unknown fallback too, but
        keep :attr:`TokenStatus.MISSING` so data-quality tooling can tell a
        gap in the shard layout from a genuine unknown token.
        """

        resolution = self.resolve(token_id)
        if resolution.status is TokenStatus.OK:
            assert resolution.shard_id is not None and resolution.local_index is not None
            return self.shard(resolution.shard_id).values[resolution.local_index], resolution
        if resolution.status is TokenStatus.DISABLED:
            return self.disabled_vector, resolution
        return self.unknown_vector, resolution

    def row_fingerprint(self, token_id: int) -> str | None:
        resolution = self.resolve(token_id)
        if resolution.status is not TokenStatus.OK:
            return None
        return self._row_fingerprints[token_id]

    def with_values(
        self,
        values: Mapping[int, Sequence[float]],
    ) -> "VocabularyVersion":
        """Return a copy with selected rows replaced (used by the registry)."""

        pending = {int(token_id): tuple(float(v) for v in row) for token_id, row in values.items()}
        new_shards: list[ShardSpec] = []
        for shard in self.shards:
            rows = list(shard.values)
            changed = False
            for offset in range(shard.length):
                token_id = shard.start + offset
                if token_id in pending:
                    row = pending.pop(token_id)
                    if len(row) != shard.dim:
                        raise VocabularyError(
                            f"replacement for token {token_id} has dim {len(row)}, "
                            f"expected {shard.dim}"
                        )
                    rows[offset] = row
                    changed = True
            if changed:
                new_shards.append(ShardSpec(shard.shard_id, shard.start, tuple(rows)))
            else:
                new_shards.append(shard)
        if pending:
            raise UnknownVocabularyVersion(
                f"cannot replace rows for ids outside the layout: {sorted(pending)}"
            )
        return VocabularyVersion(
            version=self.version,
            shards=tuple(new_shards),
            unknown_id=self.unknown_id,
            disabled=self.disabled,
            unknown_vector=self.unknown_vector,
            disabled_vector=self.disabled_vector,
        )

    def iter_rows(self) -> Iterator[tuple[int, str, int, tuple[float, ...]]]:
        """Yield ``(token_id, shard_id, local_index, values)`` for every row."""

        for shard in self.shards:
            for offset, values in enumerate(shard.values):
                yield shard.start + offset, shard.shard_id, offset, values


class VocabularyRegistry:
    """Holds multiple immutable vocabulary versions for batch-time lookup.

    A version name may be registered more than once only with identical
    content; re-registering the same snapshot returns the existing instance,
    which makes checkpoint restore idempotent.  Re-using a version name for a
    *different* snapshot (different geometry, row values, disabled set or
    fallbacks) raises :class:`LayoutConflictError` so a batch can never
    silently mix two vocabularies that share a version string.  Use
    :meth:`replace` to advance the row values of an existing version while
    keeping its layout.
    """

    def __init__(self, versions: Iterable[VocabularyVersion] = ()) -> None:
        self._versions: dict[str, VocabularyVersion] = {}
        for version in versions:
            self.register(version)

    def register(self, version: VocabularyVersion) -> VocabularyVersion:
        if not isinstance(version, VocabularyVersion):
            raise TypeError("register expects a VocabularyVersion")
        existing = self._versions.get(version.version)
        if existing is None:
            self._versions[version.version] = version
            return version
        if existing.content_fingerprint == version.content_fingerprint:
            return existing
        if existing.layout_fingerprint != version.layout_fingerprint:
            raise LayoutConflictError(
                f"version {version.version!r} is already registered with a different layout"
            )
        raise LayoutConflictError(
            f"version {version.version!r} is already registered with different row values; "
            "use replace() to advance an existing layout"
        )

    def get(self, version: str) -> VocabularyVersion:
        try:
            return self._versions[version]
        except KeyError:
            raise UnknownVocabularyVersion(
                f"vocabulary version {version!r} is not registered; "
                f"known versions: {sorted(self._versions)}"
            ) from None

    def replace(self, version: VocabularyVersion) -> VocabularyVersion:
        """Replace rows of a registered version without changing its layout."""

        existing = self.get(version.version)
        if existing.layout_fingerprint != version.layout_fingerprint:
            raise LayoutConflictError(
                f"cannot replace version {version.version!r} with a different layout"
            )
        self._versions[version.version] = version
        return version

    def __contains__(self, version: object) -> bool:
        return version in self._versions

    def __iter__(self) -> Iterator[VocabularyVersion]:
        return iter(self._versions.values())

    def __len__(self) -> int:
        return len(self._versions)

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(self._versions)

    def as_dict(self) -> dict[str, VocabularyVersion]:
        return dict(self._versions)


def build_vocabulary(
    version: str,
    shards: Iterable[ShardSpec | Mapping[str, object]],
    *,
    unknown_id: int = -1,
    disabled: Iterable[int] = (),
    unknown_vector: Sequence[float] | None = None,
    disabled_vector: Sequence[float] | None = None,
) -> VocabularyVersion:
    """Convenience constructor accepting plain mappings for shard definitions."""

    specs: list[ShardSpec] = []
    for item in shards:
        if isinstance(item, ShardSpec):
            specs.append(item)
        else:
            specs.append(
                ShardSpec(
                    shard_id=str(item["shard_id"]),
                    start=int(item["start"]),
                    values=tuple(tuple(row) for row in item["values"]),  # type: ignore[arg-type]
                )
            )
    return VocabularyVersion(
        version=version,
        shards=tuple(specs),
        unknown_id=unknown_id,
        disabled=frozenset(disabled),
        unknown_vector=tuple(unknown_vector) if unknown_vector is not None else None,
        disabled_vector=tuple(disabled_vector) if disabled_vector is not None else None,
    )


def vocabulary_from_payload(payload: Mapping[str, object]) -> VocabularyVersion:
    """Build a :class:`VocabularyVersion` from the fixture/schema JSON shape."""

    return build_vocabulary(
        version=str(payload["vocab_version"]),
        shards=payload["shards"],  # type: ignore[arg-type]
        unknown_id=int(payload.get("unknown_id", -1)),
        disabled=payload.get("disabled", ()),  # type: ignore[arg-type]
        unknown_vector=payload.get("unknown_vector"),  # type: ignore[arg-type]
        disabled_vector=payload.get("disabled_vector"),  # type: ignore[arg-type]
    )


__all__ = [
    "TokenStatus",
    "ShardSpec",
    "Resolution",
    "VocabularyVersion",
    "VocabularyRegistry",
    "build_vocabulary",
    "vocabulary_from_payload",
]
