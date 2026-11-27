"""Deterministic fingerprints used for update attribution.

The operator must be able to tell whether an update record still belongs to
the row it was produced from.  Two fingerprints are used:

``layout``
    Covers the vocabulary version plus the *geometry* of every shard (shard
    name, first token id and row count) and the embedding dimension.  Row
    values are deliberately excluded, because applying optimiser updates
    changes values without invalidating the shard layout.

``row``
    Covers the actual values of one embedding row.  It protects against
    applying a gradient that was computed from an outdated snapshot after a
    concurrent step has already modified the row.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence


def _canonical_json(parts: object) -> bytes:
    return json.dumps(
        parts,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def stable_digest(*parts: object) -> str:
    """Return a stable sha256 digest for JSON-serialisable parts."""

    return hashlib.sha256(_canonical_json(list(parts))).hexdigest()


def row_fingerprint(values: Iterable[float]) -> str:
    """Fingerprint the values of a single embedding row."""

    return stable_digest("row", [float(value) for value in values])


def layout_fingerprint(
    version: str,
    dim: int,
    geometry: Sequence[tuple[str, int, int]],
) -> str:
    """Fingerprint a vocabulary layout without embedding values."""

    return stable_digest(
        "layout",
        version,
        int(dim),
        [[str(shard_id), int(start), int(length)] for shard_id, start, length in geometry],
    )


__all__ = ["stable_digest", "row_fingerprint", "layout_fingerprint"]
