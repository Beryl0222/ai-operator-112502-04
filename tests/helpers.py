"""Fixture helpers shared by the operator tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).parents[1]
SRC = BASE / "src"
if str(SRC) not in sys.path:  # allow `unittest discover` without an install
    sys.path.insert(0, str(SRC))

from operator_embedding_shards import (  # noqa: E402
    ShardSpec,
    VocabularyRegistry,
    VocabularyVersion,
    build_vocabulary,
    vocabulary_from_payload,
)


def load_payload() -> dict:
    return json.loads((BASE / "data/input.json").read_text(encoding="utf-8"))


def fixture_vocabulary() -> VocabularyVersion:
    return vocabulary_from_payload(load_payload())


def fixture_registry() -> VocabularyRegistry:
    return VocabularyRegistry([fixture_vocabulary()])


def simple_vocabulary(
    version: str = "v1",
    *,
    unknown_id: int = -1,
    disabled: frozenset[int] = frozenset(),
    unknown_vector=None,
    disabled_vector=None,
) -> VocabularyVersion:
    """One shard with ids 0..5, each row ``(id, id * 2)``."""

    return build_vocabulary(
        version,
        [ShardSpec("s", 0, tuple((float(i), float(i * 2)) for i in range(6)))],
        unknown_id=unknown_id,
        disabled=disabled,
        unknown_vector=unknown_vector,
        disabled_vector=disabled_vector,
    )


def second_version() -> VocabularyVersion:
    """Same geometry as the fixture but different row values for ``a``."""

    return build_vocabulary(
        "2026-11-b",
        [
            ShardSpec("a", 0, ((10.0, 10.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0))),
            ShardSpec("b", 4, ((3.0, 0.0), (0.0, 3.0), (2.0, 2.0), (4.0, 1.0))),
        ],
        unknown_id=-1,
        disabled=frozenset({6}),
    )
