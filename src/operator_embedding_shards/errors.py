"""Exceptions raised by the sharded embedding operator.

The package deliberately keeps the error hierarchy small.  Callers that only
consume update records should be able to catch :class:`OperatorError` and
inspect the message while training loops that validate shard metadata can
catch the more specific vocabulary errors.
"""

from __future__ import annotations


class OperatorError(Exception):
    """Base class for every error raised by this package."""


class VocabularyError(OperatorError):
    """A vocabulary/shard definition is inconsistent."""


class ShardLayoutError(VocabularyError):
    """Shards overlap, use duplicate names, or have inconsistent dimensions."""


class UnknownVocabularyVersion(VocabularyError):
    """A request referenced a vocabulary version that is not registered."""


class LayoutConflictError(VocabularyError):
    """A version name was reused for a different shard layout."""


class BatchShapeError(OperatorError):
    """A micro-batch has an invalid shape (ids versus per-token versions)."""


class GradientShapeError(OperatorError):
    """Backward gradients do not align with the forward result."""


class AggregationError(OperatorError):
    """The requested gradient aggregation rule is not supported."""


class MissingTokenError(OperatorError):
    """An id had no row and the operator was configured to reject it."""


class CheckpointError(OperatorError):
    """A checkpoint payload is malformed or does not match its layout."""


__all__ = [
    "OperatorError",
    "VocabularyError",
    "ShardLayoutError",
    "UnknownVocabularyVersion",
    "LayoutConflictError",
    "BatchShapeError",
    "GradientShapeError",
    "AggregationError",
    "MissingTokenError",
    "CheckpointError",
]
