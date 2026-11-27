"""Public entry points for the sharded vocabulary embedding operator."""

from .errors import (
    AggregationError,
    BatchShapeError,
    CheckpointError,
    GradientShapeError,
    LayoutConflictError,
    MissingTokenError,
    OperatorError,
    ShardLayoutError,
    UnknownVocabularyVersion,
    VocabularyError,
)
from .operator import EmbeddingGather, EmbeddingShards, GatherResult, TokenVector
from .quality import DataQualityReport
from .store import (
    ApplyOutcome,
    ApplyReport,
    ApplyResult,
    ParameterStore,
    ReplayabilityReport,
)
from .updates import (
    AggregationRule,
    BlockedUpdate,
    SkippedToken,
    UpdateBatch,
    UpdateRecord,
    UpdateTarget,
    aggregate_gradients,
)
from .vocab import (
    Resolution,
    ShardSpec,
    TokenStatus,
    VocabularyRegistry,
    VocabularyVersion,
    build_vocabulary,
    vocabulary_from_payload,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # errors
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
    # vocabulary
    "TokenStatus",
    "ShardSpec",
    "Resolution",
    "VocabularyVersion",
    "VocabularyRegistry",
    "build_vocabulary",
    "vocabulary_from_payload",
    # operator
    "TokenVector",
    "GatherResult",
    "EmbeddingShards",
    "EmbeddingGather",
    # updates
    "AggregationRule",
    "UpdateTarget",
    "UpdateRecord",
    "BlockedUpdate",
    "SkippedToken",
    "UpdateBatch",
    "aggregate_gradients",
    # store
    "ApplyOutcome",
    "ApplyResult",
    "ApplyReport",
    "ReplayabilityReport",
    "ParameterStore",
    # quality
    "DataQualityReport",
]
