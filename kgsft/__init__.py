"""Public contracts for auditable KG-to-SFT compilation."""

from .compiler import compile_dataset
from .evaluation import support_document_components
from .exposure import FixtureLoaderAdapter, replay_loader
from .graph import build_multidigraph
from .schema import (
    ContractExample,
    EvidenceChunk,
    GraphBundle,
    GraphNode,
    GraphRelation,
    SourceRef,
    ValidationError,
)
from .transforms import BudgetRepairError, repair_to_budget, replace_distractors

__all__ = [
    "BudgetRepairError",
    "ContractExample",
    "EvidenceChunk",
    "FixtureLoaderAdapter",
    "GraphBundle",
    "GraphNode",
    "GraphRelation",
    "SourceRef",
    "ValidationError",
    "build_multidigraph",
    "compile_dataset",
    "repair_to_budget",
    "replace_distractors",
    "replay_loader",
    "support_document_components",
]

__version__ = "0.1.0"

