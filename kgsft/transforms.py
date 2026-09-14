"""Invariant-preserving context transformations and token-budget repair."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from .schema import ContractExample, EvidenceChunk


class BudgetRepairError(RuntimeError):
    """Raised when a row cannot fit without mutating protected fields."""


def replace_distractors(
    example: ContractExample,
    distractors: Sequence[EvidenceChunk],
) -> ContractExample:
    candidate = replace(example, distractors=tuple(distractors))
    if candidate.invariant_sha256 != example.invariant_sha256:
        raise AssertionError("context transform changed a protected field")
    return candidate


@dataclass(frozen=True)
class BudgetRepairResult:
    example: ContractExample
    tokens_before: int
    tokens_after: int
    removed_chunk_ids: tuple[str, ...]
    invariant_sha256: str


def repair_to_budget(
    example: ContractExample,
    *,
    serialize: Callable[[ContractExample], Any],
    count_tokens: Callable[[Any], int],
    max_tokens: int,
) -> BudgetRepairResult:
    """Remove distractors from the tail until an exact serializer fits.

    The tokenizer is supplied by the caller so the same production serializer
    can be replayed. Support, question, target, task type and example-level
    provenance are immutable. Failure is explicit if support alone is too long.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    invariant = example.invariant_sha256
    before = int(count_tokens(serialize(example)))
    current = example
    removed: list[str] = []
    while int(count_tokens(serialize(current))) > max_tokens and current.distractors:
        removed.append(current.distractors[-1].chunk_id)
        current = replace_distractors(current, current.distractors[:-1])
    after = int(count_tokens(serialize(current)))
    if after > max_tokens:
        raise BudgetRepairError(
            f"example {example.example_id!r} has {after} protected tokens; "
            f"cannot satisfy max_tokens={max_tokens} by removing distractors"
        )
    if current.invariant_sha256 != invariant:
        raise AssertionError("budget repair changed a protected field")
    return BudgetRepairResult(
        example=current,
        tokens_before=before,
        tokens_after=after,
        removed_chunk_ids=tuple(removed),
        invariant_sha256=invariant,
    )

