"""Loader replay interfaces and exposure ledgers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from .schema import ContractExample


VALID_ASSIGNED_SPLITS = frozenset({"train", "validation"})


class LoaderAdapter(Protocol):
    name: str

    def serialize(self, example: ContractExample) -> Any: ...

    def token_ids(self, serialized: Any) -> Sequence[int]: ...

    def partition(self, example_id: str) -> str: ...


@dataclass(frozen=True)
class ExposureRecord:
    example_id: str
    intended: bool
    retained: bool
    assigned_split: str | None
    token_count: int
    reason: str
    optimizer_exposed: bool | None


@dataclass(frozen=True)
class ExposureLedger:
    adapter: str
    records: tuple[ExposureRecord, ...]

    @property
    def summary(self) -> dict[str, Any]:
        train_records = [
            record for record in self.records if record.assigned_split == "train"
        ]
        if not train_records:
            optimizer_exposure_status = "not_applicable"
        elif all(record.optimizer_exposed is not None for record in train_records):
            optimizer_exposure_status = "known"
        else:
            optimizer_exposure_status = "unknown"
        return {
            "adapter": self.adapter,
            "intended": len(self.records),
            "retained": sum(record.retained for record in self.records),
            "dropped": sum(not record.retained for record in self.records),
            "train_assigned": sum(record.assigned_split == "train" for record in self.records),
            "validation_assigned": sum(record.assigned_split == "validation" for record in self.records),
            "optimizer_exposure_applicable": bool(train_records),
            "optimizer_exposure_known": optimizer_exposure_status == "known",
            "optimizer_exposure_status": optimizer_exposure_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "records": [asdict(record) for record in self.records]}


class FixtureLoaderAdapter:
    """Deterministic public smoke-test loader; not a production tokenizer."""

    name = "fixture-whitespace-v1"

    def __init__(self, max_tokens: int = 256, validation_fraction: float = 0.1, seed: int = 228):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        self.max_tokens = int(max_tokens)
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)

    def serialize(self, example: ContractExample) -> str:
        chunks = [*example.support, *example.distractors]
        context = "\n".join(f"[{chunk.role}] {chunk.text}" for chunk in chunks)
        return f"Question: {example.question}\nContext:\n{context}\nAnswer: {example.target}"

    def token_ids(self, serialized: Any) -> Sequence[int]:
        return tuple(range(len(str(serialized).split())))

    def partition(self, example_id: str) -> str:
        value = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{example_id}".encode("utf-8")).digest()[:8],
            "big",
        ) / 2**64
        return "validation" if value < self.validation_fraction else "train"


def replay_loader(
    examples: Sequence[ContractExample],
    adapter: LoaderAdapter,
    *,
    max_tokens: int,
) -> ExposureLedger:
    """Replay serialization/retention without claiming optimizer exposure."""

    seen: set[str] = set()
    records: list[ExposureRecord] = []
    for example in examples:
        if example.example_id in seen:
            raise ValueError(f"duplicate example_id in loader input: {example.example_id!r}")
        seen.add(example.example_id)
        serialized = adapter.serialize(example)
        token_count = len(tuple(adapter.token_ids(serialized)))
        retained = 0 < token_count <= max_tokens
        split = None
        if retained:
            raw_split = adapter.partition(example.example_id)
            if (
                not isinstance(raw_split, str)
                or raw_split not in VALID_ASSIGNED_SPLITS
            ):
                raise ValueError(
                    f"loader adapter {adapter.name!r} returned invalid split "
                    f"{raw_split!r} for {example.example_id!r}; expected "
                    "'train' or 'validation'"
                )
            split = raw_split
        records.append(
            ExposureRecord(
                example_id=example.example_id,
                intended=True,
                retained=retained,
                assigned_split=split,
                token_count=token_count,
                reason="retained" if retained else "empty_or_over_budget",
                optimizer_exposed=None,
            )
        )
    return ExposureLedger(adapter=adapter.name, records=tuple(records))
