"""Deterministic compiler from canonical examples to loader-visible SFT rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .exposure import FixtureLoaderAdapter, LoaderAdapter, replay_loader
from .graph import build_multidigraph, validate_graph_lineage
from .schema import ContractExample, GraphBundle, canonical_sha256
from .transforms import repair_to_budget


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def compile_dataset(
    examples: Sequence[ContractExample],
    graph: GraphBundle,
    output_dir: Path | str,
    *,
    max_tokens: int = 256,
    validation_fraction: float = 0.1,
    seed: int = 228,
    adapter: LoaderAdapter | None = None,
) -> dict[str, Any]:
    """Validate, repair, replay and write a self-auditing SFT bundle.

    When ``adapter`` is omitted, the deterministic public fixture adapter is
    used. Production callers should pass an adapter that reproduces their real
    serializer, tokenizer and split assignment.
    """

    output = Path(output_dir).expanduser().resolve()
    directed_graph = build_multidigraph(graph)
    lineage = validate_graph_lineage(graph, examples)
    caller_supplied_adapter = adapter is not None
    loader_adapter = adapter or FixtureLoaderAdapter(
        max_tokens=max_tokens,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    repaired = []
    repair_records = []
    for example in examples:
        result = repair_to_budget(
            example,
            serialize=loader_adapter.serialize,
            count_tokens=lambda value: len(loader_adapter.token_ids(value)),
            max_tokens=max_tokens,
        )
        repaired.append(result.example)
        repair_records.append(
            {
                "example_id": example.example_id,
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
                "removed_chunk_ids": list(result.removed_chunk_ids),
                "invariant_sha256": result.invariant_sha256,
            }
        )
    ledger = replay_loader(repaired, loader_adapter, max_tokens=max_tokens)
    split_by_id = {
        record.example_id: record.assigned_split
        for record in ledger.records
        if record.retained
    }
    train_examples = [
        example
        for example in repaired
        if split_by_id.get(example.example_id) == "train"
    ]
    validation_examples = [
        example
        for example in repaired
        if split_by_id.get(example.example_id) == "validation"
    ]
    retained_ids = {
        record.example_id for record in ledger.records if record.retained
    }
    train_ids = {example.example_id for example in train_examples}
    validation_ids = {example.example_id for example in validation_examples}
    if train_ids & validation_ids:
        raise RuntimeError("train and validation exports contain overlapping IDs")
    if train_ids | validation_ids != retained_ids:
        missing = sorted(retained_ids - train_ids - validation_ids)
        unexpected = sorted((train_ids | validation_ids) - retained_ids)
        raise RuntimeError(
            "retained/exported example-ID conservation failed: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if (
        len(train_examples) != ledger.summary["train_assigned"]
        or len(validation_examples) != ledger.summary["validation_assigned"]
        or len(train_examples) + len(validation_examples)
        != ledger.summary["retained"]
    ):
        raise RuntimeError("retained/exported row-count conservation failed")
    train = [example.to_dict() for example in train_examples]
    validation = [example.to_dict() for example in validation_examples]
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "validation.jsonl", validation)
    _write_json(output / "exposure_ledger.json", ledger.to_dict())
    audit = {
        "artifact_type": "kgsft_compile_audit",
        "schema_version": 1,
        "graph": {
            "graph_id": graph.graph_id,
            "nodes": directed_graph.number_of_nodes(),
            "directed_relations": directed_graph.number_of_edges(),
            "is_directed": directed_graph.is_directed(),
            "all_objects_have_provenance": True,
            **lineage,
        },
        "examples": {
            "input": len(examples),
            "output": len(repaired),
            "invariants_preserved": all(
                before.invariant_sha256 == after.invariant_sha256
                for before, after in zip(examples, repaired)
            ),
            "graph_origin_distractor_rows": sum(
                any(chunk.origin == "graph" for chunk in example.distractors)
                for example in repaired
            ),
        },
        "budget": {
            "max_tokens": max_tokens,
            "edited_rows": sum(bool(record["removed_chunk_ids"]) for record in repair_records),
            "removed_distractor_chunks": sum(len(record["removed_chunk_ids"]) for record in repair_records),
            "records": repair_records,
        },
        "exposure": ledger.summary,
        "exports": {
            "retained_ids_conserved": True,
            "splits_disjoint": True,
            "train_rows": len(train),
            "validation_rows": len(validation),
        },
        "config": {
            "loader_adapter": loader_adapter.name,
            "caller_supplied_adapter": caller_supplied_adapter,
            "max_tokens": max_tokens,
            "fixture_validation_fraction": (
                None if caller_supplied_adapter else validation_fraction
            ),
            "fixture_seed": None if caller_supplied_adapter else seed,
        },
    }
    audit["content_sha256"] = canonical_sha256(audit)
    _write_json(output / "compile_audit.json", audit)
    return audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
