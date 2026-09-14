#!/usr/bin/env python3
"""Run the deterministic, text-safe KG-SFT public smoke test twice."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kgsft.compiler import compile_dataset, sha256_file
from kgsft.graph import build_multidigraph
from kgsft.schema import ContractExample, GraphBundle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "compile_audit.json",
    "exposure_ledger.json",
)


def _load_inputs() -> tuple[GraphBundle, list[ContractExample]]:
    graph = GraphBundle.from_dict(
        json.loads((ROOT / "examples/atlas/graph.json").read_text(encoding="utf-8"))
    )
    examples = [
        ContractExample.from_dict(json.loads(line))
        for line in (ROOT / "examples/atlas/examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return graph, examples


def main() -> int:
    graph, examples = _load_inputs()
    directed = build_multidigraph(graph)
    with tempfile.TemporaryDirectory(prefix="kgsft-smoke-") as directory:
        root = Path(directory)
        reports = [
            compile_dataset(
                examples,
                graph,
                root / f"run-{index}",
                max_tokens=36,
                validation_fraction=0.1,
                seed=228,
            )
            for index in (1, 2)
        ]
        hashes = [
            {
                name: sha256_file(root / f"run-{index}" / name)
                for name in OUTPUT_FILES
            }
            for index in (1, 2)
        ]
    if reports[0] != reports[1] or hashes[0] != hashes[1]:
        raise RuntimeError("deterministic rerun mismatch")
    if reports[0]["exposure"]["retained"] != len(examples):
        raise RuntimeError("fixture loader did not retain every repaired example")
    if reports[0]["budget"]["edited_rows"] != 1:
        raise RuntimeError("fixture did not exercise selective budget repair")
    if reports[0]["examples"]["graph_origin_distractor_rows"] != 1:
        raise RuntimeError("fixture did not retain a graph-origin distractor")
    if reports[0]["exposure"]["train_assigned"] != 1:
        raise RuntimeError("fixture did not produce the expected train assignment")
    if reports[0]["exposure"]["validation_assigned"] != 1:
        raise RuntimeError("fixture did not produce the expected validation assignment")
    result = {
        "status": "pass",
        "fixture": "atlas-authentication-v1",
        "examples": len(examples),
        "nodes": directed.number_of_nodes(),
        "directed_relations": directed.number_of_edges(),
        "all_examples_retained": True,
        "edited_examples": reports[0]["budget"]["edited_rows"],
        "removed_distractor_chunks": reports[0]["budget"]["removed_distractor_chunks"],
        "retained_graph_origin_rows": reports[0]["examples"]["graph_origin_distractor_rows"],
        "train_assigned": reports[0]["exposure"]["train_assigned"],
        "validation_assigned": reports[0]["exposure"]["validation_assigned"],
        "protected_invariants_preserved": reports[0]["examples"]["invariants_preserved"],
        "deterministic_rerun": True,
        "output_sha256": hashes[0],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
