"""Regression tests for fail-closed integration boundaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from kgsft.compiler import compile_dataset
from kgsft.evaluation import support_document_components
from kgsft.graph import validate_graph_lineage
from kgsft.schema import (
    ContractExample,
    EvidenceChunk,
    GraphBundle,
    GraphNode,
    GraphRelation,
    SourceRef,
    ValidationError,
)


class SyntheticAdapter:
    """Production-shaped adapter with visible support serialization."""

    name = "review-synthetic-adapter-v1"

    def __init__(self, split: Any) -> None:
        self.split = split
        self.serialized_payloads: list[str] = []

    def serialize(self, example: ContractExample) -> str:
        context = "\n".join(
            chunk.text for chunk in (*example.support, *example.distractors)
        )
        payload = (
            f"Question: {example.question}\nContext: {context}\n"
            f"Answer: {example.target}"
        )
        self.serialized_payloads.append(payload)
        return payload

    def token_ids(self, serialized: Any) -> Sequence[int]:
        return tuple(range(len(str(serialized).split())))

    def partition(self, example_id: str) -> Any:
        return self.split


def make_example(
    *,
    example_id: str = "review:q1",
    distractors: Sequence[EvidenceChunk] = (),
) -> ContractExample:
    source = SourceRef("support-doc", revision="1")
    support = EvidenceChunk(
        chunk_id=f"{example_id}:support",
        text="Atlas version two uses OAuth2.",
        role="support",
        source_refs=(source,),
    )
    return ContractExample(
        example_id=example_id,
        question="What authentication does Atlas version two use?",
        target="OAuth2",
        support=(support,),
        distractors=tuple(distractors),
        source_refs=(source,),
    )


def make_graph(*refs: SourceRef) -> GraphBundle:
    sources = refs or (SourceRef("support-doc", revision="1"),)
    return GraphBundle(
        graph_id="review:graph",
        nodes=(GraphNode("review:node", "service", "Atlas", sources),),
        relations=(),
    )


def graph_distractor(*refs: SourceRef) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id="review:distractor",
        text="A legacy Atlas configuration uses an API key.",
        role="distractor",
        source_refs=refs,
        origin="graph",
        graph_record_ids=("review:node",),
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ReviewRegressionTests(unittest.TestCase):
    def test_unknown_split_fails_before_export(self) -> None:
        for split in ("unexpected_split", "", None, []):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    compile_dataset(
                        [make_example()],
                        make_graph(),
                        root,
                        max_tokens=256,
                        adapter=SyntheticAdapter(split),
                    )
                self.assertFalse((root / "train.jsonl").exists())
                self.assertFalse((root / "validation.jsonl").exists())

    def test_valid_splits_conserve_retained_ids(self) -> None:
        for split in ("train", "validation"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                example = make_example()
                adapter = SyntheticAdapter(split)
                report = compile_dataset(
                    [example],
                    make_graph(),
                    root,
                    max_tokens=256,
                    adapter=adapter,
                )
                train = read_rows(root / "train.jsonl")
                validation = read_rows(root / "validation.jsonl")
                exported_ids = [row["example_id"] for row in train + validation]
                self.assertEqual(exported_ids, [example.example_id])
                self.assertEqual(len(exported_ids), report["exposure"]["retained"])
                self.assertTrue(report["exports"]["retained_ids_conserved"])
                self.assertTrue(report["exports"]["splits_disjoint"])
                self.assertEqual(len(train), int(split == "train"))
                self.assertEqual(len(validation), int(split == "validation"))
                expected_status = "unknown" if split == "train" else "not_applicable"
                self.assertEqual(
                    report["exposure"]["optimizer_exposure_status"],
                    expected_status,
                )
                self.assertFalse(report["exposure"]["optimizer_exposure_known"])

    def test_production_shaped_adapter_preserves_support(self) -> None:
        adapter = SyntheticAdapter("train")
        example = make_example()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compile_dataset(
                [example],
                make_graph(),
                root,
                max_tokens=256,
                adapter=adapter,
            )
            exported = read_rows(root / "train.jsonl")
        self.assertTrue(adapter.serialized_payloads)
        self.assertTrue(
            all("Atlas version two uses OAuth2." in value for value in adapter.serialized_payloads)
        )
        self.assertEqual(exported[0]["support"][0]["text"], example.support[0].text)

    def test_matching_source_cannot_hide_another_conflict(self) -> None:
        source_a = SourceRef("doc-A", revision="1", sha256="a" * 64)
        source_b = SourceRef("doc-B", revision="2", sha256="b" * 64)
        conflicts = (
            SourceRef("doc-B", revision="999", sha256="b" * 64),
            SourceRef("doc-B", revision="2", sha256="c" * 64),
        )
        graph = make_graph(source_a, source_b)
        for conflicting in conflicts:
            with self.subTest(conflicting=conflicting):
                example = make_example(
                    distractors=(graph_distractor(source_a, conflicting),)
                )
                with self.assertRaises(ValidationError):
                    validate_graph_lineage(graph, [example])

    def test_consistent_source_subset_remains_valid(self) -> None:
        source_a = SourceRef("doc-A", revision="1", sha256="a" * 64)
        source_b = SourceRef("doc-B", revision="2", sha256="b" * 64)
        report = validate_graph_lineage(
            make_graph(source_a, source_b),
            [make_example(distractors=(graph_distractor(source_a),))],
        )
        self.assertEqual(report["graph_origin_chunks"], 1)
        self.assertEqual(report["referenced_graph_records"], 1)

    def test_null_or_empty_document_ids_fail_closed(self) -> None:
        for document_ids in ([None], ["doc-A", None], [""], ["  "]):
            with self.subTest(document_ids=document_ids), self.assertRaises(ValueError):
                support_document_components(
                    [
                        {
                            "example_id": "q1",
                            "support_document_ids": document_ids,
                        }
                    ]
                )

    def test_missing_direction_remains_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraphRelation.from_dict(
                {
                    "relation_id": "r1",
                    "source_node_id": "a",
                    "target_node_id": "b",
                    "relation_type": "LINK",
                    "source_refs": [{"source_id": "doc-A"}],
                }
            )


if __name__ == "__main__":
    unittest.main()
