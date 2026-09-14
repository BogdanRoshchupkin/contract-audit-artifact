from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kgsft.compiler import compile_dataset
from kgsft.adapters.rubq import import_rubq
from kgsft.evaluation import support_document_components
from kgsft.graph import build_multidigraph
from kgsft.schema import ContractExample, GraphBundle, GraphNode, GraphRelation, SourceRef, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def load_fixture(self):
        graph = GraphBundle.from_dict(
            json.loads((ROOT / "examples/atlas/graph.json").read_text(encoding="utf-8"))
        )
        examples = [
            ContractExample.from_dict(json.loads(line))
            for line in (ROOT / "examples/atlas/examples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return graph, examples

    def test_graph_is_directed_and_preserves_parallel_edge_keys(self):
        graph, _ = self.load_fixture()
        built = build_multidigraph(graph)
        self.assertTrue(built.is_directed())
        self.assertEqual(built.number_of_edges(), 3)
        self.assertFalse(built.has_edge("oauth2", "atlas-v2"))

    def test_relation_without_provenance_fails_closed(self):
        source = SourceRef("source")
        with self.assertRaises(ValidationError):
            GraphBundle(
                graph_id="broken",
                nodes=(GraphNode("a", "type", "A", (source,)), GraphNode("b", "type", "B", (source,))),
                relations=(GraphRelation("r", "a", "b", "LINK", ()),),
            )

    def test_compile_preserves_invariants_and_writes_ledger(self):
        graph, examples = self.load_fixture()
        with tempfile.TemporaryDirectory() as directory:
            report = compile_dataset(
                examples,
                graph,
                directory,
                max_tokens=36,
                validation_fraction=0.1,
            )
            self.assertTrue(report["examples"]["invariants_preserved"])
            self.assertEqual(report["examples"]["input"], 2)
            self.assertEqual(report["budget"]["edited_rows"], 1)
            self.assertEqual(report["budget"]["removed_distractor_chunks"], 1)
            self.assertEqual(report["examples"]["graph_origin_distractor_rows"], 1)
            self.assertEqual(report["exposure"]["retained"], 2)
            self.assertEqual(report["exposure"]["train_assigned"], 1)
            self.assertEqual(report["exposure"]["validation_assigned"], 1)
            self.assertTrue((Path(directory) / "exposure_ledger.json").is_file())
            self.assertTrue((Path(directory) / "compile_audit.json").is_file())

    def test_support_components_use_source_documents(self):
        report = support_document_components(
            [
                {"example_id": "a", "support_document_ids": ["d1"]},
                {"example_id": "b", "support_document_ids": ["d1", "d2"]},
                {"example_id": "c", "support_document_ids": ["d3"]},
            ]
        )
        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["component_sizes"], [2, 1])

    def test_rubq_adapter_accepts_released_all_related_field(self):
        question = {
            "uid": 0,
            "question_text": "Who approves the expense?",
            "answer_text": "Finance",
            "answers": [],
            "paragraphs_uids": {"with_answer": [10], "all_related": [10, 11]},
            "question_uris": ["wd:Q7"],
            "question_props": ["wdt:P1"],
            "RuBQ_version": "2.0",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "questions.json").write_text(json.dumps([question]), encoding="utf-8")
            (root / "paragraphs.json").write_text(
                json.dumps({"10": "Finance approves it.", "11": "Travel exception."}),
                encoding="utf-8",
            )
            examples, report = import_rubq(
                root / "questions.json", root / "paragraphs.json", limit=100
            )
        self.assertEqual(report["eligible_answer_bearing_rows"], 1)
        self.assertEqual(examples[0].example_id, "rubq:0")
        self.assertEqual(len(examples[0].support), 1)
        self.assertEqual(len(examples[0].distractors), 1)
        self.assertEqual(len(report["questions_sha256"]), 64)
        self.assertEqual(len(report["paragraphs_sha256"]), 64)
        self.assertEqual(len(report["selected_examples_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
