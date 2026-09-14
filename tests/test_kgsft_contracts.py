from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from kgsft.compiler import compile_dataset
from kgsft.adapters.rubq import import_rubq
from kgsft.evaluation import ClaimRecord, support_document_components, validate_claim_record
from kgsft.exposure import LoaderAdapter
from kgsft.graph import build_multidigraph, validate_graph_lineage
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

    def load_claim(self):
        return json.loads(
            (ROOT / "examples/atlas/claim.json").read_text(encoding="utf-8")
        )

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
                relations=(GraphRelation("r", "a", "b", "LINK", (), True),),
            )

    def test_relation_without_explicit_direction_fails_closed(self):
        payload = json.loads(
            (ROOT / "examples/atlas/graph.json").read_text(encoding="utf-8")
        )
        del payload["relations"][0]["directed"]
        with self.assertRaises(ValidationError):
            GraphBundle.from_dict(payload)

    def test_graph_origin_chunk_must_resolve_to_graph_record(self):
        graph, examples = self.load_fixture()
        broken = examples[0].to_dict()
        broken["distractors"][0]["graph_record_ids"] = ["missing-relation"]
        with self.assertRaises(ValidationError):
            validate_graph_lineage(graph, [ContractExample.from_dict(broken)])
        broken = examples[0].to_dict()
        broken["distractors"][0]["source_refs"][0]["revision"] = "999"
        with self.assertRaises(ValidationError):
            validate_graph_lineage(graph, [ContractExample.from_dict(broken)])
        broken = examples[0].to_dict()
        broken["distractors"][0]["source_refs"] = [{"source_id": "unrelated-source"}]
        with self.assertRaises(ValidationError):
            validate_graph_lineage(graph, [ContractExample.from_dict(broken)])

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

    def test_compile_uses_caller_supplied_loader_adapter(self):
        class TestAdapter:
            name = "test-production-adapter-v1"

            def __init__(self):
                self.serialized_ids = []

            def serialize(self, example):
                self.serialized_ids.append(example.example_id)
                return f"{example.question} {example.target}"

            def token_ids(self, serialized):
                return tuple(range(len(str(serialized).split())))

            def partition(self, example_id):
                return "train"

        graph, examples = self.load_fixture()
        adapter: LoaderAdapter = TestAdapter()
        with tempfile.TemporaryDirectory() as directory:
            report = compile_dataset(
                examples,
                graph,
                directory,
                max_tokens=100,
                adapter=adapter,
            )
        self.assertEqual(report["config"]["loader_adapter"], adapter.name)
        self.assertTrue(report["config"]["caller_supplied_adapter"])
        self.assertIsNone(report["config"]["fixture_validation_fraction"])
        self.assertEqual(report["exposure"]["train_assigned"], 2)
        self.assertGreaterEqual(len(adapter.serialized_ids), len(examples) * 2)

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

    def test_claim_contract_accepts_compatible_runs(self):
        report = validate_claim_record(ClaimRecord.from_dict(self.load_claim()))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["comparison_contract"]["denominator"], 2)
        self.assertEqual(len(report["claim_record_sha256"]), 64)

    def test_claim_contract_rejects_incompatible_or_incomplete_runs(self):
        cases = []
        roster_mismatch = copy.deepcopy(self.load_claim())
        roster_mismatch["run_b"]["roster_sha256"] = "9" * 64
        cases.append(roster_mismatch)
        decoding_mismatch = copy.deepcopy(self.load_claim())
        decoding_mismatch["run_b"]["decoding_fingerprint"] = "8" * 64
        cases.append(decoding_mismatch)
        missing_score_origin = copy.deepcopy(self.load_claim())
        missing_score_origin["run_a"]["score_origin"] = ""
        cases.append(missing_score_origin)
        missing_denominator = copy.deepcopy(self.load_claim())
        del missing_denominator["run_b"]["denominator"]
        cases.append(missing_denominator)
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                validate_claim_record(payload)

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
