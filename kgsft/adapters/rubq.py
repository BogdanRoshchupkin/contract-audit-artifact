"""Strict adapter for the official RuBQ 2.0 JSON release."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..schema import (
    ContractExample,
    EvidenceChunk,
    SourceRef,
    ValidationError,
    canonical_sha256,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paragraph_index(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        if "paragraphs" in value:
            return _paragraph_index(value["paragraphs"])
        return {
            str(key): str(text if not isinstance(text, Mapping) else text.get("text") or text.get("value") or "")
            for key, text in value.items()
            if str(text if not isinstance(text, Mapping) else text.get("text") or text.get("value") or "").strip()
        }
    if isinstance(value, list):
        result: dict[str, str] = {}
        for item in value:
            if not isinstance(item, Mapping):
                continue
            uid = item.get("uid", item.get("id", item.get("paragraph_id")))
            text = item.get("text", item.get("paragraph", item.get("value")))
            if uid is not None and str(text or "").strip():
                result[str(uid)] = str(text)
        return result
    raise ValidationError("RuBQ paragraph file must be an object or array")


def _answer_text(row: Mapping[str, Any]) -> str:
    direct = str(row.get("answer_text") or "").strip()
    if direct:
        return direct
    answers = row.get("answers")
    if not isinstance(answers, list):
        return ""
    labels = []
    for answer in answers:
        if not isinstance(answer, Mapping):
            continue
        label = answer.get("label", answer.get("value"))
        if str(label or "").strip() and str(label) not in labels:
            labels.append(str(label))
    return "; ".join(labels)


def _paragraph_ids(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    paragraphs = row.get("paragraphs_uids")
    if not isinstance(paragraphs, Mapping):
        raise ValidationError("RuBQ row has no paragraphs_uids object")
    support_raw = paragraphs.get("with_answer", ())
    # The published specification names this field ``value`` while the
    # released test JSON uses ``all_related``. Accept exactly those two known
    # forms and record neither as an inferred label.
    all_raw = paragraphs.get("value", paragraphs.get("all_related", ()))
    if not isinstance(support_raw, list) or not isinstance(all_raw, list):
        raise ValidationError("RuBQ paragraphs_uids fields must be arrays")
    support = [str(value) for value in support_raw]
    all_ids = [str(value) for value in all_raw]
    return support, [value for value in all_ids if value not in set(support)]


def import_rubq(
    questions_path: Path | str,
    paragraphs_path: Path | str,
    *,
    limit: int = 100,
    seed: int = 228,
) -> tuple[list[ContractExample], dict[str, Any]]:
    """Convert answer-bearing RuBQ rows while preserving public provenance."""

    questions_file = Path(questions_path).expanduser().resolve()
    paragraphs_file = Path(paragraphs_path).expanduser().resolve()
    raw_questions = _load_json(questions_file)
    if not isinstance(raw_questions, list):
        raise ValidationError("official RuBQ question file must be a JSON array")
    paragraph_text = _paragraph_index(_load_json(paragraphs_file))
    eligible: list[Mapping[str, Any]] = []
    skipped = {
        "not_object": 0,
        "missing_core": 0,
        "no_answer_paragraph": 0,
        "missing_paragraph_text": 0,
    }
    for row in raw_questions:
        if not isinstance(row, Mapping):
            skipped["not_object"] += 1
            continue
        raw_uid = row.get("uid")
        uid = "" if raw_uid is None else str(raw_uid).strip()
        question = str(row.get("question_text") or "").strip()
        target = _answer_text(row)
        if not uid or not question or not target:
            skipped["missing_core"] += 1
            continue
        try:
            support_ids, distractor_ids = _paragraph_ids(row)
        except ValidationError:
            skipped["no_answer_paragraph"] += 1
            continue
        if not support_ids:
            skipped["no_answer_paragraph"] += 1
            continue
        if any(
            paragraph_id not in paragraph_text
            for paragraph_id in (*support_ids, *distractor_ids)
        ):
            skipped["missing_paragraph_text"] += 1
            continue
        eligible.append(row)
    eligible_count = len(eligible)
    rng = random.Random(seed)
    eligible = sorted(eligible, key=lambda row: str(row["uid"]))
    rng.shuffle(eligible)
    if limit > 0:
        eligible = eligible[:limit]
    examples: list[ContractExample] = []
    for row in eligible:
        uid = str(row["uid"])
        support_ids, distractor_ids = _paragraph_ids(row)
        source = SourceRef(
            source_id=f"rubq:{uid}",
            uri="https://github.com/vladislavneon/RuBQ",
            revision=str(row.get("RuBQ_version") or "2.0"),
            metadata={
                "question_uris": row.get("question_uris", []),
                "question_props": row.get("question_props", []),
            },
        )
        support = [
            EvidenceChunk(
                chunk_id=f"rubq-paragraph:{paragraph_id}",
                text=paragraph_text[paragraph_id],
                role="support",
                source_refs=(
                    SourceRef(
                        source_id=f"rubq-paragraph:{paragraph_id}",
                        uri="https://github.com/vladislavneon/RuBQ",
                        revision="2.0",
                    ),
                ),
                origin="document",
            )
            for paragraph_id in support_ids
        ]
        distractors = [
            EvidenceChunk(
                chunk_id=f"rubq-paragraph:{paragraph_id}",
                text=paragraph_text[paragraph_id],
                role="distractor",
                source_refs=(
                    SourceRef(
                        source_id=f"rubq-paragraph:{paragraph_id}",
                        uri="https://github.com/vladislavneon/RuBQ",
                        revision="2.0",
                    ),
                ),
                origin="retrieval",
            )
            for paragraph_id in distractor_ids
        ]
        examples.append(
            ContractExample(
                example_id=f"rubq:{uid}",
                question=str(row["question_text"]),
                target=_answer_text(row),
                support=support,
                distractors=distractors,
                source_refs=(source,),
                metadata={"tags": row.get("tags", []), "query_present": bool(row.get("query"))},
            )
        )
    report = {
        "artifact_type": "rubq_import_preflight",
        "schema_version": 1,
        "input_rows": len(raw_questions),
        "paragraphs_indexed": len(paragraph_text),
        "eligible_answer_bearing_rows": eligible_count,
        "selected_rows": len(examples),
        "selected_support_chunks": sum(len(example.support) for example in examples),
        "selected_distractor_chunks": sum(len(example.distractors) for example in examples),
        "selected_multi_support_rows": sum(len(example.support) > 1 for example in examples),
        "selected_graph_origin_rows": 0,
        "all_selected_paragraph_ids_resolved": True,
        "all_selected_objects_have_provenance": True,
        "seed": seed,
        "limit": limit,
        "skipped": skipped,
        "questions_basename": questions_file.name,
        "questions_sha256": _sha256_file(questions_file),
        "paragraphs_basename": paragraphs_file.name,
        "paragraphs_sha256": _sha256_file(paragraphs_file),
        "selected_example_ids_sha256": canonical_sha256(
            [example.example_id for example in examples]
        ),
        "selected_examples_sha256": canonical_sha256(
            [example.to_dict() for example in examples]
        ),
    }
    return examples, report
