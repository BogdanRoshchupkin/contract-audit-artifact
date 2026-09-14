"""Evaluation utilities and fail-closed comparison claim records."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import ValidationError, canonical_sha256


SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValidationError(f"{field_name} must be non-empty")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValidationError(f"{field_name} must be a 64-character SHA-256 digest")
    return digest


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError("metadata must be an object")
    return dict(value)


@dataclass(frozen=True)
class ComparisonRun:
    """One side of a comparison with every identity-bearing field declared."""

    label: str
    model_artifact_id: str
    model_artifact_sha256: str
    score_origin: str
    roster_sha256: str
    prompt_fingerprint: str
    decoding_fingerprint: str
    endpoint: str
    denominator: int
    missing_row_policy: str
    evidence_unit: str
    dependence_assignment_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "label",
            "model_artifact_id",
            "score_origin",
            "endpoint",
            "missing_row_policy",
            "evidence_unit",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        for field_name in (
            "model_artifact_sha256",
            "roster_sha256",
            "prompt_fingerprint",
            "decoding_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int):
            raise ValidationError("denominator must be a positive integer")
        if self.denominator <= 0:
            raise ValidationError("denominator must be a positive integer")
        dependence = str(self.dependence_assignment_sha256 or "").strip().lower()
        if dependence:
            dependence = _required_sha256(
                dependence, "dependence_assignment_sha256"
            )
        evidence_unit = self.evidence_unit.casefold()
        if ("component" in evidence_unit or "cluster" in evidence_unit) and not dependence:
            raise ValidationError(
                "component/cluster evidence units require dependence_assignment_sha256"
            )
        object.__setattr__(self, "dependence_assignment_sha256", dependence)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComparisonRun":
        if not isinstance(value, Mapping):
            raise ValidationError("comparison run must be an object")
        return cls(
            label=value.get("label", ""),
            model_artifact_id=value.get("model_artifact_id", ""),
            model_artifact_sha256=value.get("model_artifact_sha256", ""),
            score_origin=value.get("score_origin", ""),
            roster_sha256=value.get("roster_sha256", ""),
            prompt_fingerprint=value.get("prompt_fingerprint", ""),
            decoding_fingerprint=value.get("decoding_fingerprint", ""),
            endpoint=value.get("endpoint", ""),
            denominator=value.get("denominator", 0),
            missing_row_policy=value.get("missing_row_policy", ""),
            evidence_unit=value.get("evidence_unit", ""),
            dependence_assignment_sha256=value.get(
                "dependence_assignment_sha256", ""
            ),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class ClaimRecord:
    """A bounded interpretation plus the auditable comparison supporting it."""

    claim_id: str
    interpretation: str
    analysis_design: str
    run_a: ComparisonRun
    run_b: ComparisonRun
    estimate: float
    uncertainty_method: str
    confidence_interval: Sequence[float]
    p_value: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("claim_id", "interpretation", "analysis_design", "uncertainty_method"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        run_a = self.run_a if isinstance(self.run_a, ComparisonRun) else ComparisonRun.from_dict(self.run_a)
        run_b = self.run_b if isinstance(self.run_b, ComparisonRun) else ComparisonRun.from_dict(self.run_b)
        object.__setattr__(self, "run_a", run_a)
        object.__setattr__(self, "run_b", run_b)
        try:
            estimate = float(self.estimate)
        except (TypeError, ValueError):
            raise ValidationError("estimate must be finite") from None
        if not math.isfinite(estimate):
            raise ValidationError("estimate must be finite")
        interval = tuple(self.confidence_interval)
        if len(interval) != 2:
            raise ValidationError("confidence_interval must contain [low, high]")
        try:
            interval = (float(interval[0]), float(interval[1]))
        except (TypeError, ValueError):
            raise ValidationError("confidence_interval values must be finite") from None
        if not all(math.isfinite(value) for value in interval) or interval[0] > interval[1]:
            raise ValidationError("confidence_interval must be finite and ordered")
        if self.p_value is not None:
            try:
                p_value = float(self.p_value)
            except (TypeError, ValueError):
                raise ValidationError("p_value must be null or a number in [0, 1]") from None
            if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                raise ValidationError("p_value must be null or a number in [0, 1]")
            object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "confidence_interval", interval)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimRecord":
        if not isinstance(value, Mapping):
            raise ValidationError("claim record must be an object")
        return cls(
            claim_id=value.get("claim_id", ""),
            interpretation=value.get("interpretation", ""),
            analysis_design=value.get("analysis_design", ""),
            run_a=ComparisonRun.from_dict(value.get("run_a", {})),
            run_b=ComparisonRun.from_dict(value.get("run_b", {})),
            estimate=value.get("estimate"),
            uncertainty_method=value.get("uncertainty_method", ""),
            confidence_interval=value.get("confidence_interval", ()),
            p_value=value.get("p_value"),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_claim_record(
    value: ClaimRecord | Mapping[str, Any],
) -> dict[str, Any]:
    """Fail unless both runs support one like-for-like bounded comparison."""

    claim = value if isinstance(value, ClaimRecord) else ClaimRecord.from_dict(value)
    if claim.run_a.label == claim.run_b.label:
        raise ValidationError("comparison run labels must differ")
    compatible_fields = (
        "score_origin",
        "roster_sha256",
        "prompt_fingerprint",
        "decoding_fingerprint",
        "endpoint",
        "denominator",
        "missing_row_policy",
        "evidence_unit",
        "dependence_assignment_sha256",
    )
    mismatched = [
        field_name
        for field_name in compatible_fields
        if getattr(claim.run_a, field_name) != getattr(claim.run_b, field_name)
    ]
    if mismatched:
        raise ValidationError(
            "comparison runs are incompatible: " + ", ".join(mismatched)
        )
    payload = claim.to_dict()
    return {
        "artifact_type": "kgsft_claim_validation",
        "schema_version": 1,
        "status": "pass",
        "claim_id": claim.claim_id,
        "run_labels": [claim.run_a.label, claim.run_b.label],
        "comparison_contract": {
            field_name: getattr(claim.run_a, field_name)
            for field_name in compatible_fields
        },
        "estimate": claim.estimate,
        "confidence_interval": list(claim.confidence_interval),
        "uncertainty_method": claim.uncertainty_method,
        "p_value": claim.p_value,
        "claim_record_sha256": canonical_sha256(payload),
    }


def export_claim_validation(
    value: ClaimRecord | Mapping[str, Any],
    path: Path | str,
) -> dict[str, Any]:
    """Validate and atomically export a text-free claim-contract report."""

    report = validate_claim_record(value)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return report


def support_document_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "example_id",
    documents_field: str = "support_document_ids",
) -> dict[str, Any]:
    """Connect rows sharing support documents and report component sizes."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    doc_owner: dict[str, str] = {}
    row_ids: list[str] = []
    for row in rows:
        row_id = str(row.get(id_field) or "").strip()
        if not row_id or row_id in row_ids:
            raise ValueError("rows require unique non-empty IDs")
        row_ids.append(row_id)
        find(row_id)
        raw_documents = row.get(documents_field)
        if not isinstance(raw_documents, (list, tuple, set)):
            raise ValueError(f"row {row_id!r} has no document ID sequence")
        documents: set[str] = set()
        for value in raw_documents:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"row {row_id!r} contains a non-string or empty support "
                    "document ID"
                )
            documents.add(value.strip())
        if not documents:
            raise ValueError(f"row {row_id!r} has no support document IDs")
        for document in documents:
            if document in doc_owner:
                union(row_id, doc_owner[document])
            else:
                doc_owner[document] = row_id
    components: dict[str, list[str]] = {}
    for row_id in row_ids:
        components.setdefault(find(row_id), []).append(row_id)
    ordered = sorted((sorted(values) for values in components.values()), key=lambda x: (-len(x), x))
    sizes = [len(values) for values in ordered]
    n = len(row_ids)
    concentration = n * n / sum(size * size for size in sizes) if sizes else 0.0
    return {
        "rows": n,
        "support_documents": len(doc_owner),
        "component_count": len(ordered),
        "component_sizes": sizes,
        "largest_component": max(sizes, default=0),
        "size_concentration_diagnostic": round(concentration, 6),
        "components": ordered,
        "size_histogram": dict(sorted(Counter(sizes).items())),
    }
