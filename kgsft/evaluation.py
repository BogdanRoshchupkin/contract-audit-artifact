"""Evaluation utilities that keep the dependence unit explicit."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


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
        documents = {str(value).strip() for value in raw_documents if str(value).strip()}
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

