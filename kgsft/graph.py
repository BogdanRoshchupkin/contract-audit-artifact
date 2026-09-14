"""Directed, provenance-preserving graph construction and lineage checks."""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx

from .schema import ContractExample, GraphBundle, SourceRef, ValidationError


def _source_refs_compatible(left: SourceRef, right: SourceRef) -> bool:
    """Match source identity and every version identifier supplied by either side."""

    if left.source_id != right.source_id:
        return False
    for field_name in ("revision", "sha256"):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if (left_value or right_value) and left_value != right_value:
            return False
    return True


def validate_graph_lineage(
    bundle: GraphBundle | dict[str, Any],
    examples: Sequence[ContractExample],
) -> dict[str, int]:
    """Validate graph-record references used by graph-origin evidence chunks."""

    checked = bundle if isinstance(bundle, GraphBundle) else GraphBundle.from_dict(bundle)
    records = {node.node_id: tuple(node.source_refs) for node in checked.nodes}
    records.update(
        {
            relation.relation_id: tuple(relation.source_refs)
            for relation in checked.relations
        }
    )
    referenced_record_ids: set[str] = set()
    graph_origin_chunks = 0
    for example in examples:
        for chunk in (*example.support, *example.distractors):
            if chunk.origin != "graph":
                continue
            graph_origin_chunks += 1
            for record_id in chunk.graph_record_ids:
                record_sources = records.get(record_id)
                if record_sources is None:
                    raise ValidationError(
                        f"graph-origin chunk {chunk.chunk_id!r} references unknown "
                        f"graph record {record_id!r}"
                    )
                if not any(
                    _source_refs_compatible(chunk_source, record_source)
                    for chunk_source in chunk.source_refs
                    for record_source in record_sources
                ):
                    raise ValidationError(
                        f"graph-origin chunk {chunk.chunk_id!r} and graph record "
                        f"{record_id!r} have no compatible source identity/version"
                    )
                referenced_record_ids.add(record_id)
    return {
        "graph_origin_chunks": graph_origin_chunks,
        "referenced_graph_records": len(referenced_record_ids),
    }


def build_multidigraph(bundle: GraphBundle | dict[str, Any]) -> nx.MultiDiGraph:
    """Build a typed MultiDiGraph after schema validation.

    Parallel relations are preserved and no reverse edge is inferred. Source
    references remain attached to every node and relation.
    """

    checked = bundle if isinstance(bundle, GraphBundle) else GraphBundle.from_dict(bundle)
    graph = nx.MultiDiGraph(graph_id=checked.graph_id, **dict(checked.metadata))
    for node in checked.nodes:
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            label=node.label,
            source_refs=tuple(node.source_refs),
            metadata=dict(node.metadata),
        )
    for relation in checked.relations:
        graph.add_edge(
            relation.source_node_id,
            relation.target_node_id,
            key=relation.relation_id,
            relation_id=relation.relation_id,
            relation_type=relation.relation_type,
            directed=True,
            source_refs=tuple(relation.source_refs),
            metadata=dict(relation.metadata),
        )
    return graph
