"""Directed, provenance-preserving graph construction."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .schema import GraphBundle


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

