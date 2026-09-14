"""Typed canonical schema and fail-closed validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


class ValidationError(ValueError):
    """Raised when an object violates a publication contract."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValidationError(f"{field_name} must be non-empty")
    return text


def _metadata(value: Any, field_name: str = "metadata") -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class SourceRef:
    """Reference to the source from which an object was derived."""

    source_id: str
    uri: str = ""
    revision: str = ""
    sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _required_text(self.source_id, "source_id"))
        object.__setattr__(self, "uri", str(self.uri or "").strip())
        object.__setattr__(self, "revision", str(self.revision or "").strip())
        digest = str(self.sha256 or "").strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValidationError("source sha256 must be empty or a 64-character hex digest")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRef":
        return cls(
            source_id=value.get("source_id", ""),
            uri=value.get("uri", ""),
            revision=value.get("revision", ""),
            sha256=value.get("sha256", ""),
            metadata=value.get("metadata", {}),
        )


def _source_refs(values: Iterable[SourceRef | Mapping[str, Any]], field_name: str) -> Tuple[SourceRef, ...]:
    result = tuple(
        value if isinstance(value, SourceRef) else SourceRef.from_dict(value)
        for value in values
    )
    if not result:
        raise ValidationError(f"{field_name} must contain at least one source reference")
    ids = [item.source_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValidationError(f"{field_name} contains duplicate source_id values")
    return result


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    role: str
    source_refs: Sequence[SourceRef] = field(default_factory=tuple)
    origin: str = "document"
    graph_record_ids: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _required_text(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "text", _required_text(self.text, "chunk text"))
        role = _required_text(self.role, "chunk role").lower()
        if role not in {"support", "distractor"}:
            raise ValidationError("chunk role must be support or distractor")
        object.__setattr__(self, "role", role)
        origin = _required_text(self.origin, "chunk origin").lower()
        if origin not in {"document", "graph", "retrieval", "synthetic"}:
            raise ValidationError("unsupported chunk origin")
        graph_ids = tuple(_required_text(value, "graph_record_id") for value in self.graph_record_ids)
        if origin == "graph" and not graph_ids:
            raise ValidationError("graph-origin chunks require graph_record_ids")
        if origin != "graph" and graph_ids:
            raise ValidationError("graph_record_ids are only valid for graph-origin chunks")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "graph_record_ids", graph_ids)
        object.__setattr__(self, "source_refs", _source_refs(self.source_refs, "chunk source_refs"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceChunk":
        return cls(
            chunk_id=value.get("chunk_id", ""),
            text=value.get("text", ""),
            role=value.get("role", ""),
            source_refs=value.get("source_refs", ()),
            origin=value.get("origin", "document"),
            graph_record_ids=value.get("graph_record_ids", ()),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class ContractExample:
    example_id: str
    question: str
    target: str
    support: Sequence[EvidenceChunk]
    distractors: Sequence[EvidenceChunk] = field(default_factory=tuple)
    task_type: str = "answer"
    source_refs: Sequence[SourceRef] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _required_text(self.example_id, "example_id"))
        object.__setattr__(self, "question", _required_text(self.question, "question"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))
        task_type = _required_text(self.task_type, "task_type").lower()
        if task_type not in {"answer", "refusal"}:
            raise ValidationError("task_type must be answer or refusal")
        support = tuple(
            value if isinstance(value, EvidenceChunk) else EvidenceChunk.from_dict(value)
            for value in self.support
        )
        distractors = tuple(
            value if isinstance(value, EvidenceChunk) else EvidenceChunk.from_dict(value)
            for value in self.distractors
        )
        if task_type == "answer" and not support:
            raise ValidationError("answer examples require at least one support chunk")
        if any(chunk.role != "support" for chunk in support):
            raise ValidationError("support contains a non-support chunk")
        if any(chunk.role != "distractor" for chunk in distractors):
            raise ValidationError("distractors contains a non-distractor chunk")
        ids = [chunk.chunk_id for chunk in (*support, *distractors)]
        if len(ids) != len(set(ids)):
            raise ValidationError("example contains duplicate chunk_id values")
        object.__setattr__(self, "task_type", task_type)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "distractors", distractors)
        object.__setattr__(self, "source_refs", _source_refs(self.source_refs, "example source_refs"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractExample":
        return cls(
            example_id=value.get("example_id", ""),
            question=value.get("question", ""),
            target=value.get("target", ""),
            support=value.get("support", ()),
            distractors=value.get("distractors", ()),
            task_type=value.get("task_type", "answer"),
            source_refs=value.get("source_refs", ()),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def invariant_payload(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "target": self.target,
            "task_type": self.task_type,
            "support": [asdict(chunk) for chunk in self.support],
            "source_refs": [asdict(ref) for ref in self.source_refs],
        }

    @property
    def invariant_sha256(self) -> str:
        return canonical_sha256(self.invariant_payload)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    source_refs: Sequence[SourceRef]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _required_text(self.node_id, "node_id"))
        object.__setattr__(self, "node_type", _required_text(self.node_type, "node_type"))
        object.__setattr__(self, "label", _required_text(self.label, "node label"))
        object.__setattr__(self, "source_refs", _source_refs(self.source_refs, "node source_refs"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphNode":
        return cls(
            node_id=value.get("node_id", ""),
            node_type=value.get("node_type", ""),
            label=value.get("label", ""),
            source_refs=value.get("source_refs", ()),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class GraphRelation:
    relation_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    source_refs: Sequence[SourceRef]
    directed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", _required_text(self.relation_id, "relation_id"))
        object.__setattr__(self, "source_node_id", _required_text(self.source_node_id, "source_node_id"))
        object.__setattr__(self, "target_node_id", _required_text(self.target_node_id, "target_node_id"))
        object.__setattr__(self, "relation_type", _required_text(self.relation_type, "relation_type"))
        if self.directed is not True:
            raise ValidationError("all graph relations must declare directed=true")
        object.__setattr__(self, "source_refs", _source_refs(self.source_refs, "relation source_refs"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphRelation":
        return cls(
            relation_id=value.get("relation_id", ""),
            source_node_id=value.get("source_node_id", ""),
            target_node_id=value.get("target_node_id", ""),
            relation_type=value.get("relation_type", ""),
            source_refs=value.get("source_refs", ()),
            directed=value.get("directed", True),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class GraphBundle:
    graph_id: str
    nodes: Sequence[GraphNode]
    relations: Sequence[GraphRelation]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _required_text(self.graph_id, "graph_id"))
        nodes = tuple(
            value if isinstance(value, GraphNode) else GraphNode.from_dict(value)
            for value in self.nodes
        )
        relations = tuple(
            value if isinstance(value, GraphRelation) else GraphRelation.from_dict(value)
            for value in self.relations
        )
        node_ids = [node.node_id for node in nodes]
        relation_ids = [relation.relation_id for relation in relations]
        if len(node_ids) != len(set(node_ids)):
            raise ValidationError("graph contains duplicate node_id values")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValidationError("graph contains duplicate relation_id values")
        known_nodes = set(node_ids)
        for relation in relations:
            if relation.source_node_id not in known_nodes or relation.target_node_id not in known_nodes:
                raise ValidationError(f"relation {relation.relation_id!r} references an unknown node")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphBundle":
        return cls(
            graph_id=value.get("graph_id", ""),
            nodes=value.get("nodes", ()),
            relations=value.get("relations", ()),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

