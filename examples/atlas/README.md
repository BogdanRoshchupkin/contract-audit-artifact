# Building the Atlas graph input

This fixture shows how to turn source facts into the canonical directed graph
and connect graph-derived context back to those facts. The service and runbooks
are synthetic.

`kgsft-contracts` does not extract an ontology or entities from raw documents.
Use manual curation, deterministic rules, an LLM extractor, or an existing graph
database first, then export the result in the format below.

## 1. Register source documents

Assign a stable identifier and revision to each source before extracting graph
records. A production export should also include a content SHA-256 digest when
the source can be hashed.

| `source_id` | Revision | Source fact |
| --- | --- | --- |
| `atlas-runbook-v2` | `2` | Atlas v2 requires OAuth2 and supersedes v1 |
| `atlas-runbook-v1` | `1` | Atlas v1 requires an API key |

The source text does not need to be copied into `graph.json`; `source_refs`
provide the durable link to the governed document inventory.

## 2. Export nodes

Create a stable `node_id`, a domain-specific `node_type`, a human-readable
`label`, and at least one `source_refs` entry for every accepted node:

```json
{
  "node_id": "atlas-v2",
  "node_type": "service_version",
  "label": "Service Atlas version 2",
  "source_refs": [
    {"source_id": "atlas-runbook-v2", "revision": "2"}
  ]
}
```

Node and relation IDs must be globally unique within a graph bundle.

## 3. Export directed relations

Write each directional fact from subject to object and preserve its evidence:

```json
{
  "relation_id": "rel-v2-requires-oauth2",
  "source_node_id": "atlas-v2",
  "target_node_id": "oauth2",
  "relation_type": "REQUIRES",
  "directed": true,
  "source_refs": [
    {"source_id": "atlas-runbook-v2", "revision": "2"}
  ]
}
```

Do not add the reverse edge unless the source supports a separate reverse
relation. Parallel relations between the same nodes are allowed because each
has its own `relation_id` and type.

The complete bundle is [`graph.json`](graph.json).

## 4. Link graph-derived QA chunks

A graph-origin support or distractor records which graph facts produced it.
Its `source_refs` must overlap the provenance of every referenced graph record:

```json
{
  "chunk_id": "atlas-v1-auth",
  "text": "Service Atlas version 1 authenticates requests with an API key.",
  "role": "distractor",
  "origin": "graph",
  "graph_record_ids": ["rel-v1-requires-api-key"],
  "source_refs": [
    {"source_id": "atlas-runbook-v1", "revision": "1"}
  ]
}
```

This chunk appears inside [`examples.jsonl`](examples.jsonl). Document-derived
chunks use `origin: "document"` and must not contain `graph_record_ids`.

## 5. Validate before compiling

From the repository root:

```bash
uv run --frozen kgsft validate \
  --graph examples/atlas/graph.json \
  --examples examples/atlas/examples.jsonl
```

A passing report confirms schema validity, directed graph construction, and
cross-file graph lineage. Then compile the loader-audited SFT bundle:

```bash
uv run --frozen kgsft compile \
  --graph examples/atlas/graph.json \
  --examples examples/atlas/examples.jsonl \
  --output-dir /tmp/kgsft-atlas \
  --max-tokens 36 \
  --validation-fraction 0.1 \
  --seed 228
```

For a new corpus, keep these two files as templates, replace the synthetic
records, and implement the production loader adapter described in the root
[`README.md`](../../README.md).
