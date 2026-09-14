"""Command-line interface for public compilation and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters.rubq import import_rubq
from .compiler import compile_dataset
from .evaluation import ClaimRecord, export_claim_validation
from .graph import build_multidigraph, validate_graph_lineage
from .schema import ContractExample, GraphBundle


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_examples(path: Path) -> list[ContractExample]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"example line {line_number} is not an object")
            rows.append(ContractExample.from_dict(value))
    if not rows:
        raise ValueError("example file is empty")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kgsft", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate canonical graph and examples")
    validate.add_argument("--graph", type=Path, required=True)
    validate.add_argument("--examples", type=Path, required=True)

    compile_command = subparsers.add_parser("compile", help="Compile a loader-audited SFT bundle")
    compile_command.add_argument("--graph", type=Path, required=True)
    compile_command.add_argument("--examples", type=Path, required=True)
    compile_command.add_argument("--output-dir", type=Path, required=True)
    compile_command.add_argument("--max-tokens", type=int, default=256)
    compile_command.add_argument("--validation-fraction", type=float, default=0.1)
    compile_command.add_argument("--seed", type=int, default=228)

    rubq = subparsers.add_parser("import-rubq", help="Import an answer-bearing RuBQ subset")
    rubq.add_argument("--questions", type=Path, required=True)
    rubq.add_argument("--paragraphs", type=Path, required=True)
    rubq.add_argument("--output", type=Path, required=True)
    rubq.add_argument("--report", type=Path, required=True)
    rubq.add_argument("--limit", type=int, default=100)
    rubq.add_argument("--seed", type=int, default=228)

    claim = subparsers.add_parser(
        "validate-claim",
        help="Validate and export a comparison claim contract",
    )
    claim.add_argument("--claim", type=Path, required=True)
    claim.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-claim":
        claim = ClaimRecord.from_dict(_load_json(args.claim))
        if args.output:
            report = export_claim_validation(claim, args.output)
        else:
            from .evaluation import validate_claim_record

            report = validate_claim_record(claim)
    elif args.command in {"validate", "compile"}:
        graph = GraphBundle.from_dict(_load_json(args.graph))
        examples = _load_examples(args.examples)
        directed = build_multidigraph(graph)
        lineage = validate_graph_lineage(graph, examples)
        if args.command == "validate":
            report = {
                "status": "pass",
                "examples": len(examples),
                "nodes": directed.number_of_nodes(),
                "directed_relations": directed.number_of_edges(),
                **lineage,
            }
        else:
            report = compile_dataset(
                examples,
                graph,
                args.output_dir,
                max_tokens=args.max_tokens,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
    elif args.command == "import-rubq":
        examples, report = import_rubq(
            args.questions,
            args.paragraphs,
            limit=args.limit,
            seed=args.seed,
        )
        _write_jsonl(args.output, [example.to_dict() for example in examples])
        _write_json(args.report, report)
    else:  # pragma: no cover - argparse enforces the command set
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
