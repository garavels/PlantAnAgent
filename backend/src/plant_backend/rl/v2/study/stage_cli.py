"""Stage parent, isolated child and detached verification entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registered cyclic-market study custody"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "child", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--training-root", required=True, type=Path)
        command.add_argument(
            "--stage", required=True, choices=("development", "heldout")
        )
        if name == "run":
            command.add_argument("--confirm-operator-assertions", action="store_true")
        elif name == "child":
            command.add_argument("--order", type=int, required=True)
        else:
            command.add_argument("--publish", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "child":
            from .stage_launch import run_child

            run_child(arguments.training_root, arguments.stage, arguments.order)
        elif arguments.command == "run":
            from .stage_campaign import run_stage

            result = run_stage(
                arguments.training_root,
                arguments.stage,
                confirm_operator_assertions=arguments.confirm_operator_assertions,
            )
            print(json.dumps(result, sort_keys=True))
        else:
            from . import stage_documents as documents
            from .stage_verifier import verify_stage, verdict

            verified = verify_stage(arguments.training_root, arguments.stage)
            result = verdict(verified.report, verified.archive)
            verified.recheck()
            if arguments.publish:
                documents.publish(
                    Path(verified.context.registry["verification_path"]), result
                )
            else:
                print(json.dumps(result, sort_keys=True))
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        print(
            "Registered study custody failed; evidence and consumed attempts are retained.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
