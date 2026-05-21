"""Main entry point to sacroml."""

from __future__ import annotations

import argparse
import os
import sys

from sacroml.attacks.factory import run_attacks
from sacroml.config.attack import prompt_for_attack
from sacroml.config.target import prompt_for_target
from sacroml.reporting import convert_report_file


def _run_convert_report(args: argparse.Namespace) -> int:
    """Convert a legacy report.json into the new report format.

    Returns
    -------
    int
        Process exit code (0 on success, 1 if the output is schema-invalid).
    """
    if not os.path.isfile(args.input):
        print(f"Input report not found: {args.input}")
        return 1

    result = convert_report_file(args.input, args.output, validate=not args.no_validate)
    print(f"Converted '{args.input}' -> '{args.output}'")
    print(result.summary(validated=not args.no_validate))
    return 0 if result.is_valid else 1


def main() -> None:
    """Load target and attack configurations and run attacks."""
    parser = argparse.ArgumentParser(
        description="CLI for running attacks and generating config files"
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run = subparsers.add_parser("run", help="Run attacks from YAML config files")

    run.add_argument("target_dir", type=str, help="Directory containing target.yaml")
    run.add_argument("attack_yaml", type=str, help="Attack YAML config")

    subparsers.add_parser("gen-target", help="Generate Target YAML config")
    subparsers.add_parser("gen-attack", help="Generate Attack YAML config")

    convert = subparsers.add_parser(
        "convert-report",
        help="Convert a legacy report.json into the new report format",
    )
    convert.add_argument("input", type=str, help="Path to the legacy report.json")
    convert.add_argument("output", type=str, help="Path to write the new report")
    convert.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip JSON schema validation of the converted report",
    )

    args = parser.parse_args()

    if args.cmd == "run":
        if os.path.isdir(args.target_dir) and os.path.isfile(args.attack_yaml):
            run_attacks(args.target_dir, args.attack_yaml)
        else:
            print("Both files must exist to run attacks.")
    elif args.cmd == "gen-target":
        prompt_for_target()
    elif args.cmd == "gen-attack":
        prompt_for_attack()
    elif args.cmd == "convert-report":
        sys.exit(_run_convert_report(args))


if __name__ == "__main__":  # pragma:no cover
    main()
