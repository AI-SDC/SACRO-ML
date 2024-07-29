"""Main entry point to sacroml."""

from __future__ import annotations

import argparse
import os

from sacroml.attacks.factory import run_attacks
from sacroml.config.attack import prompt_for_attack
from sacroml.config.target import prompt_for_target


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


if __name__ == "__main__":  # pragma:no cover
    main()
