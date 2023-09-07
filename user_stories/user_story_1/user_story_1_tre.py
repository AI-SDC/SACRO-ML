"""
User story 1 as TRE.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141

Running
-------

Invoke this code from the root AI-SDC folder with
python -m example_notebooks.user_stories.user_story_1.user_story_1_tre
"""
import argparse
import os
import yaml

from aisdc.attacks.attack_report_formatter import (  # pylint: disable=import-error
    GenerateTextReport,
)

def generate_report(directory, attack_results, target, outfile):
    """Generate report based on target model."""

    print()
    print("Acting as TRE...")
    print()

    t = GenerateTextReport()

    attack_pathname = os.path.join(directory, attack_results)
    t.process_attack_target_json(
        attack_pathname, target_filename=attack_pathname
    )

    out_pathname = os.path.join(directory, outfile)
    t.export_to_file(output_filename=out_pathname, move_files=True)

    print("Results written to " + out_pathname)

def main():
    """Main method to parse arguments and then invoke report generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a risk report after request_release() has been called by researcher"
        )
    )

    parser.add_argument(
        "--config_file",
        type=str,
        action="store",
        dest="config_file",
        required=False,
        default="default_config.yaml",
        help = (
            "Name of yaml configuration file"
        )
    )

    args = parser.parse_args()

    try:
        with open(args.config_file, encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.loader.SafeLoader)
    except AttributeError as error:  # pragma:no cover
        print("Invalid command. Try --help to get more details" f"error message is {error}")

    generate_report(
        config['training_artefacts_dir'],
        config['attack_results'],
        config['target_results'],
        config['outfile'],
    )

if __name__ == "__main__":  # pragma:no cover
    main()
