"""
User story 7 as TRE.

Details can be found here:
https://github.com/AI-SDC/AI-SDC/issues/141
"""

import argparse
import pickle


def generate_report(directory, target_model_filepath):
    """Main method to parse arguments and then invoke report generation."""
    print()
    print("Acting as TRE...")
    print(
        "(when researcher has provided NO INSTRUCTIONS on how to recreate the dataset)"
    )
    print()

    filename = directory + target_model_filepath
    print("Reading target model from " + filename)
    _ = pickle.load(open(filename, "rb"))

    print("Attacks cannot be run since the original dataset cannot be recreated")
    print("AISDC cannot provide any help to TRE")


def main():
    """Main method to parse arguments and then invoke report generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a risk report after request_release() has been called by researcher"
        )
    )

    parser.add_argument(
        "--training_artefacts_directory",
        type=str,
        action="store",
        dest="training_artefacts_directory",
        required=False,
        default="training_artefacts/",
        help=(
            "Folder containing training artefacts produced by researcher. Default = %(default)s."
        ),
    )

    parser.add_argument(
        "--target_model",
        type=str,
        action="store",
        dest="target_model",
        required=False,
        default="/model.pkl",
        help=("Filename of target model. Default = %(default)s."),
    )

    args = parser.parse_args()

    try:
        generate_report(args.training_artefacts_directory, args.target_model)
    except AttributeError as e:  # pragma:no cover
        print("Invalid command. Try --help to get more details" f"error mesge is {e}")


if __name__ == "__main__":  # pragma:no cover
    main()
