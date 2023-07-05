""" 
User story 1 as TRE
"""
import argparse

from aisdc.attacks.attack_report_formatter import GenerateTextReport

def generate_report(directory,attack_results,target,outfile):
    """
    Generate report based on target model
    """

    print()
    print("Acting as TRE...")
    print()

    t = GenerateTextReport()
    t.process_attack_target_json(directory+attack_results,target_filename=directory+target)

    t.export_to_file(output_filename=outfile)

    print("Results written to "+outfile)


def main():
    """main method to parse arguments and then invoke report generstion"""
    parser = argparse.ArgumentParser(
        description=("Generate a risk report after request_release() has been called by researcher")
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
        "--attack_results",
        type=str,
        action="store",
        dest="attack_results",
        required=False,
        default="attack_results.json",
        help=(
            "Filename for the saved JSON attack output. Default = %(default)s."
        ),
    )

    parser.add_argument(
        "--target_results",
        type=str,
        action="store",
        dest="target_results",
        required=False,
        default="target.json",
        help=(
            "Filename for the saved JSON model output. Default = %(default)s."
        ),
    )

    parser.add_argument(
        "--outfile",
        type=str,
        action="store",
        dest="outfile",
        required=False,
        default="summary.txt",
        help=(
            "Filename for the final results to be written to. Default = %(default)s."
        ),
    )

    args = parser.parse_args()

    try:
        generate_report(args.training_artefacts_directory,
            args.attack_results,
            args.target_results,
            args.outfile)
    except AttributeError as e:  # pragma:no cover
        print("Invalid command. Try --help to get more details"
              f"error mesge is {e}")

if __name__ == "__main__":  # pragma:no cover
    main()
