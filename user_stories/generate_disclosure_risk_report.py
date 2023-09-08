import argparse
import yaml

from user_story_1 import user_story_1_tre
from user_story_2 import user_story_2_tre
from user_story_3 import user_story_3_tre
from user_story_7 import user_story_7_tre

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run user stories code from a config file"
        )
    )

    parser.add_argument(
        "--config_file",
        type=str,
        action="store",
        dest="config_file",
        required=False,
        default="default_config.yaml",
        help=("Name of yaml configuration file"),
    )

    args = parser.parse_args()

    try:
        with open(args.config_file, encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.loader.SafeLoader)
    except AttributeError as error:  # pragma:no cover
        print(
            "Invalid command. Try --help to get more details"
            f"error message is {error}"
        )

    user_story = config['user_story']
    if user_story == 1:
        user_story_1_tre.run_user_story(config)
    elif user_story == 2:
        user_story_2_tre.run_user_story(config)
    elif user_story == 3:
        user_story_3_tre.run_user_story(config)
    elif user_story == 7:
        user_story_7_tre.run_user_story(config)
    else:
        raise NotImplementedError('User story %s has not been implemented' % (user_story))
