import argparse
import os


def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger", action="store_true", help="Do not use logging with colors"
    )
    return parser


def plot_parser():
    parser = base_parser()
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("~/www/ZCounting/"),
        help="Base path for output",
    )
    parser.add_argument(
        "-f", "--outfolder", type=str, default="./test", help="Subfolder for output"
    )
    parser.add_argument(
        "--cmsDecor",
        default="Work in progress",
        nargs="?",
        type=str,
        choices=[
            None,
            " ",
            "Preliminary",
            "Work in progress",
            "Internal",
            "Supplementary",
        ],
        help="CMS label",
    )
    return parser
