import argparse
import os


def base_parser():
    parser = argparse.ArgumentParser()
    return parser


def plot_parser():
    parser = base_parser()
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("~/www/WMassAnalysis"),
        help="Base path for output",
    )
    parser.add_argument(
        "--cmsDecor",
        default="Preliminary",
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
