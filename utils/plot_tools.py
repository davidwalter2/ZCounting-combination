import datetime
import json
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import common, logging

logger = logging.child_logger(__name__)


def save_pdf_and_png(outdir, basename, fig=None):
    fname = f"{outdir}/{basename}.pdf"
    if fig:
        fig.savefig(fname, bbox_inches="tight")
        fig.savefig(fname.replace(".pdf", ".png"), bbox_inches="tight")
    else:
        plt.savefig(fname, bbox_inches="tight")
        plt.savefig(fname.replace(".pdf", ".png"), bbox_inches="tight")
    logger.info(f"Wrote file(s) {fname}(.png)")
    logger.info(f"Wrote file(s) {fname}(.png)")


def script_command_to_str(argv, parser_args):
    call_args = np.array(argv[1:], dtype=object)
    match_expr = "|".join(
        ["^-+([a-z]+[1-9]*-*)+"]
        + (
            []
            if not parser_args
            else [f"^-*{x.replace('_', '.')}" for x in vars(parser_args).keys()]
        )
    )
    if call_args.size != 0:
        flags = np.vectorize(lambda x: bool(re.match(match_expr, x)))(call_args)
        special_chars = np.vectorize(lambda x: not x.isalnum())(call_args)
        select = ~flags & special_chars
        if np.count_nonzero(select):
            call_args[select] = np.vectorize(lambda x: f"'{x}'")(call_args[select])
    return " ".join([argv[0], *call_args])


def write_index_and_log(
    outpath,
    logname,
    template_dir=f"{common.dir_data}/miscellaneous/",
    yield_tables=None,
    analysis_meta_info=None,
    args={},
    nround=2,
):
    indexname = "index.php"
    shutil.copyfile(f"{template_dir}/{indexname}", f"{outpath}/index.php")
    logname = f"{outpath}/{logname}.log"

    with open(logname, "w") as logf:
        meta_info = (
            "-" * 80
            + "\n"
            + f"Script called at {datetime.datetime.now()}\n"
            + f"The command was: {script_command_to_str(sys.argv, args)}\n"
            + "-" * 80
            + "\n"
        )
        logf.write(meta_info)

        if yield_tables is not None:
            if isinstance(yield_tables, dict):
                for k, v in yield_tables.items():
                    logf.write(f"Yield information for {k}\n")
                    logf.write("-" * 80 + "\n")
                    logf.write(str(v.round(nround)) + "\n\n")

                if (
                    "Unstacked processes" in yield_tables
                    and "Stacked processes" in yield_tables
                ):
                    if "Data" in yield_tables["Unstacked processes"]["Process"].values:
                        unstacked = yield_tables["Unstacked processes"]
                        data_yield = unstacked[unstacked["Process"] == "Data"][
                            "Yield"
                        ].iloc[0]
                        ratio = (
                            float(
                                yield_tables["Stacked processes"]["Yield"].sum()
                                / data_yield
                            )
                            * 100
                        )
                        logf.write(f"===> Sum unstacked to data is {ratio:.2f}%")
            elif isinstance(yield_tables, pd.DataFrame):
                logf.write("\t".join(yield_tables.columns) + "\n")
                for index, row in yield_tables.iterrows():
                    line = "\t".join(map(str, row.values))
                    logf.write(line + "\n")
            if isinstance(yield_tables, list):
                for df in yield_tables:
                    logf.write("-" * 80 + "\n")
                    if isinstance(df, pd.DataFrame):
                        logf.write("\t".join(df.columns) + "\n")
                        for index, row in df.iterrows():
                            line = "\t".join(map(str, row.values))
                            logf.write(line + "\n")

        if analysis_meta_info:
            for k, analysis_info in analysis_meta_info.items():
                logf.write("\n" + "-" * 80 + "\n")
                logf.write(f"Meta info from input file {k}\n")
                logf.write("\n" + "-" * 80 + "\n")
                logf.write(json.dumps(analysis_info, indent=5).replace("\\n", "\n"))
        logger.info(f"Writing file {logname}")


def save_plot(outdir, outfile, yield_tables=None, args=None):

    save_pdf_and_png(outdir, outfile)

    write_index_and_log(
        outdir,
        outfile,
        yield_tables=yield_tables,
        args=args,
    )
