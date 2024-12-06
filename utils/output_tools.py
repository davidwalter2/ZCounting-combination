import glob
import os
import shutil
import subprocess
import tempfile

from utils import logging

logger = logging.child_logger(__name__)


def is_eosuser_path(path):
    if not path:
        return False
    path = os.path.realpath(path)
    return path.startswith("/eos/user") or path.startswith("/eos/home-")


def split_eos_path(path):

    path = os.path.realpath(path)
    if not is_eosuser_path(path):
        raise ValueError(f"Expected a path on /eos/user, found {path}!")

    splitpath = [x for x in path.split("/") if x]
    # Can be /eos/user/<letter>/<username> or <letter-username>
    if "home-" in splitpath[1]:
        eospath = "/".join(["/eos/user", splitpath[1].split("-")[-1], splitpath[2]])
        basepath = "/".join(splitpath[3:])
    else:
        eospath = "/".join(splitpath[:4])
        basepath = "/".join(splitpath[4:])

    if path[0] == "/":
        eospath = "/" + eospath

    return eospath, basepath


def make_plot_dir(outpath, outfolder=None, eoscp=False, allowCreateLocalFolder=True):
    if eoscp and is_eosuser_path(outpath):
        # Create a unique temporary directory
        unique_temp_dir = tempfile.mkdtemp()
        outpath = os.path.join(unique_temp_dir, split_eos_path(outpath)[1])
        if not os.path.isdir(outpath):
            logger.info(f"Making temporary directory {outpath}")
            os.makedirs(outpath)

    full_outpath = outpath
    if outfolder:
        full_outpath = os.path.join(outpath, outfolder)
    if not full_outpath.endswith("/"):
        full_outpath += "/"
    if outpath and not os.path.isdir(outpath):
        # instead of raising, create folder to deal with cases where nested folders are created during code execution
        # (this would happen when outpath is already a path to a local subfolder not created in the very beginning)
        if allowCreateLocalFolder:
            logger.debug(f"Creating new directory {outpath}")
            os.makedirs(outpath)
        else:
            raise IOError(
                f"The path {outpath} doesn't not exist. You should create it (and possibly link it to your web area)"
            )

    if full_outpath and not os.path.isdir(full_outpath):
        try:
            os.makedirs(full_outpath)
            logger.info(f"Creating folder {full_outpath}")
        except FileExistsError as e:
            logger.warning(e)

    return full_outpath


def copy_to_eos(tmpFolder, outpath, outfolder=None, deleteFullTmp=False):
    eospath, outpath = split_eos_path(outpath)
    fullpath = outpath
    if outfolder:
        fullpath = os.path.join(outpath, outfolder)
    logger.info(f"Copying {tmpFolder} to {fullpath}")

    for f in glob.glob(tmpFolder + "/*"):
        if not (os.path.isfile(f) or os.path.isdir(f)):
            continue

        outPathForCopy = "/".join(
            ["root://eosuser.cern.ch", eospath, f.replace(tmpFolder, f"{fullpath}/")]
        )
        if os.path.isdir(f):
            # remove last folder to do "xrdcp -fr /path/to/folder/ root://eosuser.cern.ch//eos/cms/path/to/"
            # in this way one can copy the whole subfolder through xrdcp without first creating the structure
            outPathForCopy = os.path.dirname(outPathForCopy.rstrip("/"))
        command = ["xrdcp", "-fr", f, outPathForCopy]

        logger.debug(f"Executing {' '.join(command)}")
        if subprocess.call(command):
            raise IOError(
                "Failed to copy the files to eos! Perhaps you are missing a kerberos ticket and need to run kinit <user>@CERN.CH?"
                " from lxplus you can run without eoscp and take your luck with the mount."
            )

    shutil.rmtree(tmpFolder.replace(fullpath, ""))
