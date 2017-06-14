import os
import shutil


def empty_dir(path):
    """
    Delete everything in a directory
    :param path: string
    :return: nothing
    """
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            os.unlink(fpath)
        elif os.path.isdir(fpath):
            shutil.rmtree(fpath)


def create_dir(path):
    """
    Creates a directory if it does not exist
    :param path: string
    :return: nothing
    """
    if not os.path.exists(path):
        os.makedirs(path)