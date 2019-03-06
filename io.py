import shutil
import os
import errno


def delete_dir_quiet(directory):
    try:
        shutil.rmtree(directory, ignore_errors=True)
        while os.path.exists(directory):  # check if it exists
            pass
    except OSError as exc:
        if exc.errno != errno.EEXIST and exc.errno != errno.ENOENT:
            raise exc
        pass


def make_dirs_quiet(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass
