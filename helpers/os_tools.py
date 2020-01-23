import os
import shutil
from glob import iglob
from pathlib import Path


recursive_indexes_path = '**/index.html'
def get_files_from_recursive_path(path):
    for f in iglob(path, recursive=True) :
        if os.path.isfile(f):
            yield str(Path(f).resolve())

def make_fresh_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)  # removes all the subdirectories!
        os.makedirs(dir)

def get_filename_from_path(path):
    return os.path.basename(path)
