import glob
import os

def delete_files_from_dir(outd):
    print('Deleting files from: %s' % outd)
    files = glob.glob(os.path.join(outd, '*'))
    for file in files:
        os.remove(file)