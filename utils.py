import os, fnmatch
import numpy as np
def find_files(directory, pattern='*.png', sortby="shuffle"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    if sortby == 'auto':
        files = np.sort(files)
    elif sortby == 'shuffle':
        np.random.shuffle(files)
    return files
