import os, fnmatch
import numpy as np
def find_files(directory, pattern='*.png', sortby="shuffle", flag=None):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            path = os.path.join(root, filename)
            if flag is not None and not os.path.basename(path).startswith(flag):
                #if a flagged file exists, read that file instead of the original, for injected 
                flagged = os.path.dirname(path)+'/'+flag+os.path.basename(path)
                if os.path.exists(flagged):
                    files.append(flagged)
                    print flagged
                    continue

            files.append(path)

    if sortby == 'auto':
        files = np.sort(files)
    elif sortby == 'shuffle':
        np.random.shuffle(files)
    return files
