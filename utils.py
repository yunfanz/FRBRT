import os, fnmatch
import numpy as np
from sigpyproc.Readers import FilReader
import tensorflow as tf

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

def load_graph(frozen_graph_filename):
    """ Function to load frozen TensorFlow graph"""
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def get_readers(fil_files, nbeams=16):
    """Load blimpy Waterfall objects for filterbank file reading"""
    fils = []
    for f in sorted(fil_files)[:nbeams]:
        fils.append(FilReader(f))
    return fils

def read_input(readers, t0, a=None, tstep=1024, nchan=320):
    """Read a chunck of data from each beam
    output:
    array of shape (nbeam, tstep, nchan, 1)
    """
    nbeams = len(readers)
    u8 = (readers[0].header['nbits'] == 8)
    if a is None:
        a = np.zeros((nbeams, tstep, nchan, 1), dtype=np.uint8)
    for i in range(nbeams):
        a[i, ..., 0] = readers[i].readBlock(start=t0, nsamps=tstep).T
    return a