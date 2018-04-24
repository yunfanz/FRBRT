import tensorflow as tf, numpy as np
import argparse 
from utils import find_files
from blimpy import Waterfall
import os
from time import time
from skimage import measure

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./models/molonglo.pb", type=str, help="Frozen model file to import")
parser.add_argument("--filterbank_dir", default="/data2/molonglo/", type=str, help="Directory containing filterbanks")
args = parser.parse_args()

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
    wfs = []
    for f in sorted(fil_files)[:nbeams]:
        wfs.append(Waterfall(f, load_data=False))
    return wfs

def read_input(readers, t0, a=None, tstep=1024, nchan=320):
    """Read a chunck of data from each beam
    output:
    array of shape (nbeam, tstep, nchan, 1)
    """
    nbeams = len(readers)
    u8 = (readers[0].header['nbits'] == 8)
    if a is None:
        a = np.zeros((nbeams, tstep, nchan, 1), dtype=np.float32)
    for i in range(nbeams):
        readers[i].read_data(t_start=t0, t_stop=t0+tstep)
        a[i, ..., 0] = readers[i].data.squeeze().astype('uint8').astype(np.float32)
    return a

def filter_detection(detections, n=3):
    """Function to filter out detections in more than n adjacent beams"""
    lab = measure.label(detections)
    cluster, count = np.unique(lab, return_counts=True)
    for i, c in enumerate(cluster):
        if c == 0: continue
        if count[i] > n:
            detections[lab==cluster] = False
    return detections
    
if __name__ == '__main__':

    graph = load_graph(args.model)
    TSTEP = 1024 #window of time stamps
    NBEAMS = 351

    # We access the input and output nodes 
    is_training = graph.get_tensor_by_name('prefix/is_training:0')
    x = graph.get_tensor_by_name('prefix/input_placeholder:0')
    y = graph.get_tensor_by_name('prefix/output:0')


    files = find_files(args.filterbank_dir, pattern='2018*.fil')
    print(len(files))
    files = sorted(files)[1:] #ignore primary beam
    print(files[:NBEAMS])
    readers = get_readers(files, NBEAMS)
    NT = readers[0].n_ints_in_file
    dt = readers[0].header['tsamp']
    print('sampling time', dt)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        t0 = 0
        a = None
        while t0 + TSTEP < NT:
            a = read_input(readers, t0, a=a)
            t0 += TSTEP
            #print(a.shape, a.dtype)
            #import IPython; IPython.embed()
            start = time()
            y_out = sess.run(y, feed_dict={ x: a, is_training:False })
            duration = time() - start
            if t0 % 10240 == 0:
                speed = dt*TSTEP/duration
                print'{} / {},  speed: {} times real time'.format(t0,NT, speed) #print(y_out.shape)
            scores = y_out[:,1].copy()
            detections = scores > 0.5
            detections = filter_detection(detections, n=3) 
            ndetections = np.sum(detections)
            if ndetections > 0 and ndetections<5:
                beams_with_detection = np.asarray([ind for ind, val in enumerate(detections) if val])
                print("Detections ",t0, beams_with_detection, scores[beams_with_detection])
