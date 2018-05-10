import tensorflow as tf, numpy as np
import argparse 
from utils import *
from blimpy import Waterfall
from sigpyproc.Readers import FilReader
import os
from time import time
from skimage import measure

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./models/molonglo.pb", type=str, help="Frozen model file to import")
parser.add_argument("--filterbank_dir", default="/data2/molonglo/", type=str, help="Directory containing filterbanks")
parser.add_argument("--test_flag", default=None, type=str, help="flag of file to test")
args = parser.parse_args()



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


    files = find_files(args.filterbank_dir, pattern='2018*.fil', flag=args.test_flag)
    print(len(files))
    files = sorted(files)[1:] #ignore primary beam
    print(files[:NBEAMS])
    readers = get_readers(files, NBEAMS)
    NT = readers[0].header['nsamples']
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
                print("Detections ",t0, beams_with_detection+1, scores[beams_with_detection])
