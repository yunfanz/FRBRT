import tensorflow as tf, numpy as np
import argparse 
from utils import find_files
from blimpy import Waterfall
import os
from time import time
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs='+', default="./dump/frozen_model.pb", type=str, help="Frozen model file to import")
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
    if a is None:
        a = np.zeros((nbeams, tstep, nchan, 1), dtype=np.uint8)
    for i in range(nbeams):
        readers[i].read_data(t_start=t0, t_stop=t0+tstep)
        a[i, ..., 0] = readers[i].data.squeeze().astype('uint8')
    return a

def get_name(fname, t0, level=2):
    basename = '_'.join(fname.split('/')[-level:])
    return basename.split('.')[0]+'_'+str(t0)+'.npy'
if __name__ == '__main__':

    graph = load_graph(args.model)
    TSTEP = 1024 #window of time stamps
    NBEAMS = 352

    # We access the input and output nodes 
    is_training = graph.get_tensor_by_name('prefix/is_training:0')
    x = graph.get_tensor_by_name('prefix/input_placeholder:0')
    y = graph.get_tensor_by_name('prefix/output:0')
    
    outdir = "./false_positives/signa/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    files = find_files(args.filterbank_dir, pattern='201*.fil')
    print(len(files))
    print(sorted(files[:NBEAMS]))
    readers = get_readers(files, NBEAMS)
    NT = readers[0].n_ints_in_file
    dt = readers[0].header['tsamp']
    print('sampling time', dt)
    detection_stats = []
    with tf.Session(graph=graph) as sess:
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
                print'{} / {},  speed: {} times real time'.format(t0,NT, speed)
            scores = y_out[:,1].copy()
            detections = scores > 0.5
            for i, val in enumerate(detections):
                if not val: continue
                fname = get_name(sorted(files)[i], t0)
                print "Saving", outdir+fname
                np.save(outdir+fname, a[i].squeeze())

