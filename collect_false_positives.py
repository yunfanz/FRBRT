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

def load_graph(frozen_graph_filename, name="prefix"):
    """ Function to load frozen TensorFlow graph"""
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=name)
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

    graphs = [load_graph(model) for model in args.models]
    TSTEP = 1024 #window of time stamps
    NBEAMS = 352

    # We access the input and output nodes 
    is_training = [graph.get_tensor_by_name('prefix/is_training:0') for graph in graphs]
    x = [graph.get_tensor_by_name('prefix/input_placeholder:0') for graph in graphs]
    y = [graph.get_tensor_by_name('prefix/output:0') for graph in graphs]
    
    outdirs = ["./false_positives/" + os.path.basename(model).split('.')[0] for model in args.models]
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
    sessions = [tf.Session(graph=graph) for graph in graphs]
    t0 = 0
    a = None
    while t0 + TSTEP < NT:
        a = read_input(readers, t0, a=a)
        t0 += TSTEP
        #print(a.shape, a.dtype)
        #import IPython; IPython.embed()
        start = time()
        y_outs = [sess.run(y, feed_dict={ x[i]: a, is_training[i]:False }) for i, sess in enumerate(sessions)]
        duration = time() - start
        if t0 % 10240 == 0:
            speed = dt*TSTEP/duration
            print'{} / {},  speed: {} times real time'.format(t0,NT, speed)
        scores = np.asarray([y_out[:,1].copy() for y_out in y_outs])
        detections = scores > 0.5

        for i in range(detetions.shape[0]):
            for j in range(detections.shape[1]):
                if not detections[i,j]: continue
                fname = get_name(sorted(files)[j], t0)
                print "Saving", outdirs[i]+fname
                np.save(outdirs[i]+fname, a[j].squeeze())

