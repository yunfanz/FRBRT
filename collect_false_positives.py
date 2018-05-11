import tensorflow as tf, numpy as np
import argparse 
from utils import *
from sigpyproc.Readers import FilReader
import os
from time import time
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./dump/frozen_model.pb", type=str, help="Frozen model file to import")
parser.add_argument("--offset", default=0, type=int, help="Initial offset from start of file")
parser.add_argument("--filterbank_dir", default="/data2/molonglo/", type=str, help="Directory containing filterbanks")
parser.add_argument("--test_flag", default=None, type=str, help="flag of file to test")
args = parser.parse_args()



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
    
    outdir = "./false_positives/" + os.path.basename(args.model).split('.')[0]+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    files = find_files(args.filterbank_dir, pattern='201*.fil', flag=args.test_flag)
    print(len(files))
    print(sorted(files[:NBEAMS]))
    readers = get_readers(files, NBEAMS)
    NT = readers[0].header['nsamples']
    dt = readers[0].header['tsamp']
    print('sampling time', dt)
    detection_stats = []
    with tf.Session(graph=graph) as sess:
        t0 = args.offset
        a = None
        while t0 + TSTEP < NT:
            a = read_input(readers, t0, a=a)
            t0 += TSTEP
            #print(a.shape, a.dtype)
            #import IPython; IPython.embed()
            start = time()
            y_out = sess.run(y, feed_dict={ x: a, is_training:False })
            duration = time() - start
            if (t0-args.offset)/TSTEP % 200 == 0:
                speed = dt*TSTEP/duration
                print'{} / {},  speed: {} times real time'.format(t0,NT, speed)
            scores = y_out[:,1].copy()
            detections = scores > 0.5
            for i, val in enumerate(detections):
                if not val: continue
                fname = get_name(sorted(files)[i], t0)
                print "Saving", outdir+fname
                np.save(outdir+fname, a[i].squeeze())

