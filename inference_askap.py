import tensorflow as tf, numpy as np
import argparse 
from utils import find_files
from blimpy import Waterfall
import os
from time import time
from skimage import measure
parser = argparse.ArgumentParser()
parser.add_argument("--in_memory", default=False, type=bool, help="Frozen model file to import")
parser.add_argument("--model", default="./models/molonglo.pb", type=str, help="Frozen model file to import")
parser.add_argument("--filterbank_dir", default="/data2/molonglo/", type=str, help="Directory containing filterbanks")
parser.add_argument("--out_dir", default="./false_pos/", type=str, help="Directory store false positives")
args = parser.parse_args()

def load_graph(frozen_graph_filename):
    """ Function to load frozen TensorFlow graph"""
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def get_readers(fil_files, nbeams=36, load_data=True):
    """Load blimpy Waterfall objects for filterbank file reading"""
    wfs = []
    if nbeams is None:
        nbeams = len(fil_files)
    for f in sorted(fil_files)[:nbeams]:
        if load_data:
            print "loading into memory", f
        wfs.append(Waterfall(f, load_data=load_data))
    return wfs

def read_input(readers, t0, a=None, tstep=1024, nchan=336, inmem=True, batch_size=10):
    """Read a chunck of data from each beam
    output:
    array of shape (nbeam, tstep, nchan, 1)
    """
    nbeams = len(readers) #actually present beams
    if t0+tstep >= readers[0].n_ints_in_file:
        t0 = readers[0].n_ints_in_file - tstep - 1
   # print t0, tstep, readers[0].n_ints_in_file
    u8 = (readers[0].header['nbits'] == 8)
    if a is None:
        a = np.zeros((nbeams*batch_size, tstep, nchan, 1), dtype=np.uint8)
    for i in range(nbeams):
        for b in range(batch_size):
            if inmem:
                a[i+b*nbeams, ..., 0] = readers[i].data.squeeze()[t0+b*tstep:t0+(b+1)*tstep].astype('uint8')
            else:
                readers[i].read_data(t_start=t0+tstep*b, t_stop=t0+tstep*(b+1))
                a[i+b*nbeams, ..., 0] = readers[i].data.squeeze().astype('uint8')
    return a

def get_name(fname, t0, level=4):
    basename = '_'.join(fname.split('/')[-level:])
    return '_'.join(basename.split('.')[:-1])+'_'+str(t0)+'.npy'

def filter_detection(detections, beam_ids, n=3, nbeam=36):
    """Function to filter out detections in more than n adjacent beams"""
    #detections = detections.reshape((-1,36))
    detections[-1,:] = 0 #mask out bad beams
    if detections.shape[0] != nbeam:
        raise ValueError('Missing Beams')
    detections = detections.T.reshape((-1,6,6))
    labs = [measure.label(antenna) for antenna in detections]
    for ant, lab in enumerate(labs):
        cluster, count = np.unique(lab, return_counts=True)
        for i, c in enumerate(cluster):
            if c == 0: continue
            if count[i] > n:
                detections[ant][lab==cluster] = False
    #detections = detections.reshape((-1, 36))
    return detections
    
if __name__ == '__main__':

    graph = load_graph(args.model)
    TSTEP = 1024 #window of time stamps

    NBEAMS = None
    # We access the input and output nodes 
    is_training = graph.get_tensor_by_name('prefix/is_training:0')
    x = graph.get_tensor_by_name('prefix/input_placeholder:0')
    y = graph.get_tensor_by_name('prefix/output:0')

    basedir = os.path.basename(os.path.normpath(args.filterbank_dir))
    if basedir.startswith("SB"):
        observations = sorted([os.path.join(args.filterbank_dir, sdir) for sdir in os.listdir(args.filterbank_dir)])
    elif basedir.startswith("20"):
        observations = [args.filterbank_dir]
    else:
        print basedir, args.filterbank_dir
        raise ValueError("filterbank_dir must be observation or scheduling block")
    print "processing {}  observations".format(len(observations))
    outdir = os.path.join(args.out_dir, basedir)
    if not outdir.endswith('/'):
        outdir = outdir+'/'
    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:

        for ob in observations:
            antennas = sorted([os.path.join(ob, sdir) for sdir in os.listdir(ob)])
            for ant in antennas:
                files = sorted(find_files(ant, pattern='201*.fil'))
                if len(files) == 0:
                    print "no files in", ant
                    continue
                beam_ids = [int(fn.split('.')[-2]) for fn in files]
                nbeams_present = len(beam_ids)
                readers = get_readers(files, nbeams_present, load_data=args.in_memory)
                if max(beam_ids) > 35:
                    NBEAMS = 72
                else:
                    NBEAMS = 36
                print "{} / {} beams present".format(nbeams_present, NBEAMS)
                NT = min([reader.n_ints_in_file for reader in readers])
                dt = readers[0].header['tsamp']
                print('sampling time', dt)
            
                t0 = 0
                a = None
                batch_size = 10
                while t0 < NT:
                    start_read = time()
                    while t0 + TSTEP*batch_size > NT and batch_size>1:
                        batch_size /= 2
                        print "Adjusting batch_size", batch_size
                        
                    a = read_input(readers, t0, a=a, batch_size=batch_size, inmem=args.in_memory)
         
                    start = time()
                    y_out = sess.run(y, feed_dict={ x: a, is_training:False })
                    duration = time() - start

                    speed = dt*TSTEP/duration
                    read_time = start - start_read
                    print'{} / {},  speed: {} times real time, reading time'.format(t0,NT, speed, read_time) #print(y_out.shape)
                    
                    scores = y_out[:,1].copy()
                    scores = scores.reshape((nbeams_present, -1))
                    detections = scores > 0.5
                    print "False positive rate {}".format(float(np.sum(detections))/detections.size)
                    print detections.shape

                    # Save false positives
                    for i, val in enumerate(detections): #loop over beam
                       for j, jval in enumerate(val): #loop over time stamps
                           if not jval: continue
                           fname = get_name(sorted(files)[i], t0+j*TSTEP, level=4)
                           print "Saving", outdir+fname
                           np.save(outdir+fname, a[i+j*nbeams_present].squeeze())
                    t0 += TSTEP*batch_size

                    #report detections
                    detections = filter_detection(detections, beam_ids, n=3) 
                    ndetections = np.sum(detections, axis=(1,2))
                    for j, nd in enumerate(ndetections):
                        if nd > 0:
                            print detections[j].astype(np.int8)
                            np.set_printoptions(precision=2)
                            print scores[:,j].reshape((6,6))
                            #beams_with_detection = np.asarray([ind for ind, val in enumerate(detections) if val])
                            print("Detections ",t0+j*TSTEP)
