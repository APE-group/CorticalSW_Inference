import numpy as np
import argparse
import neo
import quantities as pq
import matplotlib.pyplot as plt
import json
import os
import sys

from utils.parse import parse_string2dict, none_or_float, none_or_int, none_or_str
from utils.neo import imagesequences_to_analogsignals, flip_image, rotate_image, time_slice
from utils.io import load_input, write_output, load_neo, write_neo

#from utils import parse_string2dict, ImageSequence2AnalogSignal
#from utils import none_or_float, none_or_str, write_neo, time_slice
#from utils import flip_image, rotate_image
import scipy.io as sio
#import imageio


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data directory")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float,
                     default=None, help="sampling rate in Hz")
    CLI.add_argument("--spatial_scale", nargs='?', type=float, required=True,
                     help="distance between electrodes or pixels in mm")
    CLI.add_argument("--data_name", nargs='?', type=str, default='None',
                     help="chosen name of the dataset")
    CLI.add_argument("--annotations", nargs='+', type=none_or_str, default=None,
                     help="metadata of the dataset")
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str,
                     default=None, help="channel-wise metadata")
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str, default=None,
                     help="additional optional arguments")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=None,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=none_or_float, default=None,
                     help="stop time in seconds")
    CLI.add_argument("--trial",  nargs='?', type=none_or_float, default=None,
                     help="stop time in seconds")
    CLI.add_argument("--orientation_top", nargs='?', type=str, required=True,
                     help="upward orientation of the recorded cortical region")
    CLI.add_argument("--orientation_right", nargs='?', type=str, required=True,
                     help="right-facing orientation of the recorded cortical region")
    CLI.add_argument("--hemodynamics_correction", nargs='+', type=bool, default=False,
                     help="whether emodynamics correction is applicable")
    CLI.add_argument("--path_to_reflectance_data", nargs='+', type=str, default=None,
                     help="path to reflectance data")

    args = CLI.parse_args()

    # Load optical data

    mat_fname = args.data
    Signal = np.array(sio.loadmat(mat_fname)['NuE']) #array of 1000x 2500 (50x50) images
    y_pos_sel = np.array(sio.loadmat(mat_fname)['y_pos_sel']).T #array of 1000x 2500 (50x50) images
    x_pos_sel = np.array(sio.loadmat(mat_fname)['x_pos_sel']).T #array of 1000x 2500 (50x50) images
    print('signal', np.shape(Signal))
    print('len signal', len(Signal))
    print('y', np.shape(y_pos_sel))
    # 50 numero di pixel per lato. DEVI AGGIUNGERLO NEL CONFIG E SNAKEMAKE
    Piexel = x_pos_sel + y_pos_sel*50
    image_seq = np.empty([len(Signal), 50, 50])
    image_seq[:] = np.nan
    print('shape image_seq', np.shape(image_seq))
    '''
    path =  args.output + "/temp_images/"
    # checking path
    if not os.path.exists(path):
        os.makedirs(path)
    '''
        
    for t in range(len(Signal)):
        # per ogni tempo 
        for px in range(len(y_pos_sel)):
            # per ogni pixel
            image_seq[t][x_pos_sel[px], y_pos_sel[px]] = Signal[t][px]

        #imageio.imwrite(path + str(t)  + '.tif', image_seq[t])

    #plt.figure()
    #plt.imshow(image_seq[0])
    #plt.savefig('time' + str(ciao) + '.png')


    imageSequences = neo.ImageSequence(image_seq,
                       sampling_rate=args.sampling_rate * pq.Hz,
                       spatial_scale=args.spatial_scale * pq.mm,
                       units='dimensionless')
    '''

    num = 0
    for img in len(Signal[0]):
        image_seq[img][x_pos_sel][y_pos_sel] = 
        scipy.misc.imsave(path + str(num)  + '.tif', img)
        num +=1

        
    print('NuE', Signal)
    print(' shape NuE', np.shape(Signal))
    print('shape pos', np.shape(x_pos_sel))

    '''    
    '''

    io = neo.io.tiffio.TiffIO(directory_path=args.data,
                              sampling_rate=args.sampling_rate*pq.Hz,
                              spatial_scale=args.spatial_scale*pq.mm,
                              units='dimensionless')
    '''
    # loading the data flips the images vertically!

    #block = io.read_block()
    block = neo.Block()
    seg = neo.Segment(name='segment 0', index=0)
    block.segments.append(seg)
    print('vlock', block)
    print('seg', block.segments[0])

    block.segments[0].imagesequences.append(imageSequences)

    # change data orientation to be top=ventral, right=lateral
    imgseq = block.segments[0].imagesequences[0]
    imgseq = flip_image(imgseq, axis=-2)
    imgseq = rotate_image(imgseq, rotation=-90)
    block.segments[0].imagesequences[0] = imgseq
    print('shape imageseq', np.shape(imgseq))

    # Transform into analogsignals
    block.segments[0].analogsignals = []
    block = imagesequences_to_analogsignals(block)
    print('block', block)
    block.segments[0].analogsignals[0] = time_slice(
                block.segments[0].analogsignals[0], args.t_start, args.t_stop)

    if args.annotations is not None:
        block.segments[0].analogsignals[0].annotations.\
                                    update(parse_string2dict(args.annotations))

    block.segments[0].analogsignals[0].annotations.update(orientation_top=args.orientation_top)
    block.segments[0].analogsignals[0].annotations.update(orientation_right=args.orientation_right)

    # ToDo: add metadata
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded from mat file. '\
                                    .format(neo.__version__)
    if block.segments[0].analogsignals[0].description is None:
        block.segments[0].analogsignals[0].description = ''
    block.segments[0].analogsignals[0].description += 'MF simulation output '
    
    # Save data
    write_output(args.output, block)
    #write_neo(args.output, block)
