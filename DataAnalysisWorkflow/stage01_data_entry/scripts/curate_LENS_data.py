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
                     help="whether hemodynamics correction is applicable")
    CLI.add_argument("--path_to_reflectance_data", nargs='+', type=str, default=None,
                     help="path to reflectance data")


    args, unknown = CLI.parse_known_args()

    # Load optical data
    io = neo.io.tiffio.TiffIO(directory_path=args.data,
                              sampling_rate=args.sampling_rate*pq.Hz,
                              spatial_scale=args.spatial_scale*pq.mm,
                              units='dimensionless')
    # loading the data flips the images vertically!

    block = io.read_block()

    # change data orientation to be top=ventral, right=lateral
    imgseq = block.segments[0].imagesequences[0]
    imgseq = flip_image(imgseq, axis=-2)
    imgseq = rotate_image(imgseq, rotation=-90)
    block.segments[0].imagesequences[0] = imgseq

    # Transform into analogsignals
    block.segments[0].analogsignals = []
    block = imagesequences_to_analogsignals(block)

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
    block.segments[0].description = 'Loaded with neo.TiffIO (neo version {}). '\
                                    .format(neo.__version__)
    if block.segments[0].analogsignals[0].description is None:
        block.segments[0].analogsignals[0].description = ''
    block.segments[0].analogsignals[0].description += 'Ca+ imaging signal. '

    # Save data
    write_output(args.output, block)
