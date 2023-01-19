import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import random
from utils.io import load_input, save_plot
from utils.neo import time_slice


def plot_traces(original_asig, processed_asig, channel):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax1 = plt.subplots()
    palette = sns.color_palette()

    ax1.plot(original_asig.times,
            original_asig.as_array()[:,channel],
            color=palette[0])
    ax1.set_ylabel('original signal', color=palette[0])
    ax1.tick_params('y', colors=palette[0])

    ax2 = ax1.twinx()
    ax2.plot(processed_asig.times,
            processed_asig.as_array()[:,channel],
            color=palette[1])
    ax2.set_ylabel('processed signal', color=palette[1])
    ax2.tick_params('y', colors=palette[1])

    ax1.set_title('Channel {}'.format(channel))
    ax1.set_xlabel('time [{}]'.format(original_asig.times.units.dimensionality.string))

    return ax1, ax2


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--original_data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--processed_data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--img_dir",  nargs='?', type=str, required=True,
                     help="path of output figure directory")
    CLI.add_argument("--img_name", nargs='?', type=str,
                     default='processed_trace_channel0.png',
                     help='example filename for channel 0')
    CLI.add_argument("--t_start", nargs='?', type=float, default=0,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=float, default=10,
                     help="stop time in seconds")
    CLI.add_argument("--channels", nargs='+', type=int, default=0,
                     help="channel to plot")
    args = CLI.parse_args()

    #orig_asig = load_neo(args.original_data, 'analogsignal', lazy=False)
    orig_asig = (load_input(args.original_data)).segments[0].analogsignals[0]
    orig_asig = time_slice(orig_asig, t_start=args.t_start, t_stop=args.t_stop,
                           lazy=False, channel_indexes=args.channels)

    proc_asig = (load_input(args.processed_data)).segments[0].analogsignals[0]
    proc_asig = time_slice(proc_asig, t_start=args.t_start, t_stop=args.t_stop,
                           lazy=False, channel_indexes=args.channels)

    for channel in args.channels:
        plot_traces(orig_asig, proc_asig, channel)
        output_path = os.path.join(args.img_dir,
                                   args.img_name.replace('_channel0', f'_channel{channel}'))
        save_plot(output_path)
