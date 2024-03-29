"""
Docstring
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.io import load_neo, save_plot
from utils.parse import none_or_str


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to spatial derivative dataframe")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                     help="name of neo.Event to analyze (must contain waves)")
    args, unknown = CLI.parse_known_args()

    df = pd.read_csv(args.data)

    direction_df = pd.DataFrame(df.channel_id, columns=['channel_id'])
    direction_df['direction_local_x'] = df.dt_x
    direction_df['direction_local_y'] = df.dt_y
    direction_df[f'{args.event_name}_id'] = df[f'{args.event_name}_id']
    
    # code to make directions less noisy
    x_reduced = np.array(df.x_coords // 8)
    y_reduced = np.array(df.y_coords // 8)
    dt_x_red = []
    dt_y_red = []

    for x in np.unique(x_reduced):
        for y in np.unique(y_reduced):
            idx = np.intersect1d(np.where(x_reduced == x)[0], np.where(y_reduced == y)[0])
            dt_x_red.append(np.mean(np.array(df.dt_x)[idx]))
            dt_y_red.append(np.mean(np.array(df.dt_y)[idx]))
    dt_x_red_all = np.empty(len(df.dt_x))*np.nan
    dt_x_red_all[0:len(dt_x_red)] = dt_x_red
    dt_y_red_all = np.empty(len(df.dt_y))*np.nan
    dt_y_red_all[0:len(dt_y_red)] = dt_y_red
    direction_df['direction_local_x_red'] = dt_x_red_all
    direction_df['direction_local_y_red'] = dt_y_red_all
    direction_df.to_csv(args.output)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #ax.hist(np.angle(df.dt_x + 1j*df.dt_y), bins=36, range=[-np.pi, np.pi])
    
    ax.hist(np.arctan2(1./np.array(dt_x_red), 1./np.array(dt_y_red)), bins=36, range=[-np.pi, np.pi])
    #ax.hist(np.arctan2(1./np.array(dt_x_red), 1./np.array(dt_y_red)), bins=36, range=[-np.pi, np.pi])
    #plt.show()
    if args.output_img is not None:
        save_plot(args.output_img)
