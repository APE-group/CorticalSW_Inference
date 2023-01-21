#import matplotlib
#matplotlib.use('Agg') 

import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import neo 
import pickle
from sklearn.mixture import GaussianMixture
from skimage.measure import block_reduce

from scipy import stats
from scipy.special import i0
import sys
utils_path = os.path.join('..', 'DataAnalysisWorkflow/')
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(utils_path)

from utils.io import load_input

import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from FuncUtils import ComputeKDE, ComputeMacroEMD, ExtractMaxDonw, ExtractWaveDuration


CLI = argparse.ArgumentParser(description=__doc__,
                  formatter_class=argparse.RawDescriptionHelpFormatter)
CLI.add_argument("--mouseRec", nargs='?', type=str, required=True, help="mouse number")
CLI.add_argument("--trial", nargs='?', type=int, required=True, help="trials number")
args = CLI.parse_args()

# Input parameters
home = os.getcwd()

mouseRec = args.mouseRec#'170111'
trial = args.trial #1

t_arr = [trial]

path_exp_pipe = home + '/../Output/TI_LENS_' + str(mouseRec) + '_t'
path_sim = home + '/../Output/TI_LENS_MF_' + str(mouseRec) + '_t' + str(trial) + '_'
save_data = home + '/../Output/OutputData/TI_LENS_MF' + str(mouseRec) + '_t' + str(trial)
path_sim_data = home + '/../Output/ConvolvedData/'+ str(mouseRec) + '_t' + str(trial) +'_Ampl_Period/'
if not os.path.exists(save_data):
    os.makedirs(save_data)


# Read amplitude and period values

Period = []
Amplitude = []

for filename in os.listdir(path_sim_data):
     
    tmp = filename.split("_")
    idx = tmp.index("DAMP")
    Amplitude.append(tmp[idx+1])

    idx = tmp.index("PERIOD")
    Period.append(tmp[idx+1])

Amplitude = np.unique(Amplitude)
Period = np.unique(Period)

# load experimental data
# STAGE 04 output
fname_wave_exp = path_exp_pipe + str(args.trial) + '/stage04_wave_detection/WaveHunt/waves.pkl'
path_macro_exp = path_exp_pipe + str(args.trial) + '/stage05_channel-wave_characterization/'
block_w = load_input(fname_wave_exp)#io_w.read_block()
waves_exp = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
N_exp = len(np.unique(waves_exp.labels))



downstate, std = ExtractMaxDonw(waves_exp)
duration = ExtractWaveDuration(waves_exp)
np.save(save_data + '_exp.npy', np.array([std, downstate, duration]))


#Macroscopic observables
velocities_df = pd.read_csv(path_macro_exp + 'velocity_local/wavefronts_velocity_local.csv')
isi_df = pd.read_csv(path_macro_exp + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
directions_df = pd.read_csv(path_macro_exp + 'direction_local/wavefronts_direction_local.csv')

# macro dimentions non-cropped
velocities = np.array(velocities_df['velocity_local'])
np.save(save_data + '_VelStdexp.npy', np.std(velocities))
vel_std_exp = np.std(velocities)

directions = np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red'])
isi = np.array(isi_df['inter_wave_interval_local'])
isi = isi[~np.isnan(isi)]
MacroObsExp = ComputeKDE(velocities, directions, isi)

# initialize emd matrixes
EMD_vel_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_dir_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_isi_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_num_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_down_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_waves_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_velstd_matrix = np.empty([len(Amplitude), len(Period)])*np.nan
EMD_duration_matrix = np.empty([len(Amplitude), len(Period)])*np.nan

# load data
for a_idx, a in enumerate(Amplitude):
    for p_idx, p in enumerate(Period):
        fname_wave = path_sim + str(a) + '_' + str(p) + '/stage04_wave_detection/WaveHunt/waves.pkl'
        path_macro = path_sim + str(a) + '_' + str(p) + '/stage05_channel-wave_characterization/'

        try:
            # STAGE 04 output
            block_w = load_input(fname_wave)#io_w.read_block()
            waves = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
            if len(np.unique(waves.labels)) > 1:
                N_sim = len(np.unique(waves.labels))
                EMD_waves_matrix[a_idx, p_idx] = np.abs(N_sim - N_exp)/N_exp
                
                downstate_sim, _ = ExtractMaxDonw(waves)
                duration_sim = ExtractWaveDuration(waves)

                # load_dataframe stage05
                velocities_df = pd.read_csv(path_macro + 'velocity_local/wavefronts_velocity_local.csv')
                isi_df = pd.read_csv(path_macro + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
                directions_df = pd.read_csv(path_macro + 'direction_local/wavefronts_direction_local.csv')

                # macro dimentions non-cropped
                velocities = np.array(velocities_df['velocity_local'])
                EMD_velstd_matrix[a_idx, p_idx] = np.std(velocities)
                vel_std = np.std(velocities)

                directions = np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red'])
                isi = np.array(isi_df['inter_wave_interval_local'])
                isi = isi[~np.isnan(isi)]

                MacroObsSim = ComputeKDE(velocities, directions, isi)

                EMD_vel_matrix[a_idx, p_idx], EMD_dir_matrix[a_idx, p_idx], EMD_isi_matrix[a_idx, p_idx] = ComputeMacroEMD(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])/np.sum(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])),
                                                                                                                           MacroObsSim['kde_vel'](MacroObsSim['xx_vel'])/np.sum(MacroObsSim['kde_vel'](MacroObsSim['xx_vel'])),
                                                                                                                           MacroObsExp['kde_dir']/np.sum(MacroObsExp['kde_dir']),
                                                                                                                           MacroObsSim['kde_dir']/np.sum(MacroObsSim['kde_dir']),
                                                                                                                           MacroObsExp['xx_dir'],
                                                                                                                           MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])/np.sum(MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])),
                                                                                                                           MacroObsSim['kde_isi'](MacroObsSim['xx_isi'])/np.sum(MacroObsSim['kde_isi'](MacroObsSim['xx_isi'])))
                
                EMD_num = 1
                if downstate_sim > downstate+ std: #/2.: #+std: # < 0.25 Hz
                    EMD_num = np.nan
                if duration_sim/duration>1:#.5: #+std: # < 0.25 Hz
                    EMD_num = np.nan


                EMD_num_matrix[a_idx, p_idx] = EMD_num
                EMD_down_matrix[a_idx, p_idx] = downstate_sim
                EMD_duration_matrix[a_idx, p_idx] =duration_sim


        except FileNotFoundError:
            print('FILE NOT FOUND')





np.save(save_data + '_EMD_vel.npy', EMD_vel_matrix)
np.save(save_data + '_EMD_dir.npy', EMD_dir_matrix)
np.save(save_data + '_EMD_isi.npy', EMD_isi_matrix)
np.save(save_data + '_EMD_num.npy', EMD_num_matrix)
np.save(save_data + '_down.npy', EMD_down_matrix)
np.save(save_data + '_waves.npy', EMD_waves_matrix)
np.save(save_data + '_velstd.npy', EMD_velstd_matrix)
np.save(save_data + '_duration.npy', EMD_duration_matrix)

