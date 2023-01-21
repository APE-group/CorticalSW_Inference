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
from FuncUtils2 import Compute_NumberOfGaussian, interpolate_wave, MyBestAndWorst, ComputePlatonicWavesAll, ComputePlatonicWaves



CLI = argparse.ArgumentParser(description=__doc__,
                  formatter_class=argparse.RawDescriptionHelpFormatter)
CLI.add_argument("--mouseRec", nargs='?', type=str, required=True, help="mouse number")
CLI.add_argument("--trial", nargs='?', type=int, required=True, help="trial number")
args = CLI.parse_args()

# Input parameters
home = os.getcwd()

mouseRec = args.mouseRec#'170111'
trial = args.trial #1

t_arr = [trial] 
path_exp_pipe = home + '/../Output/TI_LENS_' + str(mouseRec) + '_t' + str(trial)
path_sim = home + '/../Output/TI_LENS_MF_' + str(mouseRec) + '_t' + str(trial) + '_'
path_sim_inner = home + '/../Output/TI_LENS_MF_' + str(mouseRec) + '_t' + str(trial)  + '_InnerLoop'
path_sim_data = home + '/../Output/ConvolvedData/'+ str(mouseRec) + '_t' + str(trial) +'_Ampl_Period/'
save_data = home + '/../Output/OutputData/TI_LENS_MF' + str(mouseRec) + '_t' + str(trial)
save_path_fig = home + '/../OutputPlot/Figures/'
if not os.path.exists(save_path_fig):
    os.makedirs(save_path_fig)

# Read amplitude and period values and get best value
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


import scipy as sp
EMD_Vel = np.load(save_data + '_EMD_vel.npy')
EMD_Dir = np.load(save_data + '_EMD_dir.npy')
EMD_Isi = np.load(save_data + '_EMD_isi.npy')
EMD_Num = np.load(save_data + '_EMD_num.npy')
EMD_combination, EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num = MyBestAndWorst(EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num)

BestIdx = np.unravel_index(np.nanargmin(EMD_combination), EMD_combination.shape)
a = Amplitude[BestIdx[0]]
p = Period[BestIdx[1]]

WorstIdx = np.unravel_index(np.nanargmax(EMD_combination), EMD_combination.shape)
bad_a = Amplitude[WorstIdx[0]]
bad_p = Period[WorstIdx[1]]

# load experimental data
# STAGE 04 output
fname_wave_exp = path_exp_pipe + '/stage04_wave_detection/WaveHunt/waves.pkl'
fname_signal_exp = path_exp_pipe + '/stage02_processing/normalization/normalization.pkl'
block_s = load_input(fname_signal_exp)#io_w.read_block()
sig_exp = np.nanmean(block_s.segments[0].analogsignals[0], axis = 1)
block_w = load_input(fname_wave_exp)#io_w.read_block()
waves_exp = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
gmm_exp, label_exp = ComputePlatonicWaves(waves_exp,'', mask_idx = 0, return_mask = True)


fname_wave = path_sim_inner +  '/stage04_wave_detection/WaveHunt/waves.pkl'
block_w = load_input(fname_wave)#io_w.read_block()
waves_inner = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
fname_signal_inner = path_sim_inner + '/stage02_processing/normalization/normalization.pkl'
block_s = load_input(fname_signal_inner)#io_w.read_block()
sig_inner = np.nanmean(block_s.segments[0].analogsignals[0], axis = 1)
gmm_inner, label_inner = ComputePlatonicWaves(waves_inner, '', mask_idx = 0, return_mask = True)

fname_wave = path_sim + str(a) + '_' + str(p) + '/stage04_wave_detection/WaveHunt/waves.pkl'
block_w = load_input(fname_wave)#io_w.read_block()
waves = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
fname_signal_sim = path_sim + str(a) + '_' + str(p) + '/stage02_processing/normalization/normalization.pkl'
block_s = load_input(fname_signal_sim)#io_w.read_block()
sig_sim = np.nanmean(block_s.segments[0].analogsignals[0], axis = 1)
gmm_sim, label_sim = ComputePlatonicWaves(waves,'', mask_idx = 0, return_mask = True)

# all 
gmm_all, label_all = ComputePlatonicWavesAll(waves_exp, waves_inner, waves, mask_idx = 0, return_mask = True)


############################# Plotting ########################################################
max_time = 30
color = ['#ffa600', '#7a5195', '#003f5c', '#ef5675']


col_exp = np.array(["#000000" for i in range(0, len(waves_exp.times))])
for w_idx, w in enumerate(np.unique(waves_exp.labels)):
    idx = np.where(waves_exp.labels == w)[0]
    cl = label_all[0+w_idx]
    col_exp[idx] = color[cl]

col_inner = np.array(["#000000" for i in range(0, len(waves_inner.times))])
for w_idx, w in enumerate(np.unique(waves_inner.labels)):
    idx = np.where(waves_inner.labels == w)[0]
    cl = label_all[len(label_exp)+w_idx]
    col_inner[idx] = color[cl]

col_outer = np.array(["#000000" for i in range(0, len(waves.times))])
for w_idx, w in enumerate(np.unique(waves.labels)):
    idx = np.where(waves.labels == w)[0]
    cl = label_all[len(label_exp)+len(label_inner)+w_idx]
    col_outer[idx] = color[cl]

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
fig.set_size_inches(7.48,5, forward=True)

axs[0].tick_params(axis='both', which='major', labelsize=7)
axs[1].tick_params(axis='both', which='major', labelsize=7)
axs[2].tick_params(axis='both', which='major', labelsize=7)

axs[0].scatter(waves_exp.times, waves_exp.array_annotations['channels'], s=0.02, c = col_exp)# c=np.int32(waves_exp.labels), cmap = 'prism')
axs[0].set_title('experimental data', fontsize = 7.) 
axs[0].set_ylabel('channel id', fontsize = 7.) 
axs[0].set_xlim([0, max_time])
#axs[0].set_xticks([])

axs[1].scatter(waves_inner.times, waves_inner.array_annotations['channels'], s=0.02, c = col_inner)#np.int32(waves_inner.labels), cmap = 'prism')
axs[1].set_xlim([0, max_time])
axs[1].set_title('inner loop', fontsize = 7.) 
axs[1].set_ylabel('channel id', fontsize = 7.) 
#axs[1].set_xticks([])

axs[2].scatter(waves.times, waves.array_annotations['channels'], s=0.02, c=col_outer)#np.int32(waves.labels), cmap = 'prism')
axs[2].set_xlim([0, max_time])
axs[2].set_title('outer loop', fontsize = 7.) 
axs[2].set_ylabel('channel id', fontsize = 7.) 
axs[2].set_xlabel('time (s)', fontsize = 7.) 

#plt.show()
plt.tight_layout()
plt.savefig(save_path_fig + 'Activity_Topo'+ str(mouseRec) + '_t' + str(trial) +'.eps', format = 'eps')
plt.savefig(save_path_fig + 'Activity_Topo'+ str(mouseRec) + '_t' + str(trial) +'.png')
plt.show()
plt.close()


