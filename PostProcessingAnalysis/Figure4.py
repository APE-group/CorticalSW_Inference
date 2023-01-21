
#import matplotlib
#matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import cmocean

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


from FuncUtils import ComputeKDE, ComputeOptimalValue, MyBestAndWorst, BestAndWorst


#############################################################################################



CLI = argparse.ArgumentParser(description=__doc__,
                  formatter_class=argparse.RawDescriptionHelpFormatter)
CLI.add_argument("--mouseRec", nargs='?', type=str, required=True, help="mouse number")
CLI.add_argument("--trial", nargs='*', type=int, required=True, help="trial number")
args = CLI.parse_args()

# Input parameters
home = os.getcwd()

mouseRec = args.mouseRec
trial = args.trial[0]


path_exp_pipe = home + '/../Output/TI_LENS_' + str(mouseRec) + '_t' + str(trial)
path_macro_exp = path_exp_pipe + '/stage05_channel-wave_characterization/'
path_sim = home + '/../Output/TI_LENS_MF_' + str(mouseRec) + '_t' + str(trial) + '_'
path_sim_inner = home + '/../Output/TI_LENS_MF_' + str(mouseRec) + '_t' + str(trial)  + '_InnerLoop'
path_sim_data = home + '/../Output/ConvolvedData/'+ str(mouseRec) + '_t' + str(trial) +'_Ampl_Period/'
save_data = home + '/../Output/OutputData/TI_LENS_MF' + str(mouseRec) + '_t' + str(trial)
save_path_fig = home + '/../OutputPlot/Figures/'
if not os.path.exists(save_path_fig):
    os.makedirs(save_path_fig)


import matplotlib.gridspec as gridspec
fig = plt.figure() #plt.subplots(3, 1, sharex = True)
fig.set_size_inches(8, 6.2, forward=True)
spec = fig.add_gridspec(ncols=4, nrows=5, wspace = 0.4, hspace = 0.6, height_ratios = [1,1,0.3,1.7,1])
##############################################################################################
#                                       PLOT FIG 5 MACRO                                     #
##############################################################################################

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

EMD_Vel = np.load(save_data + '_EMD_vel.npy')
EMD_Dir = np.load(save_data + '_EMD_dir.npy')
EMD_Isi = np.load(save_data + '_EMD_isi.npy')
EMD_Num = np.load(save_data + '_EMD_num.npy')
EMD_combination, EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num = MyBestAndWorst(EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num)

BestIdx = np.unravel_index(np.nanargmin(EMD_combination), EMD_combination.shape)
good_a = Amplitude[BestIdx[0]]
good_p = Period[BestIdx[1]]

WorstIdx = np.unravel_index(np.nanargmax(EMD_combination), EMD_combination.shape)
bad_a = Amplitude[WorstIdx[0]]
bad_p = Period[WorstIdx[1]]


# load experimental data
#Macroscopic observables
velocities_df = pd.read_csv(path_macro_exp + 'velocity_local/wavefronts_velocity_local.csv')
isi_df = pd.read_csv(path_macro_exp + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
directions_df = pd.read_csv(path_macro_exp + 'direction_local/wavefronts_direction_local.csv')

# macro dimentions non-cropped
velocities = np.array(velocities_df['velocity_local'])
directions = np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red'])
isi = np.array(isi_df['inter_wave_interval_local'])
isi = isi[~np.isnan(isi)]
MacroObsExp = ComputeKDE(velocities, directions, isi)

# GOOOD DATA
path_macro = path_sim + str(good_a) + '_' + str(good_p) + '/stage05_channel-wave_characterization/'

velocities_df = pd.read_csv(path_macro + 'velocity_local/wavefronts_velocity_local.csv')
isi_df = pd.read_csv(path_macro + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
directions_df = pd.read_csv(path_macro + 'direction_local/wavefronts_direction_local.csv')

velocities = np.array(velocities_df['velocity_local'])
directions = np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red'])
isi = np.array(isi_df['inter_wave_interval_local'])
isi = isi[~np.isnan(isi)]
MacroObsSim_Good = ComputeKDE(velocities, directions, isi)

# BAD DATA
path_macro = path_sim + str(bad_a) + '_' + str(bad_p) + '/stage05_channel-wave_characterization/'

# load_dataframe stage05
velocities_df = pd.read_csv(path_macro + 'velocity_local/wavefronts_velocity_local.csv')
isi_df = pd.read_csv(path_macro + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
directions_df = pd.read_csv(path_macro + 'direction_local/wavefronts_direction_local.csv')

# macro dimentions non-cropped
velocities = np.array(velocities_df['velocity_local'])
directions = np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red'])
isi = np.array(isi_df['inter_wave_interval_local'])
isi = isi[~np.isnan(isi)]
MacroObsSim_Bad = ComputeKDE(velocities, directions, isi)


################################################# plot macro good

subspec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[0,1:4], wspace = 0.01)
ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1], projection='polar'),
      fig.add_subplot(subspec[2])]


ax[0].tick_params(axis='both', which='major', labelsize=7)
ax[1].tick_params(axis='both', which='major', labelsize=7)
ax[2].tick_params(axis='both', which='major', labelsize=7)

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[0].set_box_aspect(1)
ax[2].set_box_aspect(1)

    
ax[0].plot(np.exp(MacroObsSim_Good['xx_vel']), MacroObsSim_Good['kde_vel'](MacroObsSim_Good['xx_vel'])/np.sum(MacroObsSim_Good['kde_vel'](MacroObsSim_Good['xx_vel'])*np.diff(np.insert(np.exp(MacroObsSim_Good['xx_vel']), 0, 0))), color = 'red', linewidth = 1.)
ax[0].plot(np.exp(MacroObsExp['xx_vel']), MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])/np.sum(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])*np.diff(np.insert(np.exp(MacroObsExp['xx_vel']), 0, 0))), color = 'black', linewidth = 1.)
#ax[0].set_xlabel('wave velocity (mm/s)', fontsize = 7.)
ax[0].set_ylabel('density', fontsize = 7.)
ax[0].set_yticks([])
ax[0].set_xticks([])
#axs0.set_xlim([0,30])

#directions
ax[1].plot(MacroObsSim_Good['xx_dir']-np.pi/2., MacroObsSim_Good['kde_dir']/np.sum(MacroObsSim_Good['kde_dir']), color = 'red', linewidth = 1.)
ax[1].plot(MacroObsExp['xx_dir']-np.pi/2., MacroObsExp['kde_dir']/np.sum(MacroObsExp['kde_dir']), color = 'black', linewidth = 1.)
#ax[1].set_xlabel('wave directions', fontsize = 7.)
ax[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax[1].set_xticklabels(['L', 'A', 'M', 'P'], fontsize = 7)
ax[1].tick_params(axis='both', which='major', pad=0.00001)
ax[1].set_yticks([])

#isi
ax[2].plot(MacroObsSim_Good['xx_isi'], MacroObsSim_Good['kde_isi'](MacroObsSim_Good['xx_isi'])/np.sum(MacroObsSim_Good['kde_isi'](MacroObsSim_Good['xx_isi'])), color = 'red', linewidth = 1, label = 'sim')
ax[2].plot(MacroObsExp['xx_isi'], MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])/np.sum(MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])), color = 'black', linewidth = 1, label = 'exp')
ax[2].tick_params(axis='both', which='major', labelsize=7)
#ax[2].legend(fontsize = 7.)
ax[2].set_yticks([])
ax[2].set_xticks([])
#ax[2].set_xlabel('IWI (s)', fontsize = 7.)

################################################# plot macro good

subspec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[1,1:4], wspace = 0.2)
ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1], projection='polar'),
      fig.add_subplot(subspec[2])]


ax[0].tick_params(axis='both', which='major', labelsize=7)
ax[1].tick_params(axis='both', which='major', labelsize=7)
ax[2].tick_params(axis='both', which='major', labelsize=7)

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[0].set_box_aspect(1)
ax[2].set_box_aspect(1)

ax[0].plot(np.exp(MacroObsSim_Bad['xx_vel']), MacroObsSim_Bad['kde_vel'](MacroObsSim_Bad['xx_vel'])/np.sum(MacroObsSim_Bad['kde_vel'](MacroObsSim_Bad['xx_vel'])*np.diff(np.insert(np.exp(MacroObsSim_Bad['xx_vel']), 0, 0))), color = 'red', linewidth = 1.)
ax[0].plot(np.exp(MacroObsExp['xx_vel']), MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])/np.sum(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])*np.diff(np.insert(np.exp(MacroObsExp['xx_vel']), 0, 0))), color = 'black', linewidth = 1.)
ax[0].tick_params(axis='both', which='major', labelsize=7)
ax[0].set_xlabel('wave velocity (mm/s)', fontsize = 7.)
ax[0].set_ylabel('density', fontsize = 7.)
ax[0].set_yticks([])
#axs3.set_xlim([0,30])

#directions
ax[1].plot(MacroObsSim_Bad['xx_dir']-np.pi/2., MacroObsSim_Bad['kde_dir']/np.sum(MacroObsSim_Bad['kde_dir']), color = 'red', linewidth = 1.)
ax[1].plot(MacroObsExp['xx_dir']-np.pi/2., MacroObsExp['kde_dir']/np.sum(MacroObsExp['kde_dir']), color = 'black', linewidth = 1.)
ax[1].set_xlabel('wave directions', fontsize = 7.)
ax[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax[1].set_xticklabels(['L', 'A', 'M', 'P'], fontsize = 7)
ax[1].set_yticks([])
ax[1].tick_params(axis='both', which='major', pad=0.00001)

#isi
ax[2].plot(MacroObsSim_Bad['xx_isi'], MacroObsSim_Bad['kde_isi'](MacroObsSim_Bad['xx_isi'])/np.sum(MacroObsSim_Bad['kde_isi'](MacroObsSim_Bad['xx_isi'])), color = 'red', linewidth = 1, label = 'sim')
ax[2].plot(MacroObsExp['xx_isi'], MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])/np.sum(MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])), color = 'black', linewidth = 1, label = 'exp')
ax[2].legend(fontsize = 7.)
ax[2].tick_params(axis='both', which='major', labelsize=7)
ax[2].set_yticks([])
ax[2].set_xlabel('isi (s)', fontsize = 7.)


###############################################################################################
##                                       PLOT FIG 5 EMD                                       #
###############################################################################################


AmplitudeLabel = ['0','3', '6']
AmplitudeTick = [1,9,20]
PeriodLabels = ['0.37', '3.5', '6.9']

import scipy as sp
EMD_combination2, _,_,_ = BestAndWorst(EMD_Vel, EMD_Dir, EMD_Isi)
duration_matrix = np.load(save_data + '_duration.npy')
down_matrix = np.load(save_data + '_down.npy')
exp = np.load(save_data + '_exp.npy')
duration = exp[2]
downstate = exp[1]
std = exp[0]

duration_matrix = sp.ndimage.gaussian_filter(duration_matrix, sigma=1., order = 0, truncate = 2., mode = 'constant', cval = 0)#np.nanmean(EMD_Isi)+0.05)
duration_matrix[:, 0]  = duration_matrix[:,0] *13/9.
duration_matrix[:, -1] = duration_matrix[:,-1]*13/9.
duration_matrix[0, :]  = duration_matrix[0,:] *13/9.
duration_matrix[-1,:]  = duration_matrix[-1,:]*13/9.

duration_matrix[:, 1]  = duration_matrix[:,1] *13/12.
duration_matrix[:, -2] = duration_matrix[:,-2]*13/12.
duration_matrix[1, :]  = duration_matrix[1,:] *13/12.
duration_matrix[-2,:]  = duration_matrix[-2,:]*13/12.

down_matrix = sp.ndimage.gaussian_filter(down_matrix, sigma=1., order = 0, truncate = 2., mode = 'constant', cval = 0)#np.nanmean(EMD_Isi)+0.05)
down_matrix[:, 0]  = down_matrix[:,0] *13/9.
down_matrix[:, -1] = down_matrix[:,-1]*13/9.
down_matrix[0, :]  = down_matrix[0,:] *13/9.
down_matrix[-1,:]  = down_matrix[-1,:]*13/9.
down_matrix[:, 1]  = down_matrix[:,1] *13/12.
down_matrix[:, -2] = down_matrix[:,-2]*13/12.
down_matrix[1, :]  = down_matrix[1,:] *13/12.
down_matrix[-2,:]  = down_matrix[-2,:]*13/12.

# load data
for a_idx, a in enumerate(Amplitude):
    for p_idx, p in enumerate(Period):
        downstate_sim = down_matrix[a_idx, p_idx]
        EMD_num = 1
        if downstate_sim > downstate+std*2:#/2.:
            EMD_num = np.nan
        if duration_matrix[a_idx, p_idx]/duration >1.5:
            EMD_num = np.nan
        if np.isnan(downstate_sim):
            EMD_num = np.nan
        if np.isnan(duration_matrix[a_idx, p_idx]):
            EMD_num = np.nan
        EMD_num_matrix[a_idx, p_idx] = EMD_num

    

EMD_Comb, EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num = MyBestAndWorst(EMD_Vel, EMD_Dir, EMD_Isi, EMD_num_matrix)
    #BestIdx = np.unravel_index(np.nanargmin(EMD_combination), EMD_combination.shape)


# Plot della grid search
subspec = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=spec[3:4,0:4], wspace = 0.1)
ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1]),
      fig.add_subplot(subspec[2]),
      fig.add_subplot(subspec[3])]

print(Amplitude)
print(Period)

ax[0].pcolormesh(Period, Amplitude, np.log10(EMD_combination2**2), shading='nearest', cmap = 'Greys')
im = ax[0].pcolormesh(Period, Amplitude, np.log10(EMD_Comb**2), shading='nearest', cmap = cmap_val)
ax[0].contour(Period, Amplitude, duration_matrix, levels=[duration*1.5], colors = 'orange', linewidths = 0.8)
ax[0].contour(Period, Amplitude, down_matrix, levels=[downstate+std*2], colors = 'green', linewidths = 0.8)
#im = axs[0].pcolormesh(xnew, ynew, znew, shading='nearest', cmap = cmap_val)
ax[0].tick_params(axis='both', which='major', labelsize=7, pad = 0.01)
ax[0].xaxis.tick_top()
ax[0].xaxis.set_label_position('top')
clb=fig.colorbar(im, ax = ax[0], orientation = 'horizontal', fraction=0.046, pad=0.04)
clb.ax.tick_params(labelsize=7) 
ax[0].set_title('combination',fontsize=7)
#im.set_clim(min_val,max_val)
#for label in ax[0].get_xticklabels()[::2]:
#    label.set_visible(False)
ax[0].set_xlabel('Period', fontsize = 7.)
ax[0].set_ylabel('Amplitude', fontsize = 7.)
ax[0].set_box_aspect(1)
ax[0].set_yticks(AmplitudeTick)
ax[0].set_xticks(PeriodLabels)
ax[0].set_yticklabels(AmplitudeLabel)

im = ax[1].pcolormesh(Period, Amplitude, np.log10(EMD_Vel**2), shading='nearest', cmap = cmap_val)
ax[1].tick_params(axis='both', which='major', labelsize=7, pad = 0.01)
ax[1].set_title('local velocities',fontsize=7)
ax[1].xaxis.tick_top()
ax[1].xaxis.set_label_position('top')
#for label in ax[1].get_xticklabels()[::2]:
#    label.set_visible(False)
clb=fig.colorbar(im, ax = ax[1], orientation = 'horizontal', fraction=0.046, pad=0.04)
clb.ax.tick_params(labelsize=7) 
#im.set_clim(min_val,max_val)
ax[1].set_xlabel('Period', fontsize = 7.)
ax[1].set_box_aspect(1)
ax[1].set_xticks(PeriodLabels)
ax[1].set_yticks([])

im = ax[2].pcolormesh(Period, Amplitude, np.log10(EMD_Dir**2), shading='nearest', cmap = cmap_val)
ax[2].tick_params(axis='both', which='major', labelsize=7, pad = 0.01)
ax[2].xaxis.tick_top()
ax[2].xaxis.set_label_position('top')
#for label in ax[2].get_xticklabels()[::2]:
#    label.set_visible(False)
ax[2].set_title('local directions',fontsize=7)
clb=fig.colorbar(im, ax = ax[2], orientation = 'horizontal', fraction=0.046, pad=0.04)
clb.ax.tick_params(labelsize=7) 
ax[2].set_xlabel('Period', fontsize = 7.)
ax[2].set_box_aspect(1)
ax[2].set_xticks(PeriodLabels)
ax[2].set_yticks([])

im = ax[3].pcolormesh(Period, Amplitude, np.log10(EMD_Isi**2), shading='nearest', cmap = cmap_val)
ax[3].tick_params(axis='both', which='major', labelsize=7)
ax[3].xaxis.tick_top()
ax[3].xaxis.set_label_position('top')
#for label in ax[3].get_xticklabels()[::2]:
#    label.set_visible(False)
ax[3].set_title('local isi',fontsize=7)
clb=fig.colorbar(im, ax = ax[3], orientation = 'horizontal', fraction=0.046, pad=0.04)
clb.ax.tick_params(labelsize=7) 
ax[3].set_xlabel('Period', fontsize = 7.)
ax[3].set_box_aspect(1)
ax[3].set_xticks(PeriodLabels)
ax[3].set_yticks([])

plt.savefig(save_path_fig + 'Fig4.eps', format = 'eps')
plt.savefig(save_path_fig + 'Fig4.png')
plt.show()
plt.close()
