
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from FuncUtils2 import Compute_NumberOfGaussian, ComputeKDE, ComputeExpWavesAll, ExtractWaves, ComputeVelocityMacro


#############################################################################################

CLI = argparse.ArgumentParser(description=__doc__,
                  formatter_class=argparse.RawDescriptionHelpFormatter)
CLI.add_argument("--mouseRec", nargs='?', type=str, required=True, help="mouse number")
CLI.add_argument("--trial", nargs='*', type=int, required=True, help="trials number")
args = CLI.parse_args()

# analisi dei dati in input

dt = 10 #s # dt for moving variance wime window
sampling_rate = 25. #Hz
# parametri per l'analisi
DIM_X = 50
DIM_Y = 50
ReducedFactor_x = 6#4
ReducedFactor_y = 6#4
MAX_GAUSS_NUM = 4

# Input parameters
home = os.getcwd()

mouseRec = args.mouseRec#'170111'
t_arr = args.trial #1

if mouseRec == '170111':
    mask_elem_arr = [0,0,0,1,0,0]
    a_arr = [3, 2.4, 3, 1.9, 3, 2.2]
    p_arr = [4, 4, 3.3, 4, 1.3, 2]
    max_count_sim = 5
elif mouseRec == '170110':
    mask_elem_arr = [33,0,0,13,11,4]
    a_arr = [0.76, 1.9, 2.2, 1.9, 3, 2.4]
    p_arr = [3, 4, 3, 4, 2.3, 1.3]
    max_count_sim = 0

#a_arr, p_arr = ComputeOptimalValue(t_arr, home, TopoRec)
#print(a_arr)

path_exp_pipe = home + '/../Output/TI_LENS_' + str(mouseRec) + '_t'
save_path = home + '/../OutputPlot/TI_LENS_AllTrials_' + str(mouseRec)
save_path_fig = home + '/../OutputPlot/Figures/'
# append waves detected by all trials 
if not os.path.exists(save_path_fig):
    os.makedirs(save_path_fig)

waves_all = []
velocities_all = []
directions_all = []
isi_all = []
grid_list_exp_all = np.empty([0, 9, 9])

vel_indexes_exp = []
dir_indexes_exp = []
isi_indexes_exp = []
vel_indexes_sim = []
dir_indexes_sim = []
isi_indexes_sim = []

n_w = 0
for t_idx, t in enumerate(t_arr):

    # load experimental data
    # STAGE 04 output
    fname_wave_exp = path_exp_pipe + str(t) + '/stage04_wave_detection/WaveHunt/waves.pkl'
    path_macro_exp = path_exp_pipe + str(t) + '/stage05_channel-wave_characterization/'
    block_w = load_input(fname_wave_exp)#io_w.read_block()
    waves_exp = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
    grid_list_exp = ExtractWaves(waves_exp, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50)
    grid_list_exp_all = np.concatenate([grid_list_exp_all, grid_list_exp])
    #Macroscopic observables
    velocities_df = pd.read_csv(path_macro_exp + 'velocity_local/wavefronts_velocity_local.csv')
    isi_df = pd.read_csv(path_macro_exp + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
    directions_df = pd.read_csv(path_macro_exp + 'direction_local/wavefronts_direction_local.csv')

    # macro dimentions non-cropped
    velocities = np.array(velocities_df['velocity_local'])
    directions = np.array(np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red']))
    nan_idx_dir = ~np.isnan(directions)
    directions = directions[~np.isnan(directions)]
    isi = np.array(isi_df['inter_wave_interval_local'])
    nan_idx_isi = ~np.isnan(isi)
    isi = isi[~np.isnan(isi)]
    
    vel_indexes_exp.extend(velocities_df['wavefronts_id']+n_w)
    dir_indexes_exp.extend(directions_df['wavefronts_id'][nan_idx_dir]+n_w)
    isi_indexes_exp.extend(isi_df['wavefronts_id'][nan_idx_isi]+n_w)
    
    velocities_all.extend(velocities)
    directions_all.extend(directions)
    isi_all.extend(isi)
    n_w = len(grid_list_exp_all)

MacroObsExp = ComputeKDE(np.array(velocities_all), np.array(directions_all), np.array(isi_all))
kde_vel_macro, xx_vel_macro, collection = ComputeVelocityMacro(grid_list_exp_all)
gmm_exp, count_exp, mask_exp, variance, points_exp = ComputeExpWavesAll(grid_list_exp_all, mask_idx = 2, return_mask = True)

##############################################################################################
#                                            PLOT FIG 2                                      #
##############################################################################################
#! Plot onde platoniche

#fig, axs = plt.subplots(1, 6, sharex=True, sharey=True)
#fig.set_size_inches(9,2, forward=True)

fig = plt.figure() #plt.subplots(3, 1, sharex = True)
fig.set_size_inches(9, 2, forward=True)
spec = fig.add_gridspec(ncols=6, nrows=1, wspace = .4, hspace = 0.4, width_ratios = [1.,1.,1., 1.15, 1.15, 1.15])

################################################# plot macro
import matplotlib.gridspec as gridspec

subspec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[3:6], wspace = .4)#.7)

axs = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1]),
      fig.add_subplot(subspec[2])]

directions_platonic = []
velocity_platonic = []


mask_1d = np.reshape(mask_exp, np.shape(mask_exp)[0]*np.shape(mask_exp)[1])

#for w_i, platonic_w in enumerate(gmm_exp.means_[np.argsort(-count_exp)]):
for w_i, platonic_w in enumerate(gmm_exp.means_[np.argsort(-count_exp)]):
    axs[w_i].tick_params(axis='both', which='major', labelsize=6)
    platonic = np.empty(np.shape(mask_1d))*np.nan
    platonic[~np.isnan(mask_1d)] = platonic_w*variance[np.argsort(-count_exp)[w_i]]
    platonic = np.reshape(platonic, [9,9])
    im = axs[w_i].imshow(platonic*1000, cmap = 'RdBu_r')
    divider = make_axes_locatable(axs[w_i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=7.)
    if w_i == 2:
        cb.ax.set_ylabel(r'$\Delta t (s)$', rotation=270, fontsize = 7.)
    axs[w_i].tick_params(axis='both', which='major', labelsize=7)
    axs[w_i].set_xticks([])
    axs[w_i].set_yticks([])
    axs[w_i].set_box_aspect(1)
    

    dx = np.reshape(np.diff(platonic, axis = 0), (np.shape(platonic)[0]-1)*np.shape(platonic)[1])
    dy = np.reshape(np.diff(platonic, axis = 1), (np.shape(platonic)[0]-1)*np.shape(platonic)[1])
    d= np.arctan2(dy, dx)
    directions_platonic.append(d[~np.isnan(d)])
    v = (1./np.sqrt(dx**2+dy**2)*0.1*6/1000.) # mm
    velocity_platonic.append(v[~np.isnan(v)])
    axs[w_i].set_title('Fraction: ' + str(np.round(count_exp[np.argsort(-count_exp)[w_i]]/np.sum(count_exp), 2))+ '\n' +  r'  $\bar{v}=$'+ str(np.round(np.mean(v[~np.isnan(v)]),2)) + 'mm/s', fontsize = 7.)

MacroObsExpModes = []

for i_m, mode in enumerate(np.argsort(-count_exp)):
    # per ogni modo
    #1. cerco le onde identificate in quel modo
    waves_idx = np.where(points_exp == mode)[0]
    idx_vel = np.array([i for i,j in enumerate(vel_indexes_exp) if j in waves_idx], dtype = 'int32')
    idx_dir = np.array([i for i,j in enumerate(dir_indexes_exp) if j in waves_idx], dtype = 'int32')
    idx_isi = np.array([i for i,j in enumerate(isi_indexes_exp) if j in waves_idx], dtype = 'int32')
    #2. calcolo le kde su di esse
    #MacroObsExpModes.append(ComputeKDE(np.array(velocities_all)[idx_vel], np.array(directions_all)[idx_dir], np.array(isi_all)[idx_isi]))
    MacroObsExpModes.append(ComputeKDE(np.array(velocities_all)[idx_vel], directions_platonic[i_m], np.array(isi_all)[idx_isi]))



colorscheme = ['#003f5c', '#ffa600', '#7a5195', '#ef5675']
subspec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[0:3], wspace = 0.5)#.75)

ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1], projection='polar'),
      fig.add_subplot(subspec[2])]

ax[0].plot(np.exp(MacroObsExp['xx_vel']), MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])/np.sum(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])*np.diff(np.insert(np.exp(MacroObsExp['xx_vel']),0,0))), color = 'black', label = '0.1mm')#, label = 'all modes')
ax[0].plot(np.exp(xx_vel_macro), kde_vel_macro(xx_vel_macro)/np.sum(kde_vel_macro(xx_vel_macro))*25, color = 'orange', linewidth = 0.7, label = '0.6mm')
ax[0].legend(fontsize = 7., borderpad=0.1, labelspacing=0.2, handlelength=0.8)
ax[0].tick_params(axis='both', which='major', labelsize=7)
ax[0].set_xlabel('wave velocity (mm/s)', fontsize = 7.)
ax[0].set_ylabel('density', fontsize = 7.)
ax[0].set_box_aspect(1)

#directions
ax[1].plot(MacroObsExp['xx_dir']-np.pi/2., MacroObsExp['kde_dir']/np.sum(MacroObsExp['kde_dir']), color = 'black')
colorscheme = ['#003f5c', '#ffa600', '#7a5195', '#ef5675']
ax[1].set_xlabel('wave directions', fontsize = 7.)
ax[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax[1].set_xticklabels(['L', 'A', 'M', 'P'], fontsize = 7)
ax[1].set_yticks([])
ax[1].tick_params(axis="x", pad=-1)



#isi
ax[2].plot(MacroObsExp['xx_isi'], MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])/np.sum(MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])), color = 'black')
ax[2].set_ylabel('density', fontsize = 7.)
ax[2].tick_params(axis='both', which='major', labelsize=7)
ax[2].set_xlabel('IWI (s)', fontsize = 7.)
ax[2].set_box_aspect(1)

plt.tight_layout()
plt.savefig(save_path_fig + 'MacroExpFig2_Topo'+str(mouseRec) + '.eps', format = 'eps')
plt.savefig(save_path_fig + 'MacroExpFig2_Topo'+str(mouseRec) + '.png', dpi = 200)
plt.show()
plt.close()



fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
fig.set_size_inches(1.58*1,1.3, forward=True)
#colorscheme = ['#7a5195','#003f5c','#ef5675','#ffa600']
colorscheme = ['#003f5c', '#ffa600', '#7a5195']
labellist = ['mode #0', 'mode #1', 'mode #2']
pie1 = axs.pie(count_exp[np.argsort(-count_exp)], colors = colorscheme, labels = labellist, counterclock = False, labeldistance = None)
axs.set_title('Experiment', fontsize = 7.)
fig.legend([pie1], labels = labellist, fontsize = 7., loc="lower center",  ncol = 3, handletextpad= 0.2, columnspacing = 1)
plt.tight_layout()
plt.savefig(save_path_fig + 'PieChart_Exp_Fraction_' + str(mouseRec) + '.eps', format = 'eps')
#plt.show()
plt.close()
