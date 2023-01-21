
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
from FuncUtils2 import ComputeKDE, ComputeGMMWavesAll, ExtractWaves, PlotWavePlatonic, PlotMacro, MyComputeOptimalValue 




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
np.random.seed(10)


mouseRec = args.mouseRec
trial = args.trial #1

t_arr = trial

if mouseRec == '170111':
    max_count_sim = 5
elif mouseRec == '170110':
    max_count_sim = 0

path_exp_pipe = home + '/../Output/TI_LENS_' + str(mouseRec) + '_t'
save_path_fig = home + '/../OutputPlot/Figures/'
path_sim_root = home + '/../Output/TI_LENS_MF_' + str(mouseRec) + '_t'
path_sim_data = home + '/../Output/ConvolvedData/'+ str(mouseRec) + '_t'
save_data = home + '/../Output/OutputData/TI_LENS_MF' + str(mouseRec) + '_t'

a_arr, p_arr = MyComputeOptimalValue(t_arr, home, mouseRec, save_data, path_sim_data)

# append waves detected by all trials 

waves_all = []
velocities_all = []
directions_all = []
isi_all = []
grid_list_exp_all = np.empty([0, 9, 9])

for t_idx, t in enumerate(t_arr):

    # load experimental data
    # STAGE 04 output
    fname_wave_exp = path_exp_pipe + str(t) + '/stage04_wave_detection/WaveHunt/waves.pkl'
    path_macro_exp = path_exp_pipe + str(t) + '/stage05_channel-wave_characterization/'
    block_w = load_input(fname_wave_exp)#io_w.read_block()
    waves_exp = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
    grid_list_exp = ExtractWaves(waves_exp, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50)
    grid_list_exp_all = np.concatenate([grid_list_exp_all, grid_list_exp])

gmm_exp, count_exp, mask_exp, waves_exp = ComputeGMMWavesAll(grid_list_exp_all, mask_idx = 1, return_mask = True)

waves_sim_all = []
velocities_sim_all = []
directions_sim_all = []
isi_sim_all = []
grid_list_sim_all = np.empty([0, 9, 9])

for sim_idx, (a, p)  in enumerate(zip(a_arr, p_arr)):
    path_sim = path_sim_root + str(t_arr[sim_idx])  + '_'

    # load experimental data
    # STAGE 04 output
    fname_wave = path_sim + str(a) + '_' + str(p) + '/stage04_wave_detection/WaveHunt/waves.pkl'
    path_macro = path_sim + str(a) + '_' + str(p) + '/stage05_channel-wave_characterization/'

    block_w = load_input(fname_wave)#io_w.read_block()
    waves = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
    grid_list_sim = ExtractWaves(waves, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50)
    grid_list_sim_all = np.concatenate([grid_list_sim_all, grid_list_sim])

gmm_sim, count_sim, mask_sim, new_idx, waves_sim = ComputeGMMWavesAll(grid_list_sim_all, max_count = max_count_sim, mask = mask_exp.copy(), return_mask = False)

#############################################################################################
#                                                   ALL                                     #
#############################################################################################


#def func(pct,fraction, data):
#    val = pct*0 + data[np.where(np.round(np.array(fraction)*100, 3) == np.round(pct,3))[0]]
#    return "{:.1E}".format(val[0])

X_all = np.concatenate([waves_exp, waves_sim])
var_exp = np.std(waves_exp, axis = 0)
var_all_exp_sim = np.std(X_all, axis = 0)
waves_sim_shuff = waves_sim.copy()
np.random.shuffle(waves_sim_shuff.T)
X_all_exp_sim_shuff = np.concatenate([waves_exp, waves_sim_shuff])
waves_exp_shuff = waves_exp.copy()
np.random.shuffle(waves_exp_shuff.T)
X_all_exp_shuff_sim = np.concatenate([waves_exp_shuff, waves_sim])
var_all_exp_sim_shuff = np.std(X_all_exp_shuff_sim, axis = 0)

N_Comp =  4 #Compute_NumberOfGaussian(X_all, 8)
gm = (GaussianMixture(n_components=N_Comp, random_state=1).fit(X_all))
gm_exp_only = (GaussianMixture(n_components=3, random_state=1).fit(waves_exp))
gm_sim_shuff = (GaussianMixture(n_components=N_Comp, random_state=1).fit(X_all_exp_sim_shuff))
gm_exp_shuff = (GaussianMixture(n_components=N_Comp, random_state=1).fit(X_all_exp_shuff_sim))


points_exp = gm.predict(waves_exp)
points_exp_sim = gm.predict(X_all)
points_sim = gm.predict(waves_sim)
points_sim_shuff = gm.predict(waves_sim_shuff)

points_exp_sim_shuff = gm_sim_shuff.predict(waves_exp)
points_exp_sim_sim_shuff = gm_sim_shuff.predict(X_all_exp_sim_shuff)
points_sim_sim_shuff = gm_sim_shuff.predict(waves_sim)
points_sim_shuff_sim_shuff = gm_sim_shuff.predict(waves_sim_shuff)

prob_exp = gm.predict_proba(waves_exp)
prob_sim = gm.predict_proba(waves_sim)
prob_sim_shuff = gm.predict_proba(waves_sim_shuff)

prob_exp_sim_shuff = gm_sim_shuff.predict_proba(waves_exp)
prob_sim_sim_shuff = gm_sim_shuff.predict_proba(waves_sim)
prob_sim_shuff_sim_shuff = gm_sim_shuff.predict_proba(waves_sim_shuff)

mask_1d = np.reshape(mask_sim, np.shape(mask_sim)[0]*np.shape(mask_sim)[1])


#plt pie chart
fraction_exp = []
fraction_exp_sim = []
fraction_sim = []
fraction_exp_sim_shuff = []
fraction_exp_sim_sim_shuff = []
fraction_sim_sim_shuff = []
fraction_sim_shuff_sim_shuff = []
fraction_sim_shuff = []

confidence_gm = np.zeros([3, N_Comp])
confidence_gm_sim_shuff = np.zeros([3, N_Comp])

for mode in range(len(gm.means_)):
    fraction_exp.append(len(np.where(points_exp == mode)[0])/len(points_exp))
    fraction_exp_sim.append(len(np.where(points_exp_sim == mode)[0])/len(points_exp_sim))
    fraction_sim.append(len(np.where(points_sim == mode)[0])/len(points_sim))
    fraction_exp_sim_shuff.append(len(np.where(points_exp_sim_shuff == mode)[0])/len(points_exp_sim_shuff))
    fraction_sim_sim_shuff.append(len(np.where(points_sim_sim_shuff == mode)[0])/len(points_sim_sim_shuff))
    fraction_sim_shuff_sim_shuff.append(len(np.where(points_sim_shuff_sim_shuff == mode)[0])/len(points_sim_shuff_sim_shuff))
    fraction_exp_sim_sim_shuff.append(len(np.where(points_exp_sim_sim_shuff == mode)[0])/len(points_exp_sim_sim_shuff))
    fraction_sim_shuff.append(len(np.where(points_sim_shuff == mode)[0])/len(points_sim_shuff))

    entropy_exp =np.nansum(prob_exp*np.log(prob_exp), axis = 1)
    entropy_sim =np.nansum(prob_sim*np.log(prob_sim), axis = 1)
    entropy_sim_shuff =np.nansum(prob_sim_shuff*np.log(prob_sim_shuff), axis = 1)
    
    entropy_exp_sim_shuff =np.nansum(prob_exp_sim_shuff*np.log(prob_exp_sim_shuff), axis = 1)
    entropy_sim_sim_shuff =np.nansum(prob_sim_shuff_sim_shuff*np.log(prob_sim_shuff_sim_shuff), axis = 1)
    entropy_sim_shuff_sim_shuff =np.nansum(prob_sim_shuff_sim_shuff*np.log(prob_sim_shuff_sim_shuff), axis = 1)
    
    dist = np.linalg.norm(waves_exp[np.where(points_exp == mode)[0]] - gm.means_[mode], axis = 1)
    confidence_gm[0, mode] = np.mean(entropy_exp[np.where(points_exp == mode)[0]])
    dist = np.linalg.norm(waves_sim[np.where(points_sim == mode)[0]] - gm.means_[mode], axis = 1)
    confidence_gm[1, mode] = np.mean(entropy_sim[np.where(points_sim == mode)[0]])
    dist = np.linalg.norm(waves_sim_shuff[np.where(points_sim_shuff == mode)[0]] - gm.means_[mode], axis = 1)
    confidence_gm[2, mode] = np.mean(entropy_sim_shuff[np.where(points_sim_shuff == mode)[0]])
    confidence_gm_sim_shuff[0, mode] = np.mean(entropy_exp_sim_shuff[np.where(points_exp_sim_shuff == mode)[0]])
    confidence_gm_sim_shuff[1, mode] = np.mean(entropy_sim_sim_shuff[np.where(points_sim_sim_shuff == mode)[0]])
    confidence_gm_sim_shuff[2, mode] = np.mean(entropy_sim_shuff_sim_shuff[np.where(points_sim_shuff_sim_shuff == mode)[0]])

fraction_exp = np.array(fraction_exp)
fraction_exp_sim = np.array(fraction_exp_sim)
fraction_sim = np.array(fraction_sim)
fraction_exp_sim_shuff = np.array(fraction_exp_sim_shuff)
fraction_exp_sim_sim_shuff = np.array(fraction_exp_sim_sim_shuff)
fraction_sim_sim_shuff = np.array(fraction_sim_sim_shuff)
fraction_sim_shuff_sim_shuff = np.array(fraction_sim_shuff_sim_shuff)

sorted_gmm_idx = np.argsort(-fraction_exp)
sorted_gmm_shuff_idx = np.argsort(-fraction_exp_sim_shuff)


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

waves_sim_all = []
velocities_sim_all = []
directions_sim_all = []
isi_sim_all = []
grid_list_sim_all = np.empty([0, 9, 9])

for sim_idx, (a, p)  in enumerate(zip(a_arr, p_arr)):
    path_sim = path_sim_root + str(t_arr[sim_idx])  + '_'

    # load experimental data
    # STAGE 04 output
    fname_wave = path_sim + str(a) + '_' + str(p) + '/stage04_wave_detection/WaveHunt/waves.pkl'
    path_macro = path_sim + str(a) + '_' + str(p) + '/stage05_channel-wave_characterization/'

    block_w = load_input(fname_wave)#io_w.read_block()
    waves = [ev for ev in block_w.segments[0].events if ev.name == 'wavefronts'][0]
    grid_list_sim = ExtractWaves(waves, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50)
    grid_list_sim_all = np.concatenate([grid_list_sim_all, grid_list_sim])

    #Macroscopic observables
    velocities_df = pd.read_csv(path_macro + 'velocity_local/wavefronts_velocity_local.csv')
    isi_df = pd.read_csv(path_macro + 'inter_wave_interval_local/wavefronts_inter_wave_interval_local.csv')
    directions_df = pd.read_csv(path_macro + 'direction_local/wavefronts_direction_local.csv')

    # macro dimentions non-cropped
    velocities = np.array(velocities_df['velocity_local'])
    directions = np.array(np.arctan2(1./directions_df['direction_local_x_red'], 1./directions_df['direction_local_y_red']))
    directions = directions[~np.isnan(directions)]
    isi = np.array(isi_df['inter_wave_interval_local'])
    isi = isi[~np.isnan(isi)]

    velocities_sim_all.extend(velocities)
    directions_sim_all.extend(directions)
    isi_sim_all.extend(isi)
MacroObsSim = ComputeKDE(np.array(velocities_sim_all), np.array(directions_sim_all), np.array(isi_sim_all))

#############################################################################################
#                                                   FIGURE                                     #
#############################################################################################

xx = np.linspace(0, 1, 200)

from mpl_toolkits.axes_grid1 import make_axes_locatable


fig = plt.figure() #plt.subplots(3, 1, sharex = True)
fig.set_size_inches(9, 4.2, forward=True)
spec = fig.add_gridspec(ncols=7, nrows=2, wspace = 0.5, hspace = 0.4, height_ratios = [1.2,1.5])

################################################# plot macro
import matplotlib.gridspec as gridspec

subspec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[0,0:3], wspace = 0.2)

ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1], projection='polar'),
      fig.add_subplot(subspec[2])]


ax[0].tick_params(axis='both', which='major', labelsize=7)
ax[1].tick_params(axis='both', which='major', labelsize=7)
ax[2].tick_params(axis='both', which='major', labelsize=7)
PlotMacro(ax, MacroObsExp, MacroObsSim)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[0].set_box_aspect(1)
ax[2].set_box_aspect(1)

################################################# plot pies

subspec = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=spec[0,3:7], wspace = 0., hspace = 0)

ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1]),
      fig.add_subplot(subspec[2]),
      fig.add_subplot(subspec[3])]


ax[0].tick_params(axis='both', which='major', labelsize=7)
ax[1].tick_params(axis='both', which='major', labelsize=7)
ax[2].tick_params(axis='both', which='major', labelsize=7)
ax[3].tick_params(axis='both', which='major', labelsize=7)

colorscheme = ['#003f5c', '#ffa600', '#7a5195', '#ef5675']
labellist = ['mode #0', 'mode #1', 'mode #2', 'mode #3']

pie1 = ax[1].pie(fraction_exp[sorted_gmm_idx], colors = colorscheme, labels = labellist, counterclock = False, labeldistance = None, radius = 1.2)
ax[1].set_xlabel('exp', fontsize = 7.)
pie2 = ax[2].pie(fraction_sim[sorted_gmm_idx], colors = colorscheme, counterclock = False, labeldistance = None, radius = 1.2)
ax[2].set_xlabel('sim', fontsize = 7.)
pie1 = ax[0].pie(fraction_exp_sim[sorted_gmm_idx], colors = colorscheme, labels = labellist, counterclock = False, labeldistance = None, radius = 1.2)
ax[0].set_xlabel('exp + sim', fontsize = 7.)
#ax[1].legend([pie1], labels = labellist, fontsize = 7., loc=(-1.5,-0.7),  ncol = 4, handletextpad= 0.2, columnspacing = 1)

pie2 = ax[3].pie(fraction_sim_shuff_sim_shuff[sorted_gmm_shuff_idx], colors = colorscheme,
                  textprops=dict(fontsize=6, color = 'w'),
                  counterclock = False, labeldistance = None, radius = 1.2)
ax[3].set_xlabel('shuff sim', fontsize = 7.)

################################################# plot gmm
subspec = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=spec[1,0:7], wspace = 0.5)


ax = [fig.add_subplot(subspec[0]),
      fig.add_subplot(subspec[1]),
      fig.add_subplot(subspec[2]),
      fig.add_subplot(subspec[3])]

PlotWavePlatonic(fig, ax, gm.means_[sorted_gmm_idx]*var_all_exp_sim*0.5, points_exp, points_sim, mask_1d)
ax[0].set_title('mode #0', color = colorscheme[0], fontsize = 7.)
ax[1].set_title('mode #1', color = colorscheme[1], fontsize = 7.)
ax[2].set_title('mode #2', color = colorscheme[2], fontsize = 7.)
ax[3].set_title('mode #3', color = colorscheme[3], fontsize = 7.)

plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.savefig(save_path_fig + 'Figure6_' + str(mouseRec) + '.eps', format = 'eps')
plt.savefig(save_path_fig + 'Figure6_' + str(mouseRec) + '.png', dpi = 200)

plt.show()
plt.close()
