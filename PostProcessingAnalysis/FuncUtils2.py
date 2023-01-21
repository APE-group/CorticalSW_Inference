import os
import pandas as pd
import numpy as np

import pickle
from sklearn.mixture import GaussianMixture
from skimage.measure import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
from scipy.special import i0
import sys
import scipy as sp

###############################################################################################
#                                   GMM ANALYSIS                                              #
###############################################################################################

def Compute_NumberOfGaussian(data, MAX_GAUSS_NUM):

    Pvalue = []
    
    for n_gauss in range(1, MAX_GAUSS_NUM):
        try:
            GM = (GaussianMixture(n_components=n_gauss, random_state = 0).fit(data))
            labels = GM.predict(data)
            
            ps = []
            w = []
            for i in range(0, n_gauss):
                idx = np.where(labels == i)[0]
                w.append(len(idx)/len(labels))
                ps.append(stats.normaltest(data[idx], axis = None).pvalue)
            Pvalue.append(stats.combine_pvalues(ps, 'stouffer', w)[1]) 
        except ValueError:
            print('VALUE ERROR')
    if Pvalue == []:
        Pvalue = [0]
    return(np.where(Pvalue == np.min(Pvalue))[0][-1]+1)

def vonmises_kde(data, kappa, n_bins=100):
    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    data = np.array(data)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde

def ComputePlatonicWavesAll(waves_exp, waves_inner, waves_sim, MAX_GAUSS_NUM=4, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50, return_mask = True, mask_idx = 1):
    
    #definisco lo zero dell'onda come il suo baicentro (media dei tempi relativi)
    baricentro = []
    grid_list_exp = np.empty([len(np.unique(waves_exp.labels)), DIM_X, DIM_Y])*np.nan
    for w_idx, w in enumerate(np.unique(waves_exp.labels)): #per ogni onda
        idx = np.where(np.array(waves_exp.labels) == w)[0]
        times = np.array(waves_exp.times)[idx]
        times = (times - np.mean(times))
        y_ch = np.array(waves_exp.array_annotations['channels'])[idx] // DIM_X
        x_ch = np.array(waves_exp.array_annotations['channels'])[idx] % DIM_X 
        for i in range(len(times)):
            grid_list_exp[w_idx, y_ch[i], x_ch[i]] = times[i]
    grid_list_reduced_exp = block_reduce(grid_list_exp, block_size=(1, ReduceFactor_x, ReduceFactor_y), func=np.nanmean, cval = np.nan)

    #definisco lo zero dell'onda come il suo baicentro (media dei tempi relativi)
    baricentro = []
    grid_list_inner = np.empty([len(np.unique(waves_inner.labels)), DIM_X, DIM_Y])*np.nan
    for w_idx, w in enumerate(np.unique(waves_inner.labels)): #per ogni onda
        idx = np.where(np.array(waves_inner.labels) == w)[0]
        times = np.array(waves_inner.times)[idx]
        times = (times - np.mean(times))
        y_ch = np.array(waves_inner.array_annotations['channels'])[idx] // DIM_X
        x_ch = np.array(waves_inner.array_annotations['channels'])[idx] % DIM_X 
        for i in range(len(times)):
            grid_list_inner[w_idx, y_ch[i], x_ch[i]] = times[i]
    grid_list_reduced_inner = block_reduce(grid_list_inner, block_size=(1, ReduceFactor_x, ReduceFactor_y), func=np.nanmean, cval = np.nan)

    #definisco lo zero dell'onda come il suo baicentro (media dei tempi relativi)
    baricentro = []
    grid_list_sim = np.empty([len(np.unique(waves_sim.labels)), DIM_X, DIM_Y])*np.nan
    for w_idx, w in enumerate(np.unique(waves_sim.labels)): #per ogni onda
        idx = np.where(np.array(waves_sim.labels) == w)[0]
        times = np.array(waves_sim.times)[idx]
        times = (times - np.mean(times))
        y_ch = np.array(waves_sim.array_annotations['channels'])[idx] // DIM_X
        x_ch = np.array(waves_sim.array_annotations['channels'])[idx] % DIM_X 
        for i in range(len(times)):
            grid_list_sim[w_idx, y_ch[i], x_ch[i]] = times[i]
    grid_list_reduced_sim = block_reduce(grid_list_sim, block_size=(1, ReduceFactor_x, ReduceFactor_y), func=np.nanmean, cval = np.nan)

    grid_list_reduced = np.concatenate([grid_list_reduced_exp, grid_list_reduced_inner, grid_list_reduced_sim])
    mask = grid_list_reduced[19]*0+1
    mask_1d = np.reshape(mask, np.shape(mask)[0]*np.shape(mask)[1])
    for i in range(0, len(grid_list_reduced)):
        grid_list_reduced[i] = grid_list_reduced[i]*mask


    grid_list_reduced_1d = np.reshape(grid_list_reduced, [len(grid_list_reduced), np.shape(grid_list_reduced)[1]*np.shape(grid_list_reduced)[2]])
    for w in grid_list_reduced_1d:
        w[np.isnan(w)] = -1 #-1000
        w *= mask_1d

    grid_list_reduced_1d = grid_list_reduced_1d[:, ~np.isnan(mask_1d)]
    idx = []
    for i, l in enumerate(grid_list_reduced_1d):
        if np.min(l) > -1000:
            idx.append(i)

    grid_list_reduced_1d = grid_list_reduced_1d[idx]
    X = grid_list_reduced_1d
    if len(X) <= 1:
        return 0, 0
    N_Comp = Compute_NumberOfGaussian(X, MAX_GAUSS_NUM)
    gm = (GaussianMixture(n_components=N_Comp, random_state=0).fit(X))
    points = gm.predict(X)
    classes, count = np.unique(points, return_counts=True)
    if return_mask:
        return gm, points
    else:
        return gm, points

def ComputePlatonicWaves(waves, file_name, num = 1, mask = [], MAX_GAUSS_NUM=4, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50, return_mask = True, mask_idx = 1):
    #definisco lo zero dell'onda come il suo baicentro (media dei tempi relativi)
    baricentro = []
    grid_list = np.empty([len(np.unique(waves.labels)), DIM_X, DIM_Y])*np.nan

    for w_idx, w in enumerate(np.unique(waves.labels)): #per ogni onda
        idx = np.where(np.array(waves.labels) == w)[0]
        times = np.array(waves.times)[idx]
        times = (times - np.mean(times))
        y_ch = np.array(waves.array_annotations['channels'])[idx] // DIM_X
        x_ch = np.array(waves.array_annotations['channels'])[idx] % DIM_X 
        for i in range(len(times)):
            grid_list[w_idx, y_ch[i], x_ch[i]] = times[i]

    grid_list_reduced = block_reduce(grid_list, block_size=(1, ReduceFactor_x, ReduceFactor_y), func=np.nanmean, cval = np.nan)

    if mask == []:
        mask = grid_list_reduced[mask_idx]*0+1
    mask_1d = np.reshape(mask, np.shape(mask)[0]*np.shape(mask)[1])

    for i in range(0, len(grid_list_reduced)):
        grid_list_reduced[i] = grid_list_reduced[i]*mask

    grid_list_reduced_1d = np.reshape(grid_list_reduced, [len(grid_list_reduced), np.shape(grid_list_reduced)[1]*np.shape(grid_list_reduced)[2]])
    for w in grid_list_reduced_1d:
        w[np.isnan(w)] = -1 #-1000
        w *= mask_1d

    grid_list_reduced_1d = grid_list_reduced_1d[:, ~np.isnan(mask_1d)]
    idx = []
    for i, l in enumerate(grid_list_reduced_1d):
        if np.min(l) > -1000:
            idx.append(i)

    grid_list_reduced_1d = grid_list_reduced_1d[idx]
    X = grid_list_reduced_1d
    if len(X) <= 1:
        return 0, 0
    N_Comp = Compute_NumberOfGaussian(X, MAX_GAUSS_NUM)
    gm = (GaussianMixture(n_components=N_Comp, random_state=0).fit(X))
    points = gm.predict(X)
    classes, count = np.unique(points, return_counts=True)
    if return_mask:
        return gm, points
    else:
        return gm, points

def interpolate_wave(array):

    from scipy import interpolate

    x = np.arange(0, 50, 1)
    y = np.arange(0, 50, 1)
    #z = np.reshape(grid_list[23], 50*50)
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='cubic')
    return GD1

def ComputeGMMWavesAll(grid_list_reduced, max_count =0,  mask_idx = 0, num = 1, mask = [], MAX_GAUSS_NUM=4, return_mask = True):

    if mask == []:
        mask = grid_list_reduced[mask_idx]*0+1
    mask_1d = np.reshape(mask, np.shape(mask)[0]*np.shape(mask)[1])

    for i in range(0, len(grid_list_reduced)):
        grid_list_reduced[i] = grid_list_reduced[i]*mask

    grid_list_reduced_1d = np.reshape(grid_list_reduced, [len(grid_list_reduced), np.shape(grid_list_reduced)[1]*np.shape(grid_list_reduced)[2]])
    for w in grid_list_reduced_1d:
        w[np.isnan(w)] = -1
        w = w*mask_1d
    grid_list_reduced_1d = grid_list_reduced_1d[:, ~np.isnan(mask_1d)]
    idx = []
    count = []
    for i, l in enumerate(grid_list_reduced_1d):
        count.append(len(np.where(l == -1)[0]))
        l[np.where(l == -1)[0]] = np.nan

    idx = np.where(np.array(count) <= max_count)[0]
    grid_list_reduced_1d = grid_list_reduced_1d[idx]
    
    new_idx = np.where(~np.isnan(np.mean(grid_list_reduced_1d, axis = 0)))[0]
    grid_list_reduced_1d = grid_list_reduced_1d[:,new_idx]
    mask_1d = np.reshape(mask, [9*9])
    mask_1d[np.where(mask_1d == 1)[0][new_idx]] = 2
    mask_1d[np.where(mask_1d == 1)[0]] = np.nan
    mask_1d[np.where(mask_1d == 2)[0]] = 1
    mask = np.reshape(mask_1d, [9, 9])
    
    X = grid_list_reduced_1d
    N_Comp = 3 #Compute_NumberOfGaussian(X, MAX_GAUSS_NUM)
    gm = (GaussianMixture(n_components=N_Comp, random_state=1).fit(X))
    points = gm.predict_proba(X)
    #classes, count = np.unique(points, return_counts=True)
    count = gm.weights_
    #gm.means_ = gm.means_[np.argsort(-count)]
    if return_mask:
        return gm, count, mask, X
    else:
        return gm, count, mask, new_idx, X


def ExtractWaves(waves, ReduceFactor_x = 6, ReduceFactor_y = 6, DIM_X = 50, DIM_Y = 50):

    baricentro = []
    grid_list = np.empty([len(np.unique(waves.labels)), DIM_X, DIM_Y])*np.nan

    for w_idx, w in enumerate(np.unique(waves.labels)): #per ogni onda
        idx = np.where(np.array(waves.labels) == w)[0]
        times = np.array(waves.times)[idx]
        times = (times - np.mean(times))
        y_ch = np.array(waves.array_annotations['channels'])[idx] // DIM_X
        x_ch = np.array(waves.array_annotations['channels'])[idx] % DIM_X 
        for i in range(len(times)):
            grid_list[w_idx, y_ch[i], x_ch[i]] = times[i]

    grid_list_reduced = block_reduce(grid_list, block_size=(1, ReduceFactor_x, ReduceFactor_y), func=np.nanmean, cval = np.nan)

    return(grid_list_reduced)

def ComputeExpWavesAll(grid_list_reduced, max_count =0,  mask_idx = 0, num = 1, mask = [], MAX_GAUSS_NUM=4, return_mask = True):

    if mask == []:
        mask = grid_list_reduced[mask_idx]*0+1
    mask_1d = np.reshape(mask, np.shape(mask)[0]*np.shape(mask)[1])

    for i in range(0, len(grid_list_reduced)):
        grid_list_reduced[i] = grid_list_reduced[i]*mask

    grid_list_reduced_1d = np.reshape(grid_list_reduced, [len(grid_list_reduced), np.shape(grid_list_reduced)[1]*np.shape(grid_list_reduced)[2]])
    for w in grid_list_reduced_1d:
        w[np.isnan(w)] = -1
        w = w*mask_1d
    grid_list_reduced_1d = grid_list_reduced_1d[:, ~np.isnan(mask_1d)]

    idx = []
    count = []
    for i, l in enumerate(grid_list_reduced_1d):
        count.append(len(np.where(l == -1)[0]))
        l[np.where(l == -1)[0]] = np.nan

    idx = np.where(np.array(count) <= max_count)[0]
    grid_list_reduced_1d = grid_list_reduced_1d[idx]
    
    new_idx = np.where(~np.isnan(np.mean(grid_list_reduced_1d, axis = 0)))[0]
    grid_list_reduced_1d = grid_list_reduced_1d[:,new_idx]
    mask_1d = np.reshape(mask, [9*9])
    mask_1d[np.where(mask_1d == 1)[0][new_idx]] = 2
    mask_1d[np.where(mask_1d == 1)[0]] = np.nan
    mask_1d[np.where(mask_1d == 2)[0]] = 1
    mask = np.reshape(mask_1d, [9, 9])
    
    X = grid_list_reduced_1d
    variance = np.std(X, axis = 0)
    N_Comp = 3 #Compute_NumberOfGaussian(X, MAX_GAUSS_NUM)
    gm = (GaussianMixture(n_components=N_Comp, random_state=1).fit(X))
    points = gm.predict(X)
    count = gm.weights_
    if return_mask:
        return gm, count, mask, variance, points
    else:
        return gm, count, variance, points
###############################################################################################
#                                   EMD DISTANCE ANALYSIS                                     #
###############################################################################################
def ComputeKDE(velocity, direction, isi, dimension = 100):

    MaxVel = np.log(70) #mm/s
    MaxISI = 4. #Hz

    #velocity
    velocity = np.log(velocity)
    velocity = velocity[np.isfinite(velocity)]
    kde_vel = stats.gaussian_kde(velocity)
    xx_vel = np.linspace(0, MaxVel, dimension)

    #directions
    direction = direction[np.isfinite(direction)]
    xx_dir, kde_dir = vonmises_kde(direction, 20, n_bins = dimension)
    
    #isi
    isi = isi[~np.isnan(isi)]
    kde_isi = stats.gaussian_kde(isi)
    xx_isi = np.linspace(0, MaxISI, dimension)
    
    MacroObs = {'kde_vel': kde_vel, 'xx_vel': xx_vel,
                'kde_dir': kde_dir, 'xx_dir': xx_dir,
                'kde_isi': kde_isi, 'xx_isi': xx_isi}
    return(MacroObs)


def EMD(a,b):
    earth = 0
    earth1 = 0
    
    s= min(len(a), len(b))
    su = []
    diff_array = a[0:s]-b[0:s]
    
    diff = 0
    for j in range (0,s):
        earth = (earth + diff_array[j])
        earth1= abs(earth)
        su.append(earth1)
    emd_output = sum(su)/(s-1)

    emd_output = emd_output#/np.sqrt(np.std(a)*np.std(b))
    
    return(emd_output)

def ComputeMacroEMD(vel_exp, vel_sim,
                    dir_exp, dir_sim, xx_dir,
                    isi_exp, isi_sim):

    MaxVel = np.log(70) #mm/s
    MaxISI = 4. #Hz

    #velocity
    EMD_vel = EMD(vel_exp, vel_sim)

    #directions
    diff = dir_exp[0:len(xx_dir)//2] + dir_exp[len(xx_dir)//2:]
    minimum = np.argmin(diff)
    emd_dir_exp = np.roll(dir_exp, -minimum)
    emd_dir_exp /= np.sum(emd_dir_exp)

    emd_dir_sim = np.roll(dir_sim, -minimum)
    emd_dir_sim /= np.sum(emd_dir_sim)

    EMD_dir = EMD(emd_dir_exp, emd_dir_sim)

    #isi
    EMD_isi = EMD(isi_exp, isi_sim)

    return(EMD_vel, EMD_dir, EMD_isi)

def ExtractMaxDonw(waves):

    baricentro = []
    for w_idx, w in enumerate(np.unique(waves.labels)): #per ogni onda
        idx = np.where(np.array(waves.labels) == w)[0]
        times = np.array(waves.times)[idx]

        baricentro.append(np.mean(times))
    baricentro = np.array(baricentro)
    baricentro = np.sort(baricentro)
    downstate = np.diff(baricentro)
    if len(np.unique(waves.labels)) == 1:
        return(np.nan,np.nan)
    else:
        return(np.max(downstate),np.std(downstate))

def ExtractWaveDuration(waves):

    duration = []
    for w_idx, w in enumerate(np.unique(waves.labels)): #per ogni onda
        idx = np.where(np.array(waves.labels) == w)[0]
        times = np.array(waves.times)[idx]

        duration.append(np.max(times)-np.min(times))
    duration = np.array(duration)
    duration =  np.sort(duration)
    return(np.percentile(duration, 90))

def MyCombineEMD(EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num):

    EMD_combination = np.sqrt((EMD_Vel**2 +  EMD_Dir**2 + EMD_Isi**2))*EMD_Num**2
    return(EMD_combination)

def MyBestAndWorst(EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num):

    EMD_Vel = sp.ndimage.gaussian_filter(EMD_Vel, sigma=1., order = 0, truncate = 2., mode = 'constant', cval = 0)#np.nanmean(EMD_Vel)+ 0.05)

    # moltiplico i borsi per i punti mancanti
    EMD_Vel[:, 0] = EMD_Vel[:,0]  *13/9.
    EMD_Vel[:, -1] = EMD_Vel[:,-1]*13/9.
    EMD_Vel[0, :] = EMD_Vel[0,:]  *13/9.
    EMD_Vel[-1,:] = EMD_Vel[-1,:] *13/9.
    
    EMD_Vel[:, 1]  = EMD_Vel[:,1] *13/12.
    EMD_Vel[:, -2] = EMD_Vel[:,-2]*13/12.
    EMD_Vel[1, :]  = EMD_Vel[1,:] *13/12.
    EMD_Vel[-2,:]  = EMD_Vel[-2,:]*13/12.
    EMD_Dir = sp.ndimage.gaussian_filter(EMD_Dir, sigma=1., order = 0, truncate = 2., mode = 'constant', cval = 0)#np.nanmean(EMD_Dir)+0.05)
    EMD_Dir[:, 0]  = EMD_Dir[:,0] *13/9.
    EMD_Dir[:, -1] = EMD_Dir[:,-1]*13/9.
    EMD_Dir[0, :]  = EMD_Dir[0,:] *13/9.
    EMD_Dir[-1,:]  = EMD_Dir[-1,:]*13/9.
    
    EMD_Dir[:, 1]  = EMD_Dir[:,1] *13/12.
    EMD_Dir[:, -2] = EMD_Dir[:,-2]*13/12.
    EMD_Dir[1, :]  = EMD_Dir[1,:] *13/12.
    EMD_Dir[-2,:]  = EMD_Dir[-2,:]*13/12.
    
    EMD_Isi = sp.ndimage.gaussian_filter(EMD_Isi, sigma=1., order = 0, truncate = 2., mode = 'constant', cval = 0)#np.nanmean(EMD_Isi)+0.05)
    EMD_Isi[:, 0]  = EMD_Isi[:,0] *13/9.
    EMD_Isi[:, -1] = EMD_Isi[:,-1]*13/9.
    EMD_Isi[0, :]  = EMD_Isi[0,:] *13/9.
    EMD_Isi[-1,:]  = EMD_Isi[-1,:]*13/9.

    EMD_Isi[:, 1]  = EMD_Isi[:,1] *13/12.
    EMD_Isi[:, -2] = EMD_Isi[:,-2]*13/12.
    EMD_Isi[1, :]  = EMD_Isi[1,:] *13/12.
    EMD_Isi[-2,:]  = EMD_Isi[-2,:]*13/12.
    EMD_combination = MyCombineEMD(EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num)

    return(EMD_combination, EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num)

def MyComputeOptimalValue(t_arr, home, TopoRec, save_data_path, conv_data_path):

    a_arr = []
    p_arr = []

    for t_idx, t in enumerate(t_arr):
        # Read amplitude and period values
        Period = []
        Amplitude = []

        for filename in os.listdir(conv_data_path + str(t) + '_Ampl_Period'):
             
            tmp = filename.split("_")
            idx = tmp.index("DAMP")
            Amplitude.append(tmp[idx+1])

            idx = tmp.index("PERIOD")
            Period.append(tmp[idx+1])

        Amplitude = np.unique(Amplitude)
        Period = np.unique(Period)
        # load emd
        
        import scipy as sp
        EMD_Vel = np.load(save_data_path+ str(t) + '_EMD_vel.npy')
        EMD_Dir = np.load(save_data_path+ str(t) + '_EMD_dir.npy')
        EMD_Isi = np.load(save_data_path+ str(t) + '_EMD_isi.npy')
        EMD_Num = np.load(save_data_path+ str(t) + '_EMD_num.npy')
        EMD_combination = MyCombineEMD(EMD_Vel, EMD_Dir, EMD_Isi, EMD_Num)
        BestIdx = np.unravel_index(np.nanargmin(EMD_combination), EMD_combination.shape)
        a = Amplitude[BestIdx[0]]
        p = Period[BestIdx[1]]
        a_arr.append(a)
        p_arr.append(p)
    return(a_arr,p_arr)

from types import SimpleNamespace
simple_3x3 = SimpleNamespace(
             x=np.array([[-0, 0, 0],
                         [-1, 0, 1],
                         [-0, 0, 0]], dtype=float),
             y=np.array([[-0, -1, -0],
                         [ 0,  0,  0],
                         [ 0,  1,  0]], dtype=float))
def get_kernel(kernel_name):
    if kernel_name.lower() in ['simple_3x3', 'simple']:
        return simple_3x3
    elif kernel_name.lower() in ['prewitt_3x3', 'prewitt']:
        return prewitt_3x3
    elif kernel_name.lower() in ['scharr_3x3', 'scharr']:
        return scharr_3x3
    elif kernel_name.lower() in ['sobel_3x3', 'sobel']:
        return sobel_3x3
    elif kernel_name.lower() in ['sobel_5x5']:
        return sobel_5x5
    elif kernel_name.lower() in ['sobel_7x7']:
        return sobel_7x7
    else:
        warnings.warn(f'Deriviative name {kernel_name} is not implemented, '
                     + 'using sobel filter instead.')
        return sobel_3x3


def nan_conv2d(frame, kernel, kernel_center=None):
    dx, dy = kernel.shape
    dimx, dimy = frame.shape
    dframe = np.empty((dimx, dimy))*np.nan

    if kernel_center is None:
        kernel_center = [int((dim-1)/2) for dim in kernel.shape]

    # inverse kernel to mimic behavior or regular convolution algorithm
    k = kernel[::-1, ::-1]
    ci = dx - 1 - kernel_center[0]
    cj = dy - 1 - kernel_center[1]

    # loop over each frame site
    for i,j in zip(*np.where(np.isfinite(frame))):
        site = frame[i,j]

        # loop over kernel window for frame site
        window = np.zeros((dx,dy), dtype=float)*np.nan
        for di,dj in itertools.product(range(dx), range(dy)):

            # kernelsite != 0, framesite within borders and != nan
            if k[di,dj] and 0 <= i+di-ci < dimx and 0 <= j+dj-cj < dimy \
                        and np.isfinite(frame[i+di-ci,j+dj-cj]):
                sign = -1*np.sign(k[di,dj])
                window[di,dj] = sign * (site - frame[i+di-ci,j+dj-cj])

        xi, yi = np.where(np.logical_not(np.isnan(window)))
        if np.sum(np.logical_not(np.isnan(window))) > dx*dy/10:
            dframe[i,j] = np.average(window[xi,yi], weights=abs(k[xi,yi]))
    return dframe
import itertools
def ComputeVelocityMacro(grid, kernel_name = 'simple_3x3', spatial_scale=0.04): #0.06):
    
    spatial_derivative_df = pd.DataFrame()
    velocity_collection = []

    for wave_id in range(len(grid)):
        kernel = get_kernel(kernel_name)
        d_vertical = -1 * nan_conv2d(grid[wave_id], kernel.x)
        d_horizont = -1 * nan_conv2d(grid[wave_id], kernel.y)
        x_coords, y_coords = np.where(~np.isnan(d_vertical))


        dt_x = d_vertical[x_coords, y_coords]
        dt_y = d_horizont[x_coords, y_coords]

        velocity = spatial_scale * np.sqrt(1/(dt_x**2 + dt_y**2))
        velocity[~np.isfinite(velocity)] = np.nan
    velocity_collection.extend(list(velocity))
    collection = velocity_collection.copy()
    print('MEDIA', np.nanmean(collection))
    MaxVel = np.log(70) #mm/s
    velocity_collection= np.log(velocity_collection)
    velocity_collection = velocity_collection[np.isfinite(velocity_collection)]
    kde_vel_macro = stats.gaussian_kde(velocity_collection)
    xx_vel = np.linspace(0, MaxVel, 1000)
    return(kde_vel_macro, xx_vel, collection)

###############################################################################################
#                                   PLOT FUNCTIONS                                            #
###############################################################################################

def PlotWavePlatonic(fig, axs, platonic, points_exp, points_sim, mask_1d):
    
    for mode in range(len(platonic)):
        # onda platonica
        axs[mode].tick_params(axis='both', which='major', labelsize=7)
        axs[mode].set_xticks([])
        axs[mode].set_yticks([])

        grid = np.empty(np.shape(mask_1d))*np.nan
        grid[~np.isnan(mask_1d)] = platonic[mode]
        grid = np.reshape(grid, [9,9])

        im = axs[mode].imshow(1000*grid, cmap = 'RdBu_r', aspect='equal')
        divider = make_axes_locatable(axs[mode])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=7.)
        if mode == len(platonic)-1:
            cb.ax.set_ylabel(r'$\Delta t (s)$', rotation=270, fontsize = 7.)
        fraction_exp = len(np.where(points_exp == mode)[0])/len(points_exp)
        fraction_sim = len(np.where(points_sim == mode)[0])/len(points_sim)
def PlotMacro(ax, MacroObsExp, MacroObsSim,):

    color = 'black'
    temp = MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])/np.sum(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])*np.diff(np.insert(np.exp(MacroObsExp['xx_vel']), 0, 0)))
    ax[0].plot(np.exp(MacroObsExp['xx_vel']), MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])/np.sum(MacroObsExp['kde_vel'](MacroObsExp['xx_vel'])*np.diff(np.insert(np.exp(MacroObsExp['xx_vel']), 0, 0))), color = color, linewidth = 0.5, label = 'exp')
    ax[1].plot(MacroObsExp['xx_dir']-np.pi/2., MacroObsExp['kde_dir']/np.sum(MacroObsExp['kde_dir']), color = color, linewidth = 1.)
    ax[2].plot(MacroObsExp['xx_isi'], MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])/np.sum(MacroObsExp['kde_isi'](MacroObsExp['xx_isi'])), color = color, linewidth = 0.5, label = 'exp')
    
    color = 'red'
    ax[0].plot(np.exp(MacroObsSim['xx_vel']), MacroObsSim['kde_vel'](MacroObsSim['xx_vel'])/np.sum(MacroObsSim['kde_vel'](MacroObsSim['xx_vel'])*np.diff(np.insert(np.exp(MacroObsSim['xx_vel']), 0, 0))), color = color, linewidth = 0.5, label = 'sim')
    #ax[0].plot(np.exp(MacroObsSim['xx_vel']), MacroObsSim['kde_vel'](MacroObsSim['xx_vel'])/np.sum(MacroObsSim['kde_vel'](MacroObsSim['xx_vel'])), color = color, linewidth = 0.5)
    ax[1].plot(MacroObsSim['xx_dir']-np.pi/2., MacroObsSim['kde_dir']/np.sum(MacroObsSim['kde_dir']), color = color, linewidth = 1.)
    ax[2].plot(MacroObsSim['xx_isi'], MacroObsSim['kde_isi'](MacroObsSim['xx_isi'])/np.sum(MacroObsSim['kde_isi'](MacroObsSim['xx_isi'])), color = color, linewidth = 0.5, label = 'sim')

    ax[0].set_xlabel('velocities (mm/s)', fontsize = 7.)
    ax[1].set_xlabel('directions (deg)', fontsize = 7.)
    ax[2].set_xlabel('iwi (s)', fontsize = 7.)
    ax[0].set_ylabel('density', fontsize = 7.)
    #ax[2].set_ylabel('density', fontsize = 7.)
    ax[1].set_title('density', fontsize = 7.)
    ax[1].set_xticks(np.pi/180. * np.linspace(0,  360, 8, endpoint=False))

    ax[2].set_xlim([0, 4])
    #ax[0].set_ylim([0, 0.032])
    ax[2].set_ylim([0, 0.032])
    ax[2].set_yticks([])
    ax[0].legend(fontsize = 7.)#, handletextpad = 0.2)
    #ax[1].set_xticks([])
    ax[1].set_yticks([])
   
    return(ax)

