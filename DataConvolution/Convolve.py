import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as sio
from scipy import signal
import errno

import cmath
from numpy import matlib as mb
from numpy.polynomial.polynomial import polyval

####################################################################################
def lognormal(step, mu, sigma):
    return 1/(step*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(step) - mu)**2/(2*sigma))

def LogNormalFunction(TIME, mu_conv, sigma_conv):
    times=np.linspace(1, TIME,TIME)
    LogNormalFunc = [lognormal(t,mu_conv,sigma_conv) for t in times]
    median = np.exp(mu_conv-sigma_conv**2)
    LogNormalFunc[0] = 0

    return(LogNormalFunc)

####################################################################################

import argparse

CLI = argparse.ArgumentParser(description=__doc__,
			      formatter_class=argparse.RawDescriptionHelpFormatter)
CLI.add_argument("--load_dir", nargs='?', type=str, required=True,
		 help="load data path")
CLI.add_argument("--save_dir", nargs='?', type=str, required=True,
		 help="save convoluted data directory name")
args = CLI.parse_args()

# Setting dei path + loading dei parametri
home = os.getcwd()

load_path = str(args.load_dir)
try:
    os.makedirs(load_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

save_path_conv = home + '/../Output/ConvolvedData/'+ str(args.save_dir) + '/'
if not os.path.exists(save_path_conv):
    os.makedirs(save_path_conv)

#convolution parameters
sigma_conv = 0.91
mu_conv = 2.2 #2.2

TIME = 6000;
lognormalfunc = LogNormalFunction(TIME, mu_conv, sigma_conv)

for filename in os.listdir(load_path):
    try:
        Data = sio.loadmat(load_path + filename)
        # apply this to the whole dataset and save it in mat file
        #load output signal
        sampling_rate = 25 #hz
        signalarr = Data['NuE'] 
        signalconv_cut_arr = np.zeros([np.shape(signalarr)[0], np.shape(signalarr)[1]])

        for elem in range(len(signalarr)):
            Signal = signalarr[elem][:]
            signalconv = signal.convolve(Signal, lognormalfunc[0:len(Signal)])[0:len(Signal)]
            fs = 25  # sampling frequency
            # generate the time vector properly
            t=np.linspace(0,len(Signal),len(Signal))/fs
            fc = 12.  # cut-off frequency of the filter
            w = fc / (fs / 2) # normalize the frequency
            b, a = signal.butter(6, w, 'low')
            signalconv_cut = signal.filtfilt(b, a, signalconv[0:len(Signal)])
            signalconv_cut_arr[elem] = signalconv_cut
        
        sio.savemat(save_path_conv + filename, {'NuE': signalconv_cut_arr.T, 'x_pos_sel': Data['x_pos_sel'], 'y_pos_sel': Data['y_pos_sel']})
    except OSError:
        print(filename)
