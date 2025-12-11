#here we do similar analisys as for the whole trial but individually for each frame
%matplotlib qt
import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
from functions_Anton import *
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit




############Start#################################
##load the data
wd = os.getcwd()
string_path = '/predictions/Anton/'


#data is now of shape (num_images, num_noise, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
labels = np.load(wd + string_path + 'label/Bern.npy')
Spike_frames_Bern = get_neurons_of_area(np.load(wd + string_path + 'frames/Bern.npy'), labels)
Spike_frames_Gaus = get_neurons_of_area(np.load(wd + string_path + 'frames/Gaus_10.npy'), labels)

#Now substitute all values > 100 to NaN
Spike_frames_Bern[Spike_frames_Bern > 100] = np.nan
Spike_frames_Gaus[Spike_frames_Gaus > 100] = np.nan

###mean,variance, fano factor
Mean_frames_Bern = np.nanmean(Spike_frames_Bern, axis=1)  #mean across noise for each image and neuron and frame
Var_frames_Bern = np.nanvar(Spike_frames_Bern, axis=1)  #variance across noise for each image and neuron and frame
Mean_frames_Gaus = np.nanmean(Spike_frames_Gaus, axis=1)  #mean across noise for each image and neuron and frame
Var_frames_Gaus = np.nanvar(Spike_frames_Gaus, axis=1)  #variance across noise for each image and neuron and frame
FF_frames_Bern = Var_frames_Bern / (Mean_frames_Bern + 1e-10)  #fano factor across noise for each image and neuron and frame
FF_frames_Gaus = Var_frames_Gaus / (Mean_frames_Gaus + 1e-10)  #fano factor across noise for each image and neuron and frame

#compute linear slope of Var - Mean regression without the offset for each neuron and frame. Mean and Var are of shape (num_images,  num_frames, num_neurons)
slopes_Bern, r2_Bern = compute_slope_var_mean(Mean_frames_Bern, Var_frames_Bern)
slopes_Gaus, r2_Gaus = compute_slope_var_mean(Mean_frames_Gaus, Var_frames_Gaus)


global_res_Bern = compute_global_frame_fits(Mean_frames_Bern, Var_frames_Bern)
global_res_Gaus = compute_global_frame_fits(Mean_frames_Gaus, Var_frames_Gaus)

a_neurons_Bern, b_neurons_Bern, r2_neurons_Bern = compute_neuron_fits(Mean_frames_Bern, Var_frames_Bern, global_res_Bern)
a_neurons_Gaus, b_neurons_Gaus, r2_neurons_Gaus = compute_neuron_fits(Mean_frames_Gaus, Var_frames_Gaus, global_res_Gaus)



###plots
# Plot FF across frames for both types of noise
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

# 1. Linear Slope (Fano Factor estimate)
plot_dual_axis_boxplot(axes[0,0], slopes_Bern, slopes_Gaus,
                       title="Fano Factor: Linear Slope of Var-Mean Regression",
                       ylim=0)

# 2. Linear Regression R^2
plot_single_axis_boxplot(axes[0,1], r2_Bern, r2_Gaus,
                         title="Variance Explained: Linear Var-Mean Regression",
                         ylabel_suffix="Variance Explained ($R^2$)",
                         legend_loc='best')

# 3. Power Law Parameter 'a'
plot_dual_axis_boxplot(axes[1,0], a_neurons_Bern, a_neurons_Gaus,
                       title="Scaling Parameter $a$ (Power Law Fit)",
                       ylabel_suffix="Parameter $a$",
                       ylim=0)

# 4. Power Law R^2
plot_single_axis_boxplot(axes[1,1], r2_neurons_Bern, r2_neurons_Gaus,
                         title="Variance Explained: Power Law Fit",
                         ylabel_suffix="Variance Explained ($R^2$)",
                         legend_loc='best')

# 5. Power Law Parameter 'b'
plot_single_axis_boxplot(axes[2,0], b_neurons_Bern, b_neurons_Gaus,
                         title="Exponent Parameter $b$ (Power Law Fit)",
                         ylabel_suffix="Parameter $b$",
                         ylim=0,
                         legend_loc='best')

axes[2,1].axis('off')

plt.tight_layout()
#plt.show()

#save as pdf
plt.savefig(wd + '/plots/Anton/var_across_frames.pdf')