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

#############15 frames without blank ##########
string_path = '/predictions/Anton/'
#data is now of shape (num_images, num_noise, num_frames, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
labels = np.load(wd + string_path + 'label/Bern.npy')
Spike_frames_Bern = get_neurons_of_area(np.load(wd + string_path + 'frames/Bern.npy'), labels)
Spike_frames_Gaus = get_neurons_of_area(np.load(wd + string_path + 'frames/Gaus_10.npy'), labels)

#Now substitute all values > 100 to NaN
Spike_frames_Bern[Spike_frames_Bern > 100] = np.nan
Spike_frames_Gaus[Spike_frames_Gaus > 100] = np.nan

#sum over frames to get total spikes per trial
Spike_Bern = np.nansum(Spike_frames_Bern, axis=2)  #shape (num_images, num_noise, num_neurons)
Spike_Gaus = np.nansum(Spike_frames_Gaus, axis=2)  #shape (num_images, num_noise, num_neurons)

###mean,variance, fano factor
Mean_frames_Bern = np.nanmean(Spike_frames_Bern, axis=1)  #mean across noise for each image and neuron and frame
Var_frames_Bern = np.nanvar(Spike_frames_Bern, axis=1)  #variance across noise for each image and neuron and frame
Mean_frames_Gaus = np.nanmean(Spike_frames_Gaus, axis=1)  #mean across noise for each image and neuron and frame
Var_frames_Gaus = np.nanvar(Spike_frames_Gaus, axis=1)  #variance across noise for each image and neuron and frame
FF_frames_Bern = Var_frames_Bern / (Mean_frames_Bern + 1e-10)  #fano factor across noise for each image and neuron and frame
FF_frames_Gaus = Var_frames_Gaus / (Mean_frames_Gaus + 1e-10)  #fano factor across noise for each image and neuron and frame

###
Mean_Bern = np.nanmean(Spike_Bern, axis=1)  #mean across noise for each image and neuron
Var_Bern = np.nanvar(Spike_Bern, axis=1)  #variance across noise for each image and neuron
Mean_Gaus = np.nanmean(Spike_Gaus, axis=1)  #mean across noise for each image and neuron
Var_Gaus = np.nanvar(Spike_Gaus, axis=1)  #variance across noise for each image and neuron


#compute linear slope of Var - Mean regression without the offset for each neuron and frame. Mean and Var are of shape (num_images,  num_frames, num_neurons)
slopes_Bern, r2_Bern = compute_slope_var_mean(Mean_frames_Bern[], Var_frames_Bern)
slopes_Gaus, r2_Gaus = compute_slope_var_mean(Mean_frames_Gaus, Var_frames_Gaus)

#for sum over frames
slopes_Bern_sum, r2_Bern_sum = compute_slope_var_mean(Mean_Bern[:, np.newaxis, :], Var_Bern[:, np.newaxis, :])
slopes_Gaus_sum, r2_Gaus_sum = compute_slope_var_mean(Mean_Gaus[:, np.newaxis, :], Var_Gaus[:, np.newaxis, :])


global_res_Bern = compute_global_frame_fits(Mean_frames_Bern, Var_frames_Bern)
global_res_Gaus = compute_global_frame_fits(Mean_frames_Gaus, Var_frames_Gaus)

a_neurons_Bern, b_neurons_Bern, r2_neurons_Bern = compute_neuron_fits(Mean_frames_Bern, Var_frames_Bern, global_res_Bern)
a_neurons_Gaus, b_neurons_Gaus, r2_neurons_Gaus = compute_neuron_fits(Mean_frames_Gaus, Var_frames_Gaus, global_res_Gaus)


cor_Bern = [get_cor_zscored(Spike_frames_Bern[:,:,i,:])[0] for i in range(Spike_frames_Bern.shape[2])]
cor_Bern = np.array(cor_Bern)
cor_Gaus = [get_cor_zscored(Spike_frames_Gaus[:,:,i,:])[0] for i in range(Spike_frames_Gaus.shape[2])]
cor_Gaus = np.array(cor_Gaus)

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

# 6. Mean Correlation Coefficient
plot_single_axis_boxplot(axes[2,1], cor_Bern, cor_Gaus,
                         title="Noise correlation (z-scored responses)",
                         ylabel_suffix="Noise correlation",
                         legend_loc='best', yline=0)

plt.tight_layout()
#plt.show()

#save as pdf
plt.savefig(wd + '/plots/Anton/var_across_frames.pdf')

###plot fano factor slopes only
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
plot_dual_axis_boxplot(axes, slopes_Bern_sum, slopes_Gaus_sum,
                       title="Fano Factor",
                       ylim=0, whisk=(25, 50), scale = 2, xlabel = "", ylabel_suffix = "")
plt.tight_layout()
plt.savefig(wd + '/plots/Anton/FF.pdf')



###10 blank frames in front####
string_path = '/predictions/Anton/Blank_image_in_front/'
#no Gaussian
labels = np.load(wd + string_path + 'label/Bern_blank_10.npy')
Spike_frames_Bern_blank = get_neurons_of_area(np.load(wd + string_path + 'frames/Bern_blank_10.npy'), labels)
#Now substitute all values > 100 to NaN
Spike_frames_Bern_blank[Spike_frames_Bern_blank > 100] = np.nan
###mean,variance, fano factor
Mean_frames_Bern_blank = np.nanmean(Spike_frames_Bern_blank, axis=1)  #mean across noise for each image and neuron and frame
Var_frames_Bern_blank = np.nanvar(Spike_frames_Bern_blank, axis=1)  #variance across noise for each image and neuron and frame
FF_frames_Bern_blank = Var_frames_Bern_blank / (Mean_frames_Bern_blank + 1e-10)  #fano factor across noise for each image and neuron and frame
#compute linear slope of Var - Mean regression without the offset for each neuron and frame. Mean and Var are of shape (num_images,  num_frames, num_neurons)
slopes_Bern_blank, r2_Bern_blank = compute_slope_var_mean(Mean_frames_Bern_blank, Var_frames_Bern_blank)
global_res_Bern_blank = compute_global_frame_fits(Mean_frames_Bern_blank, Var_frames_Bern_blank)
a_neurons_Bern_blank, b_neurons_Bern_blank, r2_neurons_Bern_blank = compute_neuron_fits(Mean_frames_Bern_blank, Var_frames_Bern_blank, global_res_Bern_blank)
cor_Bern_blank = [get_cor_zscored(Spike_frames_Bern_blank[:,:,i,:])[0] for i in range(Spike_frames_Bern_blank.shape[2])]
cor_Bern_blank = np.array(cor_Bern_blank)

#add Bern data instead of Gaussian just to test
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
# 1. Linear Slope (Fano Factor estimate)
plot_dual_axis_boxplot(axes[0,0], slopes_Bern_blank, slopes_Bern_blank,
                       title="Fano Factor: Linear Slope of Var-Mean Regression",
                       ylim=0)
# 2. Linear Regression R^2
plot_single_axis_boxplot(axes[0,1], r2_Bern_blank, r2_Bern_blank,
                         title="Variance Explained: Linear Var-Mean Regression",
                         ylabel_suffix="Variance Explained ($R^2$)",
                         legend_loc='best')
# 3. Power Law Parameter 'a'
plot_dual_axis_boxplot(axes[1,0], a_neurons_Bern_blank, a_neurons_Bern_blank,
                       title="Scaling Parameter $a$ (Power Law Fit)",
                       ylabel_suffix="Parameter $a$",
                       ylim=0)
# 4. Power Law R^2
plot_single_axis_boxplot(axes[1,1], r2_neurons_Bern_blank, r2_neurons_Bern_blank,
                         title="Variance Explained: Power Law Fit",
                         ylabel_suffix="Variance Explained ($R^2$)",
                         legend_loc='best')
# 5. Power Law Parameter 'b'
plot_single_axis_boxplot(axes[2,0], b_neurons_Bern_blank, b_neurons_Bern_blank,
                         title="Exponent Parameter $b$ (Power Law Fit)",
                         ylabel_suffix="Parameter $b$",
                         ylim=0,
                         legend_loc='best')
# 6. Mean Correlation Coefficient
plot_single_axis_boxplot(axes[2,1], cor_Bern_blank, cor_Bern_blank,
                         title="Noise correlation (z-scored responses)",
                         ylabel_suffix="Noise correlation",
                         legend_loc='best', yline=0)
plt.tight_layout()


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
plot_single_axis_boxplot(axes, slopes_Bern_blank, None,
                       title="Fano Factor: Linear Slope of Var-Mean Regression \n starting with 10 Blank Grey Frames",
                       ylim=0, xline = 9.5, whisk = (25,75), legend_loc='best')
plt.tight_layout()

#save
plt.savefig(wd + '/plots/Anton/FF_across_frames_blank_10.pdf')
