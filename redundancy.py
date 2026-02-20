#here we do similar analisys as for the whole trial but individually for each frame
#need higher number of trials = 1000 for 10 images
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from scipy import linalg

from functions_Anton import *




############Start#################################
##load the data
wd = os.getcwd()
ADD_Poisson = True  # Whether to add Poisson on top

if ADD_Poisson:
    file_name_add = "_with_Poisson"
else:
    file_name_add = ""
####################1000 trials###################################
path = wd + '/predictions/Anton/1000_trials/'
labels = np.load(path + 'label/Bern_1000.npy')
Spike_Bern_1000 = get_neurons_of_area(np.load(path + 'sum/Bern_1000.npy'), labels)
Spike_Gaus_1000 = get_neurons_of_area(np.load(path + 'sum/Gaus_10_1000.npy'), labels)

if ADD_Poisson:
    Spike_Bern_1000 = np.random.poisson(lam=Spike_Bern_1000)
    Spike_Gaus_1000 = np.random.poisson(lam=Spike_Gaus_1000)

#set seed for reproducibility
np.random.seed(42)

nlist = [2, 5, 10, 20, 30, 50, 100, 200, 300, 500]
FI_Bern_1000, FI_shuf_Bern_1000, N_Bern_1000, Image_pairs_1000 = compute_fisher_info_all(Spike_Bern_1000, n_image_pairs= 10, n_repeats=1000, n_list=nlist)

Redundancy_Bern_1000 = 1 - (FI_Bern_1000 / FI_shuf_Bern_1000)
Redundancy_Bern_1000_abs = FI_shuf_Bern_1000 - FI_Bern_1000
#now Guassian
FI_Gaus_1000, FI_shuf_Gaus_1000, N_Gaus_1000, _ = compute_fisher_info_all(Spike_Gaus_1000, n_image_pairs= 10, n_repeats=1000,
                                                                          n_list=nlist, image_pairs=Image_pairs_1000)

Redundancy_Gaus_1000 = 1 - (FI_Gaus_1000 / FI_shuf_Gaus_1000)

#load pixel space
#Bernoulli
Pixel_space_Bern = np.load(wd + '/predictions/Anton/1000_trials/all_images_noise_Bern.npy')
#flatten last two dimensions
Pixel_space_Bern = Pixel_space_Bern.reshape(Pixel_space_Bern.shape[0], Pixel_space_Bern.shape[1], -1)
FI_Pixel_Bern,_,_,_ = compute_fisher_info_all(Pixel_space_Bern,  n_repeats=1, n_list=[Pixel_space_Bern.shape[2]], image_pairs=Image_pairs_1000, mode = "pixels")

#Gaussian
Pixel_space_Gaus = np.load(wd + '/predictions/Anton/1000_trials/all_images_noise_Gaussian.npy')
#flatten last two dimensions
Pixel_space_Gaus = Pixel_space_Gaus.reshape(Pixel_space_Gaus.shape[0], Pixel_space_Gaus.shape[1], -1)
FI_Pixel_Gaus,_,_,_ = compute_fisher_info_all(Pixel_space_Gaus,  n_repeats=1, n_list=[Pixel_space_Gaus.shape[2]], image_pairs=Image_pairs_1000, mode = "pixels")


#
#calculate normalizing FI for FI in pixel space
FI_Bern_1000_norm = FI_Bern_1000 / FI_Pixel_Bern[:, np.newaxis, np.newaxis]
FI_Gaus_1000_norm = FI_Gaus_1000 / FI_Pixel_Gaus[:, np.newaxis, np.newaxis]


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
plot_scaling_metric(FI_Bern_1000_norm, N_Bern_1000, aggregation='median', color = "tab:orange",
                    title="Fisher Info, Bernoulli noise",  ylabel = "FI (normalized)" , ylim = 0, ax= axes[0,0], scale = 1.5)
#Gaus
plot_scaling_metric(FI_Gaus_1000_norm, N_Gaus_1000, aggregation='median', color = "tab:blue",
                    title="Gaussian noise", ylabel = "" , ylim = 0, ax= axes[0,1], scale = 1.5)
#Redundancy Bern
plot_scaling_metric(100*Redundancy_Bern_1000, N_Bern_1000, aggregation='median', color = "tab:orange" , title="Information Redundancy (relative)",
                    ylim = 0, ylabel = "Redundant Information,%", ax= axes[1,0], scale = 1.5)

#Redundancy Gaus
plot_scaling_metric(100*Redundancy_Gaus_1000, N_Gaus_1000, aggregation='median', color = "tab:blue" , title="",
                    ylim = 0, ylabel = "", ax= axes[1,1], scale = 1.5)
#Redundancy Bern absolute
# plot_scaling_metric(Redundancy_Bern_1000_abs, N_Bern_1000, aggregation='median', color = "tab:orange" , title="Information Redundancy (absolute)",
#                     yline = 0, ylabel = "Redundant Information (abs. units)", ax= axes[2])

plt.tight_layout()
plt.savefig(wd +  '/plots/Anton/Fisher_Info_and_Redundancy_1000_trials'+file_name_add +'.pdf')




# fig, axes = plt.subplots(figsize=(1.2*8, 1.2*4))
# plot_scaling_metric(FI_Bern_1000/100, N_Bern_1000, aggregation='median', color = "tab:orange",
#                     title="Fisher Info in pixel space", ylim = 0, ax= axes, FI_pixel= FI_Pixel_Bern/100, scale = 1.5)
# plt.tight_layout()
# plt.savefig(wd +  '/plots/Anton/Fisher_Info_with_Pixel_1000_trials.pdf')









################100 trials predictions########################
string_path = '/predictions/Anton/'


#data is now of shape (num_images, num_noise, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
labels = np.load(wd + string_path + 'label/Bern.npy')
# Spike_frames_Bern = get_neurons_of_area(np.load(wd + string_path + 'frames/Bern.npy'), labels)
# Spike_frames_Gaus = get_neurons_of_area(np.load(wd + string_path + 'frames/Gaus_10.npy'), labels)

Spike_Bern = get_neurons_of_area(np.load(wd + string_path + 'sum/Bern.npy'), labels)
Mean_Bern = get_neurons_of_area(np.load(wd + string_path + 'mean/Bern.npy'), labels)
# Spike_Gaus = get_neurons_of_area(np.load(wd + string_path + 'sum/Gaus_10.npy'), labels)
# Mean_Gaus = get_neurons_of_area(np.load(wd + string_path + 'mean/Gaus_10.npy'), labels)

#Now substitute all values > 100 to NaN
# Spike_frames_Bern[Spike_frames_Bern > 100] = np.nan
# Spike_frames_Gaus[Spike_frames_Gaus > 100] = np.nan

#Now substitute all values > 100 to NaN
Spike_Bern[Spike_Bern > 100] = np.nan
Mean_Bern[Mean_Bern > 100] = np.nan
# Spike_Gaus[Spike_Gaus > 100] = np.nan
# Mean_Gaus[Mean_Gaus > 100] = np.nan

#compute fisher information


# FI_Bern = compute_fisher_info(Spike_Bern[1:3,:,0:100])
# FI_Bern_shuf = compute_fisher_info(Spike_Bern[1:3,:,0:100], shuffle=True)

#nlist integers from 2 to 50 by 1

#generate tasks
actual_n_pairs = 100
used_image_pairs = np.zeros((actual_n_pairs, 2), dtype=int)
for k in range(actual_n_pairs):
            used_image_pairs[k] = np.random.choice(Spike_Bern.shape[0], 2, replace=False)


FI_Bern, FI_shuf_Bern, N_Bern, Image_pairs = compute_fisher_info_all(Spike_Bern, n_image_pairs= 100, n_repeats=1000, image_pairs=used_image_pairs)
FI_Gaus, FI_shuf_Gaus, N_Gaus, Image_pairs = compute_fisher_info_all(Spike_Gaus, n_image_pairs= 100, n_repeats=1000, image_pairs=used_image_pairs)

Redundancy_Bern = 1 - (FI_Bern / FI_shuf_Bern)
Redundancy_Gaus = 1 - (FI_Gaus / FI_shuf_Gaus)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 12))

plot_scaling_metric(FI_Bern, N_Bern, aggregation='median', color = "tab:orange",  title="Fisher Info, 100 image pairs (Bernoulli)", ylim = 0, ax= axes[0,0])
#Gaus
plot_scaling_metric(FI_Gaus, N_Gaus, aggregation='median', color = "tab:blue",  title="Fisher Info, 100 image pairs (Gaussian)", ylim = 0, ax= axes[0,1])
#Redundancy Bern
plot_scaling_metric(100*Redundancy_Bern, N_Bern, aggregation='median', color = "tab:orange" , title="Information Redundancy",
                    yline = 0, ylabel = "Redundant Information,%", ax= axes[1,0])
#Redundancy Gaus
plot_scaling_metric(100*Redundancy_Gaus, N_Gaus, aggregation='median', color = "tab:blue" , title="Information Redundancy",
                    yline = 0, ylabel = "Redundant Information,%", ax= axes[1,1])
plt.tight_layout()
plt.savefig(wd +  '/plots/Anton/Fisher_Info_and_Redundancy.pdf')

# fig, ax = plt.subplots(figsize=(10, 6))
# plot_scaling_metric(FI_shuf_Bern, N_Bern, aggregation='median', title="Fisher Info suffled vs Population Size", ax=ax)
# plt.tight_layout()
