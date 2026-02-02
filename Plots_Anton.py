%matplotlib qt
import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from scipy import linalg
from prediction_Anton import make_predictions
from functions_Anton import *
from matplotlib.ticker import MaxNLocator



############Start#################################
##load the data
wd = os.getcwd()
string_path = '/predictions/Anton/'




def plot_poisson_raster(n_trials, firing_rate_hz, duration_ms, font_scale=1.0):
    """
    Generates a raster plot of Poisson spikes with a side panel showing spike counts.

    Parameters:
    - n_trials (int): Number of trials (rows).
    - firing_rate_hz (float): Average firing rate in Hz.
    - duration_ms (float): Duration of the trial in milliseconds.
    - font_scale (float): Multiplier for all font sizes (default 1.0).
    """

    # --- 0. Define Font Sizes Based on Scale ---
    label_size = 12 * font_scale
    tick_size = 10 * font_scale
    text_size = 10 * font_scale

    # --- 1. Generate Data ---
    spike_times = []
    spike_counts = []
    np.random.seed(42)

    for _ in range(n_trials):
        lam = firing_rate_hz * (duration_ms / 1000.0)
        count = np.random.poisson(lam)
        spikes = np.sort(np.random.uniform(0, duration_ms, count))
        spike_times.append(spikes)
        spike_counts.append(count)

    # --- 2. Setup Plot Layout ---
    fig, (ax_raster, ax_counts) = plt.subplots(
        1, 2,
        sharey=True,
        figsize=(10, n_trials * 0.25 + 2),
        gridspec_kw={'width_ratios': [6, 1]}
    )

    # --- 3. Plot Raster (Left Panel) ---
    ax_raster.eventplot(spike_times,
                        orientation='horizontal',
                        colors='black',
                        linelengths=0.8,
                        lineoffsets=np.arange(1, n_trials + 1))

    # Apply Scaled Font Sizes
    ax_raster.set_xlabel("Time (ms)", fontsize=label_size)
    ax_raster.set_ylabel("Trial Number", fontsize=label_size)
    ax_raster.tick_params(axis='both', labelsize=tick_size)

    ax_raster.set_xlim(0, duration_ms)
    ax_raster.set_ylim(0, n_trials + 1)

    # Force Integer Ticks
    ax_raster.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)

    # --- 4. Plot Counts (Right Panel) ---
    y_pos = np.arange(1, n_trials + 1)
    bars = ax_counts.barh(y_pos, spike_counts, height=0.8, color='#e0e0e0', edgecolor='white')

    for i, bar in enumerate(bars):
        count = spike_counts[i]
        ax_counts.text(bar.get_width() + 0.5,
                       bar.get_y() + bar.get_height() / 2,
                       str(count),
                       va='center', ha='left',
                       fontsize=text_size, fontweight='bold')

    # Apply Scaled Font Sizes to Side Panel
    ax_counts.set_xlabel("Spike count", fontsize=label_size)
    ax_counts.tick_params(axis='x', labelsize=tick_size)

    ax_counts.set_xlim(0, max(spike_counts) * 1.3)
    ax_counts.tick_params(left=False, labelleft=False)

    ax_counts.spines['top'].set_visible(False)
    ax_counts.spines['right'].set_visible(False)
    ax_counts.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# Try changing font_scale to 1.5 or 0.8 to see the difference
plot_poisson_raster(n_trials=20, firing_rate_hz=25, duration_ms=1000, font_scale=1.5)
plt.savefig(wd + '/plots/Anton/poisson_raster1.pdf')

plot_poisson_raster(n_trials=20, firing_rate_hz=40, duration_ms=1000, font_scale=1.5)
plt.savefig(wd + '/plots/Anton/poisson_raster2.pdf')

plot_poisson_raster(n_trials=20, firing_rate_hz=12, duration_ms=1000, font_scale=1.5)
plt.savefig(wd + '/plots/Anton/poisson_raster4.pdf')



##################gifs#######################
#import the packacke fo giffs imageio
import imageio


images = np.load('image_arr_anton(first100images).npy')

image_Bern =  process_images_batched(
        images[19:20],
        num_trials = 10,
        num_frames = 15,
        noise_type = "dynamic",
        stochastic_bin_param = True,
        sigma= 10,
    return_sum_over_frames = False
)

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(image_Bern[0,0,:,:,:][i * 3 + j], cmap='gray')
        axs[i, j].axis('off')
plt.show()

plt.imshow(images[19], cmap='gray')
plt.axis('off')
plt.savefig(wd + '/plots/Anton/original_image19.pdf')

plt.imshow(image_Bern[0,0,0,:,:], cmap='gray')
plt.axis('off')
plt.savefig(wd + '/plots/Anton/image_Bern.pdf')

import imageio
imageio.mimsave(wd + '/plots/Anton/Bern.gif', image_Bern[0,0,:,:,:], duration=0.033, loop=0)


#now Gaussian
image_Gaus =  process_images_batched(
        images[19:20],
        num_trials = 10,
        num_frames = 15,
        noise_type = "dynamic",
        stochastic_bin_param = False,
        sigma= 10,
    return_sum_over_frames = False
)
imageio.mimsave(wd + '/plots/Anton/Gaus.gif', image_Gaus[0,0,:,:,:], duration=0.033, loop=0)

##now with sigma = 300
image_Gaus_high_sigma =  process_images_batched(
        images[19:20],
        num_trials = 10,
        num_frames = 15,
        noise_type = "dynamic",
        stochastic_bin_param = False,
        sigma= 300,
    return_sum_over_frames = False
)

imageio.mimsave(wd + '/plots/Anton/Gaus_high_sigma.gif', image_Gaus_high_sigma[0,0,:,:,:], duration=0.033, loop=0)


#now create an artificial image with 144,256 pixels where the pixel values increase linearly from 0 to 255 and then again 0 ... 255 and so on
artificial_image = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (144, 1))
#plot
plt.imshow(artificial_image, cmap='gray')

#add new axis to make it (1,144,256)
artificial_image = artificial_image[np.newaxis, :, :]

#now loop over sigma values 0,5,10,15,20,25,30, 40,  250 and compute and store the median of the variance over trials for each sigma
sigma_values = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500]
median_vars = []
for sigma in sigma_values:
    Gaus =  process_images_batched(
            artificial_image,
            num_trials = 100,
            num_frames = 1,
            noise_type = "dynamic",
            stochastic_bin_param = False,
            sigma= sigma,
        return_sum_over_frames = True
    )
    var_Gaus = np.var(Gaus[0, :, :, :], axis=0)
    median_var_Gaus = np.median(var_Gaus)
    median_vars.append(median_var_Gaus)


#calculate for Bernoulli noise. Only for sigma = 0
image_Bern =  process_images_batched(
        artificial_image,
        num_trials = 100,
        num_frames = 1,
        noise_type = "dynamic",
        stochastic_bin_param = True,
        sigma= 0,
    return_sum_over_frames = True
)
var_Bern = np.var(image_Bern[0, :, :, :], axis=0)
median_var_Bern = np.median(var_Bern)

#plot sigma values vs median variance
plt.figure(figsize=(8, 6))
plt.plot(sigma_values, median_vars,
         #no markers at all, just line
            linestyle='-',
            color='tab:blue',
         #increase line width
            linewidth=2
            )
#now plot Bernoulli point as orange
plt.scatter(600, median_var_Bern, color='tab:orange', s=200, label='Bernoulli Noise')
plt.scatter(10, median_vars[2], color='tab:blue', s=200, label='Bernoulli Noise')
plt.xlabel('Sigma (Gaussian noise)', fontsize=1.5*12)
plt.ylabel('Pixel Variance', fontsize=1.5*12)
plt.title('', fontsize=16)
plt.tick_params(axis='both', labelsize=1.5*10)
plt.grid(False)
plt.tight_layout()
plt.savefig(wd + '/plots/Anton/sigma_vs_median_variance2.pdf')







#####mean noise vs mean no noise plots####
wd = os.getcwd()
string_path = '/predictions/Anton/'
labels = np.load(wd + string_path + 'label/Bern.npy')
#data is now of shape (num_images, num_noise, num_frames, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
Spike_frames_Bern = get_neurons_of_area(np.load(wd + string_path + 'frames/Bern.npy'), labels)
Spike_frames_noise_free = get_neurons_of_area(np.load(wd + string_path + 'frames/no_noise.npy'), labels)
Spike_frames_Gaus = get_neurons_of_area(np.load(wd + string_path + 'frames/Gaus_10.npy'), labels)


mean_Bern =  get_neurons_of_area(np.load(wd + string_path + 'mean/Bern.npy'), labels)
mean_Gaus =  get_neurons_of_area(np.load(wd + string_path + 'mean/Gaus_10.npy'), labels)
mean_no_noise =  get_neurons_of_area(np.load(wd + string_path + 'mean/no_noise.npy'), labels)

#all > 100 spikes - nan mask
mean_Bern[mean_Bern > 100] = np.nan
mean_Gaus[mean_Gaus > 100] = np.nan
mean_no_noise[mean_no_noise > 100] = np.nan


#find neurons indexes which mean no noise is not different than mean with Bernoulli noise (difference < 1 ) and mean with no noise > 10
mask = (np.abs(mean_no_noise - mean_Bern) < 1) & (mean_no_noise > 10)
#return the indexes mask is 2D so the indexes is also
indexes = np.where(mask)
i = 500
index1 = indexes[0][i]  #take the first neuron that satisfies the condition
index2 = indexes[1][i]




fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

# --- Row 1:
plot_spike_comparison(Spike_frames_noise_free[index1, 0, :, index2], 'Bernoulli Noise',
                      Spike_frames_Bern[index1,:,:, index2], 'tab:orange', font_scale=1.5, ax=axes[0,0])
plot_spike_comparison(Spike_frames_noise_free[index1, 0, :, index2], 'Gaussian Noise (σ=10)',
                      Spike_frames_Gaus[index1,:,:, index2], 'tab:blue', font_scale=1.5, ylim = np.max(Spike_frames_Bern[index1,:,:, index2]), ax=axes[0,1])

# --- Row 2:
plot_mean_comparison(axes[1,0], mean_no_noise, mean_Bern, '', 'tab:orange')
plot_mean_comparison(axes[1,1], mean_no_noise, mean_Gaus, "", 'tab:blue')

# --- Add Row Subtitles ---
fig.text(0.5, 0.96, 'Example neuron', ha='center', va='center', fontsize=20, fontweight='bold')
fig.text(0.5, 0.48, 'All neurons', ha='center', va='center', fontsize=20, fontweight='bold')

# Adjust layout to make room for the text labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
plt.savefig(wd + '/plots/Anton/mean_no_noise_vs_mean_with_noise.png', dpi=300)








####a vs b plots###

##load the data
wd = os.getcwd()
string_path = '/predictions/8-27-2025/'
#show all dirictories inside it
dirs = os.listdir(wd + '/' + string_path)

#Now load in the data
#create empty npy arrays to  append data
data = []
data_mean = []
var = []
labels = [] #this is the area labels
for dir in dirs:
    data.append(np.load(wd + '/' + string_path + dir + '/sum_arr_'+ dir + '.npy'))
    data_mean.append(np.load(wd + '/' + string_path + dir + '/mean_arr_'+ dir + '.npy'))
    labels.append(np.load(wd + '/' + string_path + dir + '/label_arr_'+ dir + '.npy'))
    #variance across noise levels
    var.append(np.var(np.load(wd + '/' + string_path + dir + '/sum_arr_'+ dir + '.npy'), axis = 1))

data = np.concatenate(data, axis = 0)
data_mean = np.concatenate(data_mean, axis = 0)
labels = np.concatenate(labels, axis = 0)
var = np.concatenate(var, axis = 0)
#data is now of shape (num_images, num_noise, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
data_V1 = data[:,:,labels[0,:] == 1]
data_mean_V1 = data_mean[:,labels[0,:] == 1]
var_V1 = var[:,labels[0,:] == 1]

#now I need the regression var vs mean, the function is a*mean^b, so i need to cllect a anad and R^2 for regression log(var) = log(a) + b*log(mean)
from scipy.stats import linregress
a_list = []
b_list = []
r2_list = []
for neuron in range(data_V1.shape[2]):
    mean_vals = data_mean_V1[:,neuron]
    var_vals = var_V1[:,neuron]
    #remove zeros
    mask = (mean_vals > 0) & (var_vals > 0)
    mean_vals = mean_vals[mask]
    var_vals = var_vals[mask]
    log_mean = np.log(mean_vals)
    log_var = np.log(var_vals)
    slope, intercept, r_value, p_value, std_err = linregress(log_mean, log_var)
    a = np.exp(intercept)
    b = slope
    r2 = r_value**2
    a_list.append(a)
    b_list.append(b)
    r2_list.append(r2)

#convert to numpy arrays
a_array = np.array(a_list)
b_array = np.array(b_list)
r2_array = np.array(r2_list)

#plot a vs b scatter plot, be aware that there are 5000 points so make them small and transparent
plt.figure(figsize=(10, 6))
plt.scatter(b_array, a_array, s=1, alpha=0.5)
plt.xlabel('b parameter', fontsize=16)
plt.ylabel('a parameter', fontsize=16)
plt.title('Bernoulli noise', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#add the regression to the plot as a black line with confidence interval
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(b_array, a_array)
x_vals = np.array([np.min(b_array), np.max(b_array)])
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='black', label='Regression line')


#save the figure in pdf
plt.savefig(wd + '/plots/Anton/a_vs_b_scatter_plot_bernulli.pdf')
#png
plt.savefig(wd + '/plots/Anton/a_vs_b_scatter_plot_bernulli.png', dpi=300)



