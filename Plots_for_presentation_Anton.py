%matplotlib qt
import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from scipy import linalg
from prediction import make_predictions
from functions_Anton import *
from matplotlib.ticker import MaxNLocator



############Start#################################
##load the data
wd = os.getcwd()
string_path = '/predictions/Anton/'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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







########within trial dynamic############
#prediction for the constant stimulus
pred_arrs_const = make_predictions('dynamic', images[0:4], 0, [[4, 7]],
                                 stochastic_bin_param=False,
                                 noise_seeds=1, before_sum=True)


def plot_spike_comparison(ref_array, title_text, stack_array=None, line_color='blue', font_scale=1.0):
    """
    Plots a single reference line (black). Optionally plots a stack of background lines if provided.
    Forces integer ticks on the X-axis.

    Parameters:
    - ref_array (1D numpy array): The main data trace (plotted in black).
    - title_text (str): Title of the plot.
    - stack_array (2D numpy array, optional): Multiple lines (N_lines x N_frames). Defaults to None.
    - line_color (str, optional): Color for the stack_array lines. Defaults to 'blue'.
    - font_scale (float, optional): Multiplier for all font sizes. Defaults to 1.0.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    # Calculate scaled font sizes
    title_size = 14 * font_scale
    label_size = 12 * font_scale
    tick_size = 10 * font_scale

    # Define x-axis based on the reference array length
    x_axis = np.arange(ref_array.shape[0]) + 1

    # 1. Plot the 2D stack (ONLY if it is not None)
    if stack_array is not None:
        # Loop through rows and plot each
        for i in range(stack_array.shape[0]):
            ax.plot(x_axis, stack_array[i, :], color=line_color, alpha=0.4, linewidth=1.5)

    # 2. Plot the 1D reference array (Black line)
    # This always runs
    ax.plot(x_axis, ref_array, color='black', linewidth=2.5, label='Reference')

    # 3. Formatting
    ax.set_title(title_text, fontsize=title_size)
    ax.set_xlabel("Frames", fontsize=label_size)
    ax.set_ylabel("Spike Count", fontsize=label_size)

    # --- FIX: Force Integer Ticks on X-Axis ---
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Update tick label sizes
    ax.tick_params(axis='both', labelsize=tick_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


string_path = '/predictions/Anton/'
#data is now of shape (num_images, num_noise, num_frames, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
Spike_frames_Bern = np.load(wd + string_path + 'frames/Bern.npy')

plot_spike_comparison(pred_arrs_const[0][3, 0, :, 13], "",
                      Spike_frames_Bern[3,0:4,:, 13], 'white', font_scale=1.5)

plt.savefig(wd + '/plots/Anton/spike_comparison_neuron0.pdf')


plot_spike_comparison(pred_arrs_const[0][3, 0, :, 13], "Bernoulli noise",
                      Spike_frames_Bern[3,0:1,:, 13], 'tab:orange', font_scale=1.5)

plt.savefig(wd + '/plots/Anton/spike_comparison_Bern_neuron1.pdf')

plot_spike_comparison(pred_arrs_const[0][3, 0, :, 13], "Bernoulli noise",
                      Spike_frames_Bern[3,0:2,:, 13], 'tab:orange', font_scale=1.5)

plt.savefig(wd + '/plots/Anton/spike_comparison_Bern_neuron2.pdf')

plot_spike_comparison(pred_arrs_const[0][3, 0, :, 13], "Bernoulli noise",
                      Spike_frames_Bern[3,0:3,:, 13], 'tab:orange', font_scale=1.5)

plt.savefig(wd + '/plots/Anton/spike_comparison_Bern_neuron3.pdf')


Spike_frames_Gaus = np.load(wd + string_path + 'frames/Gaus_10.npy')
plot_spike_comparison(pred_arrs_const[0][3, 0, :, 13], "Gaussian noise",
                      Spike_frames_Gaus[3,0:3,:, 13], 'tab:blue', font_scale=1.5)
plt.savefig(wd + '/plots/Anton/spike_comparison_Gaus_neuron3.pdf')




plt.plot(pred_arrs_const[0][3, 0, :, 13]
         #color black
         , color='black',
         #y axis labels
            label='Neuron 0'
            )



plt.plot(pred_arrs_Bern[0][0, 0, :, 13])
plt.plot(pred_arrs_Bern[0][0, 1, :, 13])
plt.plot(pred_arrs_Bern[0][0, 2, :, 13])