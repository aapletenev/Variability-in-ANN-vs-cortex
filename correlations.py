import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from functions_Anton import *
import matplotlib.ticker as ticker

matplotlib.use('Agg')


############Start#################################

##load the data
wd = os.getcwd()
string_path = '/predictions/Anton/'
path_out = wd + '/intermediate_results/correlations/'
os.makedirs(path_out, exist_ok=True)
CALCULATE_COR = False
CALCULATE_GEOM_MEAN = False

#data is now of shape (num_images, num_noise, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
labels = np.load(wd + string_path + 'label/Bern.npy')
Spike_Bern = get_neurons_of_area(np.load(wd + string_path + 'sum/Bern.npy'), labels)
Mean_Bern = get_neurons_of_area(np.load(wd + string_path + 'mean/Bern.npy'), labels)
Spike_Gaus = get_neurons_of_area(np.load(wd + string_path + 'sum/Gaus_10.npy'), labels)
Mean_Gaus = get_neurons_of_area(np.load(wd + string_path + 'mean/Gaus_10.npy'), labels)

#Now substitute all values > 100 to NaN
Spike_Bern[Spike_Bern > 100] = np.nan
Mean_Bern[Mean_Bern > 100] = np.nan
Spike_Gaus[Spike_Gaus > 100] = np.nan
Mean_Gaus[Mean_Gaus > 100] = np.nan


##get all correlations
cor_Bern, cor_z_Bern, signal_cor_Bern, cor_z_top_Bern, signal_cor_top_Bern, top_neurons_Bern = get_correlations_all(Spike_Bern, Mean_Bern)
cor_Gaus, cor_z_Gaus, signal_cor_Gaus, cor_z_top_Gaus, signal_cor_top_Gaus, top_neurons_Gaus = get_correlations_all(Spike_Gaus, Mean_Gaus)

    #save all correlations in wd + '/intermediate results/correlations', make a for loop for both types

    # for suffix in ['Bern', 'Gaus']:
    #     for p in ['cor', 'top_neurons']:
    #         # locals() gets the variable safely by string name (e.g., "cor_Bern")
    #         np.save(f"{path_out}{p}_{suffix}.npy", locals()[f"{p}_{suffix}"])


####compute geometric mean of pairs of means
# if CALCULATE_GEOM_MEAN:
#     Geom_mean_Bern = geometric_mean_pairwise(Mean_Bern)
#     Geom_mean_Gaus = geometric_mean_pairwise(Mean_Gaus)
#
#     ###save geometric means
#     np.save(path_out + 'Geom_mean_Bern.npy', Geom_mean_Bern)
#     np.save(path_out + 'Geom_mean_Gaus.npy', Geom_mean_Gaus)

# #load top_neurons
# top_neurons_Bern = np.load(path_out + 'top_neurons_Bern.npy')
# top_neurons_Gaus = np.load(path_out + 'top_neurons_Gaus.npy')

#load geometric means
# Geom_mean_Bern = np.load(path_out + 'Geom_mean_Bern.npy')
# Geom_mean_Gaus = np.load(path_out + 'Geom_mean_Gaus.npy')

###get geometric mean of top neurons only
Geom_mean_Bern_top = get_lower_triangle(geometric_mean_pairwise(Mean_Bern[:, top_neurons_Bern]))
Geom_mean_Gaus_top = get_lower_triangle(geometric_mean_pairwise(Mean_Gaus[:, top_neurons_Gaus]))

#load cor_Bern and cor_Gaus
# cor_Bern = np.load(path_out + 'cor_Bern.npy')
# cor_Gaus = np.load(path_out + 'cor_Gaus.npy')

#compute cor_Bern_top and cor_Gaus_top
cor_Bern_top = get_lower_triangle(compute_corr(Spike_Bern[:, :, top_neurons_Bern]))
cor_Gaus_top = get_lower_triangle(compute_corr(Spike_Gaus[:, :, top_neurons_Gaus]))




# Create the 3x3 plot
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

plot_hist_comparison(axes[0, 0], cor_z_Bern, cor_z_top_Bern, color='tab:orange', title="Bernoulli Noise")
plot_hist_comparison(axes[0, 1], cor_z_Gaus, cor_z_top_Gaus, color='tab:blue', title="Gaussian Noise (σ=10)")


plot_scatter_noise_vs_signal(axes[1, 0], cor_z_Bern, signal_cor_Bern, cor_z_top_Bern, signal_cor_top_Bern,
    color='navajowhite', color_top='tab:orange', point_size=0.3, point_alpha=0.1, title="",  ylabel='Noise Correlation (z-scored responses)')

plot_scatter_noise_vs_signal(axes[1, 1], cor_z_Gaus, signal_cor_Gaus, cor_z_top_Gaus, signal_cor_top_Gaus,
                             color='powderblue', color_top='tab:blue', point_size=0.3, point_alpha=0.1,  title="", ylabel='Noise Correlation (z-scored responses)', yaxis_step=0.2)

plot_scatter_noise_vs_signal(axes[2,0], None, None, np.abs(cor_Bern_top.flatten()), Geom_mean_Bern_top.flatten(),
                                color='navajowhite', color_top='tab:orange', xlabel='Geometric Mean Response of a Pair', ylabel='Noise Correlation (absolute value)',
                             title = "", xline = None, xlim = (0, 80), ylim = 0, trendline= 'moving_average', window_size=5000, point_alpha=0.001, point_size=0.03)
plot_scatter_noise_vs_signal(axes[2,1], None, None, np.abs(cor_Gaus_top.flatten()), Geom_mean_Gaus_top.flatten(),
                                color='powderblue', color_top='tab:blue', xlabel='Geometric Mean Response of a Pair', ylabel='Noise Correlation (absolute value)',
                             title = "", xline = None, xlim = (0, 80), ylim = (0, None), trendline= 'moving_average', window_size=5000 , point_alpha=0.001, point_size=0.03)

# Global Title
fig.suptitle("Noise Correlations", fontsize=16)
plt.savefig(wd + '/plots/Anton/noise_correlation.png', dpi=300)













###check - finding pairs with high NC and low geometric mean

def find_specific_pairs(nc_matrix, gm_matrix, nc_thresh=0.15, gm_thresh=3.0):
    """
    Finds indices [img_id, n1, n2] where GM < 3 and NC > 0.15.

    Parameters:
    -----------
    nc_matrix : np.ndarray (N_img, N_neu, N_neu)
    gm_matrix : np.ndarray (N_img, N_neu, N_neu)

    Returns:
    --------
    results : np.ndarray
        List of [img_id, neuron1_id, neuron2_id]
    """

    # 1. Create a mask for your value conditions
    # "Geom mean < 3" AND "NC > 0.15"
    value_mask = (gm_matrix < gm_thresh) & (nc_matrix > nc_thresh)

    # 2. Create a mask for the Upper Triangle (to exclude n1==n2 and duplicates)
    # We create a 2D mask for one matrix slice and broadcast it to the 3D stack
    n_neurons = nc_matrix.shape[1]

    # k=1 means "start from the first diagonal ABOVE the main diagonal"
    # This explicitly excludes n1 == n2
    upper_tri_mask = np.triu(np.ones((n_neurons, n_neurons), dtype=bool), k=1)

    # 3. Combine masks
    # NumPy automatically broadcasts the 2D upper_tri_mask to the 3D value_mask
    final_mask = value_mask & upper_tri_mask

    # 4. Extract indices
    # np.argwhere returns an array of shape (N_found, 3) -> [dim0, dim1, dim2]
    # which corresponds to [img_id, n1, n2]
    result_indices = np.argwhere(final_mask)

    return result_indices

cor_top_Bern = compute_corr(Spike_Bern[:, :, top_neurons_Bern])
geom_mean_top_Bern = geometric_mean_pairwise(Mean_Bern[:, top_neurons_Bern])

specific_pairs = find_specific_pairs(compute_corr(Spike_Bern[:, :, top_neurons_Bern]),
                                     geometric_mean_pairwise(Mean_Bern[:, top_neurons_Bern]), nc_thresh=0.15, gm_thresh=3.0)

pair = specific_pairs[0]
#plot the scatter plot of spikes
n1 = Spike_Bern[:, :, top_neurons_Bern][pair[0], :, pair[1]]
n2 = Spike_Bern[:, :, top_neurons_Bern][pair[0], :, pair[2]]
NC = cor_top_Bern[pair[0], pair[1], pair[2]]
G_mean = geom_mean_top_Bern[pair[0], pair[1], pair[2]]
mean1 = Mean_Bern[:, top_neurons_Bern][pair[0], pair[1]]
mean2 = Mean_Bern[:, top_neurons_Bern][pair[0], pair[2]]

#plot n1 vs n2 and print the NC and G_mean
plt.figure(figsize=(8, 6))
plt.scatter(n1, n2, alpha=0.5, color = 'tab:orange')
plt.title(f'Scatter Plot of Neuron Pair\nNC: {NC:.2f}, Geometric Mean: {G_mean:.2f}')
plt.xlabel('Neuron 1 Spike Counts')
plt.ylabel('Neuron 2 Spike Counts')
plt.savefig(wd + '/plots/Anton/specific_pair_scatter.png', dpi=300)






#Now take 10 random pairs and plot 10 subplots with histogram of the correlations across images
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(10):
    ax = axs[i//5, i%5]
    neuron1 = np.random.randint(0, cor_Bern.shape[1])
    neuron2 = np.random.randint(0, cor_Bern.shape[1])
    corr_values = cor_Bern[:, neuron1, neuron2]
    ax.hist(corr_values, bins=20, color='blue', alpha=0.7)
    ax.set_title(f'Neuron {neuron1} vs Neuron {neuron2}')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    #add a vertical line at mean
    mean_corr = np.nanmean(corr_values)
    ax.axvline(mean_corr, color='red', linestyle='dashed', linewidth=1)

plt.tight_layout()
plt.show()

