import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('MacOSX')

##load the data
wd = os.getcwd()
string_path = '/predictions/8-27-2025/'
#show all dirictories inside it
dirs = os.listdir(wd + '/' + string_path)

#Now load in the data
#create empty npy arrays to  append data
data = []
data_mean = []
labels = [] #this is the area labels
for dir in dirs:
    data.append(np.load(wd + '/' + string_path + dir + '/sum_arr_'+ dir + '.npy'))
    data_mean.append(np.load(wd + '/' + string_path + dir + '/mean_arr_'+ dir + '.npy'))
    labels.append(np.load(wd + '/' + string_path + dir + '/label_arr_'+ dir + '.npy'))

data = np.concatenate(data, axis = 0)
data_mean = np.concatenate(data_mean, axis = 0)
labels = np.concatenate(labels, axis = 0)
#data is now of shape (num_images, num_noise, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
data_V1 = data[:,:,labels[0,:] == 1]
data_mean_V1 = data_mean[:,labels[0,:] == 1]

#now calculate the correlation for data_V1 across noise levels, so output shape (num_images, num_neurons, num_neurons)
correlations = []
for i in range(data_V1.shape[0]):
    corr_matrix = np.corrcoef(data_V1[i,:,:].T)
    correlations.append(corr_matrix)

correlations = np.array(correlations)

#Now take 10 randon pairs and plot 10 subplots with histogram of the correlations across images
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(10):
    ax = axs[i//5, i%5]
    neuron1 = np.random.randint(0, correlations.shape[1])
    neuron2 = np.random.randint(0, correlations.shape[1])
    corr_values = correlations[:, neuron1, neuron2]
    ax.hist(corr_values, bins=20, color='blue', alpha=0.7)
    ax.set_title(f'Neuron {neuron1} vs Neuron {neuron2}')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    #add a vertical line at mean
    mean_corr = np.mean(corr_values)
    ax.axvline(mean_corr, color='red', linestyle='dashed', linewidth=1)

plt.tight_layout()
plt.show()

#computer average correlation matrix across images
avg_correlation_matrix = np.mean(correlations, axis=0)
#now take lower triangle without diagonal and plot histogram
lower_triangle_indices = np.tril_indices(avg_correlation_matrix.shape[0], k=-1)
lower_triangle_values = avg_correlation_matrix[lower_triangle_indices]


#####now compute noise correlations as correlations of z-scoring responses per each image
data_V1_zscored = (data_V1 - np.mean(data_V1, axis=1, keepdims=True)) / np.std(data_V1, axis=1, keepdims=True)
#now flatten across images and noise levels to shape (num_images * num_noise, num_neurons)
data_V1_zscored = data_V1_zscored.reshape(-1, data_V1_zscored.shape[2])
#compute correlation matrix
corr_matrix_zscored = np.corrcoef(data_V1_zscored.T)
#take lower triangle without diagonal and plot histogram
lower_triangle_values_zscored = corr_matrix_zscored[lower_triangle_indices]


###now compute signal correlation matrix as correlation of mean responses across images
data_mean_V1_reshaped = data_mean_V1.reshape(-1, data_mean_V1.shape[1])
signal_corr_matrix = np.corrcoef(data_mean_V1_reshaped.T)

#take lower triangle without diagonal
lower_triangle_values_signal = signal_corr_matrix[lower_triangle_indices]


#now calculate noise correlation(z-scored) and signal correlation for only 10% of the neurons with highest median mean response
num_neurons_to_select = int(0.1 * data_mean_V1.shape[1])
median_mean_responses = np.median(data_mean_V1_reshaped, axis=0)
top_neuron_indices = np.argsort(median_mean_responses)[-num_neurons_to_select:]
#compute noise correlation matrix for these neurons
data_V1_zscored_top = data_V1_zscored[:, top_neuron_indices]
corr_matrix_zscored_top = np.corrcoef(data_V1_zscored_top.T)
#need to get lower triangle indices for smaller matrix
lower_triangle_indices_top = np.tril_indices(corr_matrix_zscored_top.shape[0], k=-1)
lower_triangle_values_zscored_top = corr_matrix_zscored_top[lower_triangle_indices_top]
#compute signal correlation matrix for these neurons
data_mean_V1_top = data_mean_V1_reshaped[:, top_neuron_indices]
signal_corr_matrix_top = np.corrcoef(data_mean_V1_top.T)
lower_triangle_values_signal_top = signal_corr_matrix_top[lower_triangle_indices_top]

#now plot scatter plot noise correlation vs signal correlation for all neurons (blue points) and for top 10% neurons (red points)
plt.figure(figsize=(8, 6))
#make points small
plt.scatter(lower_triangle_values_signal, lower_triangle_values_zscored, s=0.3, alpha=0.1, label='All Neurons', color='blue')
plt.scatter(lower_triangle_values_signal_top, lower_triangle_values_zscored_top, s=1, alpha=0.5, label='Top 10% Neurons', color='red')
plt.title('Noise Correlation vs Signal Correlation')
plt.xlabel('Signal Correlation')
plt.ylabel('Noise Correlation')
#add line of best fit for all neurons
m, b = np.polyfit(lower_triangle_values_signal, lower_triangle_values, 1)
#not dashed line but smaller dashed line
plt.plot(lower_triangle_values_signal, m*lower_triangle_values_signal + b, color='black', linestyle='dotted', label='Fit All Neurons')
#add line of best fit for top neurons
m_top, b_top = np.polyfit(lower_triangle_values_signal_top, lower_triangle_values_zscored_top, 1)
plt.plot(lower_triangle_values_signal_top, m_top*lower_triangle_values_signal_top + b_top, color='black', label='Fit Top 10% Neurons')
#add legend
plt.legend()
#compute pearson correlation coefficient between noise and signal correlations
pearson_corr = np.corrcoef(lower_triangle_values_signal, lower_triangle_values_zscored)[0, 1]
pearson_corr_top = np.corrcoef(lower_triangle_values_signal_top, lower_triangle_values_zscored_top)[0, 1]

#add text box with pearson correlation coefficient and slope value
textstr_scatter = f'Pearson r (All): {pearson_corr:.3f}\nPearson r (Top 10%): {pearson_corr_top:.3f}\nSlope (All): {m:.3f}\nSlope (Top 10%): {m_top:.3f}'
props_scatter = dict(boxstyle='round', facecolor='white', alpha=0.5)
#make text box at bottom right
plt.text(0.95, 0.05, textstr_scatter, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props_scatter)
#add 0 x and y lines with light gray color
plt.axhline(0, color='lightgray', linestyle='dashed', linewidth=1)
plt.axvline(0, color='lightgray', linestyle='dashed', linewidth=1)


#save plot in pdf format in directory "plots/Anton/"
plt.savefig(wd + '/plots/Anton/noise_vs_signal_correlation_scatter_top10.pdf')
#save it also as png as very heavy file
plt.savefig(wd + '/plots/Anton/noise_vs_signal_correlation_scatter_top10.png', dpi=300)

#compute the slope


#now histogram of noise correlations with all neurons - white with black edges and top 10% - red(transparent) on secondary y axis as the number of pairs is too small
bins = np.histogram_bin_edges(lower_triangle_values, bins=30)
counts_all, _ = np.histogram(lower_triangle_values, bins=bins)
counts_top, _ = np.histogram(lower_triangle_values_zscored_top, bins=bins)
scaling_factor = 0.5
counts_top_scaled = counts_top * scaling_factor

fig, ax1 = plt.subplots(figsize=(8, 6))
bars_all = ax1.bar(bins[:-1], counts_all, width=np.diff(bins), align='edge',
                   color='white', edgecolor='black', alpha=0.7)

ax2 = ax1.twinx()
bars_top = ax2.bar(bins[:-1], counts_top_scaled, width=np.diff(bins), align='edge',
                   color='red', edgecolor='black', alpha=0.5)

mean_corr = np.mean(lower_triangle_values)
mean_corr_top = np.mean(lower_triangle_values_zscored_top)
ax1.axvline(mean_corr, color='black', linestyle='dashed', linewidth=1)
ax2.axvline(mean_corr_top, color='red', linestyle='dashed', linewidth=1)

ax1.set_xlabel('Noise Correlation')
ax1.set_ylabel('Frequency (All Neurons)')
ax2.set_ylabel('Frequency (Top 10% Neurons)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Correct legend using proxy artists with correct facecolor
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor='white', edgecolor='black', label='All Neurons'),
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Top 10% Neurons')
]
ax1.legend(handles=legend_patches, loc='upper left')

textstr_hist = f'Mean (All): {mean_corr:.3f}\nMean (Top 10%): {mean_corr_top:.3f}'
props_hist = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax1.text(0.95, 0.95, textstr_hist, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props_hist)

plt.title('Noise correlation (z-scored responses)')
plt.tight_layout()


#save plot in pdf format in directory "plots/Anton/"
plt.savefig(wd + '/plots/Anton/noise_correlation_histogram_comparison_top10.pdf')
#save it also as png as very heavy file
plt.savefig(wd + '/plots/Anton/noise_correlation_histogram_comparison_top10.png', dpi=300)

#compute the slope