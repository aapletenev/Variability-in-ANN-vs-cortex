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

#plot histogram with number of counts on y axis and white fill and black edges
plt.figure(figsize=(8, 6))
plt.hist(lower_triangle_values, bins=30, color='white', edgecolor='black', alpha=0.7)
plt.title('Noise correlation (average across images)')
plt.xlabel('Noise Correlation')
plt.ylabel('Frequency')
plt.axvline(np.mean(lower_triangle_values), color='black', linestyle='dashed', linewidth=1)
#add text box with mean and interquartile range as 1st and 3rd quartiles
mean_corr = np.mean(lower_triangle_values)
q1 = np.percentile(lower_triangle_values, 25)
q3 = np.percentile(lower_triangle_values, 75)
textstr = f'Mean: {mean_corr:.3f}\nQ1: {q1:.3f}\nQ3: {q3:.3f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

#save plot in pdf format in directory "plots/Anton/"
if not os.path.exists(wd + '/plots/Anton/'):
    os.makedirs(wd + '/plots/Anton/')
plt.savefig(wd + '/plots/Anton/noise_correlation_histogram_avg.pdf')

#####now compute noise correlations as correlations of z-scoring responses per each image
data_V1_zscored = (data_V1 - np.mean(data_V1, axis=1, keepdims=True)) / np.std(data_V1, axis=1, keepdims=True)
#now flatten across images and noise levels to shape (num_images * num_noise, num_neurons)
data_V1_zscored = data_V1_zscored.reshape(-1, data_V1_zscored.shape[2])
#compute correlation matrix
corr_matrix_zscored = np.corrcoef(data_V1_zscored.T)
#take lower triangle without diagonal and plot histogram
lower_triangle_values_zscored = corr_matrix_zscored[lower_triangle_indices]
#plot histogram with number of counts on y axis and white fill and black edges
plt.figure(figsize=(8, 6))
plt.hist(lower_triangle_values_zscored, bins=30, color='white', edgecolor='black', alpha=0.7)
plt.title('Noise correlation (z-scored responses)')
plt.xlabel('Noise Correlation')
plt.ylabel('Frequency')
plt.axvline(np.mean(lower_triangle_values_zscored), color='black', linestyle='dashed', linewidth=1)
#add text box with mean and interquartile range as 1st and 3rd quartiles
mean_corr_zscored = np.mean(lower_triangle_values_zscored)
q1_zscored = np.percentile(lower_triangle_values_zscored, 25)
q3_zscored = np.percentile(lower_triangle_values_zscored, 75)
textstr_zscored = f'Mean: {mean_corr_zscored:.3f}\nQ1: {q1_zscored:.3f}\nQ3: {q3_zscored:.3f}'
props_zscored = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.95, 0.95, textstr_zscored, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props_zscored)
#save plot in pdf format in directory "plots/Anton/"
plt.savefig(wd + '/plots/Anton/noise_correlation_histogram_zscored.pdf')

