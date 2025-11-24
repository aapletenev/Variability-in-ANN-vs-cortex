import numpy as np
from numpy import full
from prediction import make_predictions
import time
import tracemalloc
from random_images import load_random_images
import os
"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""

# note that image input must have shape (x, 144, 256) where x is the number of frames 
#frames = full(shape = (5,144,256), fill_value = 128)
  # Save array 'a' for this batch    
n_images = 10000
batch_size = 1000
output_dir = 'predictions/incremental_saves'
image_stack = np.zeros((100, 128, 256)) # in reality this will be saved array
os.makedirs(output_dir, exist_ok=True)

# Process in batches
for batch_idx in range(0, n_images, batch_size):
    end_idx = min(batch_idx + batch_size, n_images)
    
    # Slice the pre-allocated image_stack for this batch
    images_batch = image_stack[batch_idx:end_idx]
    
    # Make predictions on this batch
    a, b, c, d = make_predictions('dynamic', images_batch, 3, [[4,7]], 
                                   stochastic_bin_param=False, 
                                   noise_seeds=2, 
                                   num_frames=3)
    
    # Save array 'a' for this batch
    np.save(f'{output_dir}/a_batch_{batch_idx:05d}.npy', a)
    print(f"saved batch {batch_idx}")
    

"""
Anton test
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
"""