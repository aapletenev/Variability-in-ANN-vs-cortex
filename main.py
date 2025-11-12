import numpy as np
from numpy import full
from prediction import make_predictions
import os

"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""

# note that image input must have shape (x, 144, 256) where x is the number of frames 
#frames = full(shape = (5,144,256), fill_value = 128)

from random_images import load_random_images
images = load_random_images('imagenetmini/imagenet-mini', 10)

a, b, c, d = make_predictions('dynamic', images, 3, [[4,7]], stochastic_bin_param = False, noise_seeds = 2, num_frames = 3)
print(a.shape)

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
