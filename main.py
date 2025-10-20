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
frames = full(shape = (5,144,256), fill_value = 128)

a, b, c, d = make_predictions('dynamic', frames, 3, [[4,7]], noise_seeds = 2, num_frames = 3)

ano, bno, cno, dno = make_predictions('no noise', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3)

print('dynamic to no noise: ', np.unique(a == ano, return_counts = True))
print(f'max of dynamic: {np.max(a):.3f}; max of no noise: {np.max(ano):.3f}\n')

a, b, c, d = make_predictions('dynamic', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3, stochastic_bin_param = True)

ano, bno, cno, dno = make_predictions('constant', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3, stochastic_bin_param = True)

print('two stoch bin: ', np.unique(a == ano, return_counts = True))
print(f'max of first: {np.max(a):.3f}; max of no second" {np.max(ano):.3f}')

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