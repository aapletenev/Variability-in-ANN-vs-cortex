import numpy as np
from numpy import full
from prediction import make_predictions
import time
import tracemalloc
from random_images import load_random_images
import os
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('MacOSX')


"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""
#the format is (num_images, height, width)
images = np.load('image_arr_anton(first100images).npy')

#plot first  image = images[0]  to verify loading worked
plt.imshow(images[3], cmap='gray')


start_time = time.perf_counter()

# Make predictions on this batch
pred_arrs_Gaus = make_predictions('dynamic', images, 10, [[4, 7]],
                                 stochastic_bin_param=False,
                                 noise_seeds=100, before_sum=True)


duration = (time.perf_counter()- start_time)/60
print(f"Code execution duration: {duration:.4f} minutes")

#save the predictions in predictions/Anton
wd = os.getcwd()
path = wd + '/predictions/Anton/'
#create 5 directories in path if it does not exist "frames", "sum", "mean", "var", "label"
os.makedirs(path, exist_ok=True)
dirs = ['frames', 'sum', 'mean', 'var', 'label']
for dir in dirs:
    os.makedirs(path + dir, exist_ok=True)
for i,dir in enumerate(dirs):
    np.save(path + dir + '/Gaus_10.npy', pred_arrs_Gaus[i])



pred_arrs_Bern = make_predictions('dynamic', images, 0, [[4, 7]],
                                 stochastic_bin_param=True,
                                 noise_seeds=100, before_sum=True)

#save the predictions in predictions/Anton
for i,dir in enumerate(dirs):
    np.save(path + dir + '/Bern.npy', pred_arrs_Bern[i])



###sanity check plots
#now plot var vs mean for all neurons for Bernulli
var = pred_arrs_Bern[3]
mean = pred_arrs_Bern[2]
labels = np.array(pred_arrs_Bern[4])
var_V1 = var[:, labels[0,:] == 1]
mean_V1 = mean[:, labels[0,:] == 1]

#substitute all var and mean values to NA if mean > 100
var_V1[mean_V1 > 100] = np.nan
mean_V1[mean_V1 > 100] = np.nan

#do a scatter plot of var vs mean for all neurons in V1
plt.scatter(mean_V1.flatten(), var_V1.flatten(), alpha=0.1)
plt.xlabel('Mean Prediction')

#in log log space
plt.xscale('log')
plt.yscale('log')



#prediction for the constant stimulus
pred_arrs_const = make_predictions('dynamic', images[3:4], 0, [[4, 7]],
                                 stochastic_bin_param=False,
                                 noise_seeds=2, before_sum=True)

plt.plot(pred_arrs_const[0][0, 0, :, 13])
plt.plot(pred_arrs_Bern[0][0, 0, :, 13])
plt.plot(pred_arrs_Bern[0][0, 1, :, 13])
plt.plot(pred_arrs_Bern[0][0, 2, :, 13])

