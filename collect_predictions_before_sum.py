%matplotlib qt
import numpy as np
from numpy import full
from prediction import make_predictions
import time
import tracemalloc
from random_images import load_random_images
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time




"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""



#the format is (num_images, height, width)
images = np.load('image_arr_anton(first100images).npy')

#plot first  image = images[0]  to verify loading worked
plt.imshow(images[0], cmap='gray')

wd = os.getcwd()
dirs = ['frames', 'sum', 'mean', 'var', 'label']

#################100 images predictions########################
path = wd + '/predictions/Anton/'
#create 5 directories in path if it does not exist "frames", "sum", "mean", "var", "label"
os.makedirs(path, exist_ok=True)

# Make predictions on this batch
pred_arrs_Gaus = make_predictions('dynamic', images, 10, [[4, 7]],
                                 stochastic_bin_param=False,
                                 noise_seeds=100, before_sum=True)


duration = (time.perf_counter()- start_time)/60
print(f"Code execution duration: {duration:.4f} minutes")

#save the predictions in predictions/Anton
save_predictions(pred_arrs_Gaus, path, 'Gaus_10.npy', dirs)



pred_arrs_Bern = make_predictions('dynamic', images, 0, [[4, 7]],
                                 stochastic_bin_param=True,
                                 noise_seeds=100, before_sum=True)

#save the predictions in predictions/Anton
save_predictions(pred_arrs_Bern, path, 'Bern.npy', dirs)



#####add predictions for 100 images but 100 trials but with 10 blank images at the start######
path = wd + '/predictions/Anton/Blank_image_in_front/'
os.makedirs(path, exist_ok=True)

pred_arrs_Bern_blank = make_predictions('dynamic', images, 0, [[4, 7]],
                                    stochastic_bin_param=True,
                                    noise_seeds=100, num_frames = 20, num_frames_blank=10, before_sum=True)
#save the predictions in predictions/Anton
save_predictions(pred_arrs_Bern_blank, path, 'Bern_blank_10.npy', dirs)
pred_arrs_Gaus_blank = make_predictions('dynamic', images, 10, [[4, 7]],
                                    stochastic_bin_param=False,
                                    noise_seeds=20, num_frames = 20, num_frames_blank=10, before_sum=True)
#save the predictions in predictions/Anton
save_predictions(pred_arrs_Gaus_blank, path, 'Gaus_10_blank_10.npy', dirs)



##############10 images but 1000 trials##################
path = wd + '/predictions/Anton/1000_trials/'
os.makedirs(path, exist_ok=True)

pred_arrs_Bern_1000 = make_predictions('dynamic', images[0:19], 0, [[4, 7]],
                                 stochastic_bin_param=True,
                                 noise_seeds=1000, before_sum=False)
save_predictions(pred_arrs_Bern_1000, path, 'Bern_1000.npy', dirs = ['sum', 'mean', 'var', 'label'])


pred_arrs_Gaus_1000 = make_predictions('dynamic', images[0:19], 10, [[4, 7]],
                                    stochastic_bin_param=False,
                                    noise_seeds=1000, before_sum=False)
save_predictions(pred_arrs_Gaus_1000, path, 'Gaus_10_1000.npy', dirs = ['sum', 'mean', 'var', 'label'])



####pixel space##################

##collect noise images for 19 first images
all_images_noise =  process_images_batched(
        images[0:19],
        num_trials = 1000,
        num_frames = 15,
        noise_type = "dynamic",
        stochastic_bin_param = True,
        sigma= 10,
    return_sum_over_frames = True
)

#save images after noise
path = wd + '/predictions/Anton/1000_trials/'
np.save(path + 'all_images_noise_Bern.npy', all_images_noise)



#plot 9 frames as 3x3 grid

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(all_images_noise[4,1,:,:,:][i * 3 + j], cmap='gray')
        axs[i, j].axis('off')
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(final_output[4,1,:,:,:][i * 3 + j], cmap='gray')
        axs[i, j].axis('off')
plt.show()




fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(image_noise_Gaussian[i * 3 + j], cmap='gray')
        axs[i, j].axis('off')
plt.show()


fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(image_sample[i * 3 + j], cmap='gray')
        axs[i, j].axis('off')
plt.show()


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
pred_arrs_const = make_predictions('dynamic', images[0:4], 0, [[4, 7]],
                                 stochastic_bin_param=False,
                                 noise_seeds=2, before_sum=True)

plt.plot(pred_arrs_const[0][0, 0, :, 13])
plt.plot(pred_arrs_Bern[0][0, 0, :, 13])
plt.plot(pred_arrs_Bern[0][0, 1, :, 13])
plt.plot(pred_arrs_Bern[0][0, 2, :, 13])


plt.plot(pred_arrs_const[0][0, 0, :, 145])
plt.plot(pred_arrs_Bern[0][0, 0, :, 145])
plt.plot(pred_arrs_Bern_blank[0][0, 0, :, 145])