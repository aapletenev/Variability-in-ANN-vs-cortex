FILE DICTIONARY
---------------

d = dynamic noise
c = constant noise
any number following a d/c indicates the sigma (STD.) for which noise is applied
if bin follows a d/c it means a stochastic binarization was applied to the input images

_sum: this array contains predictions for all images, noise seeds, neurons
_mean: this array contains predictions for all images averaged over noise seeds
_var: this array contains variance of predictions for all images, calculated over noise seeds

for example, d3_mean.npy contains mean spike count predictions (averaged over noise seeds) for a sigma of 3 with dynamic noise added to input

similarly, cbin_sum.npy contains neuron spike predictions with a constant stochastic binarization applied for all images, noise seeds, and neurons


PARAMETERS AND RUNTIME INFORMATION
----------------------------------

for these predictions, image_stack_july16.npy was used where the first image was a full numpy array filled with the value 128 (gray frames)

image_stack_july16.npy images 1-4 -> folder id: 909; image ids :[ 0 11 13 5]

predictions ran for one scan [[4,7]], 15 frames and with 100 noise seeds; runtime was aproximately just under 2 hours