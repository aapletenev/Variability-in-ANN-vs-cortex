This folder contains files for neuron spike count predictions under the following conditions: 
d = dynamic noise
c = constant noise
any number following a d/c indicates the sigma (STD.) for which noise is applied
if bin follows a d/c it means a stochastic binarization was applied to the input images

_sum: this array contains predictions for all images, noise seeds, neurons
_mean: this array contains predictions for all images averaged over noise seeds
_var: this array contains variance in predictions for all images over noise seeds

for example, d3_mean.npy contains mean spike count predictions (averaged over noise seeds) for a sigma of 3 with dynamic noise added to input

similarly, cbin_sum.npy contains neuron spike predictions with a constant stochastic binarization applied for all images, noise seeds, and neurons

for these predictions, image_stack_july16.npy was used where the first image was a full numpy array filled with the value 128 (gray frames):
folder id: 909; image ids :[ 0 11 13 5]
