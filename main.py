import numpy as np
from prediction import make_predictions

"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
3. note where image_stack is saved, this is imagenet-mini file from kaggle
"""

image_stack = np.load('image_stacks/final_image_stack/image_stack(12-24-2025).npy')
gray_frames = np.full((1, 144, 256), 128)
image_stack = np.concatenate((gray_frames, image_stack), 0)

n_images = 3 # adjust based on TOTAL number of images wanted for one prediction batch
batch_size = 1
save_folder = "predictions/test_predictions(1-8-26)"
beg_index = 0 # put where you want to start batch collection

for batch_idx in range(beg_index, n_images, batch_size): 

	end_idx = min(batch_idx + batch_size, n_images)
	
	image_batch = image_stack[batch_idx: end_idx]	
# just save sum (first array ) here to save storage, also note regions not needed we can reuse from last time
	a = make_predictions(noise_type = 'dynamic', images = image_batch, sigma = 10, scans = [[4,7]], 
					  stochastic_bin_param = True, noise_seeds=100, num_frames=15, return_before_sum = False)
	sum_arr, label_arr = a[0], a[3]
	np.save(f"{save_folder}/bern_sum({batch_idx})", sum_arr)
	np.save(f"{save_folder}/bern_labels({batch_idx})", label_arr)

	print(f"---SAVED BATCH {batch_idx}---")
