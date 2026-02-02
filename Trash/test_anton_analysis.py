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

# Parameters
# BATCHES OCCUR IN MULTIPLES OF 100
start_batch = 0   # Starting batch index (0-based)
end_batch = 1     # Ending batch index (exclusive), None for all remaining batches
# Example: start_batch=0, end_batch=2 processes batches 0-1 (images 0-199)
# Example: start_batch=5, end_batch=10 processes batches 5-9 (images 500-999)
# Example: start_batch=0, end_batch=None processes all batches

image_stacks_dir = 'image_stacks/dec3_splitup'
output_dir = 'predictions/incremental_saves'
os.makedirs(output_dir, exist_ok=True)

# Get all batch folders
batch_folders = sorted([f for f in os.listdir(image_stacks_dir) 
                       if os.path.isdir(os.path.join(image_stacks_dir, f))])

# Select range of batches
if end_batch is None:
    selected_batches = batch_folders[start_batch:]
else:
    selected_batches = batch_folders[start_batch:end_batch]

print(f"Processing batches {start_batch} to {end_batch if end_batch else len(batch_folders)}")
print(f"Total: {len(selected_batches)} batches")

# Process each batch folder
for batch_idx, batch_folder in enumerate(selected_batches, start=start_batch):
    batch_path = os.path.join(image_stacks_dir, batch_folder)
    
    # Find the .npy file in this batch folder
    npy_files = [f for f in os.listdir(batch_path) if f.endswith('.npy')]
    if len(npy_files) == 0:
        print(f"No .npy file found in {batch_folder}, skipping...")
        continue
    
    # Load the image batch
    image_batch_path = os.path.join(batch_path, npy_files[0])
    image_batch = np.load(image_batch_path)
    
    print(f"[Batch {batch_idx}] Processing {batch_folder}: shape {image_batch.shape}")
    
    # Make predictions on this batch
    og_arr = make_predictions('dynamic', image_batch, 3, [[4,7]], 
                                   stochastic_bin_param=False, 
                                   noise_seeds=2, 
                                   num_frames=15, return_before_sum=True)
    
    # Create output folder for this batch
    output_batch_folder = os.path.join(output_dir, batch_folder)
    os.makedirs(output_batch_folder, exist_ok=True)
    
    # Save predictions
    np.save(f'{output_batch_folder}/predictions.npy', pred_arrs[0])
    
    print(f"[Batch {batch_idx}] Saved predictions to {output_batch_folder}")

print(f"\nAll predictions saved to {output_dir}/")
print(f"Processed batches {start_batch} to {batch_idx}")