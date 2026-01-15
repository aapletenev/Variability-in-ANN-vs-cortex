import numpy as np
import pandas as pd
from pathlib import Path
from figure1_utils import figure1_collection
from plot_figure1 import plot_figure1

# Path where predictions are stored
path = "predictions/test_predictions(1-8-26)"
#path = "predictions/final predictions"
test_region = 2
file_name = "figure1"

# Initialize empty lists to collect arrays
bern_arrays = []
gaus_arrays = []

# Iterate through all .npy files in the path
for file in Path(path).glob("*.npy"):
    # Get filename and split on underscore
    filename = file.stem  # Gets filename without extension
    parts = filename.split("_")
    
    # Load the array
    arr = np.load(file, allow_pickle=True)
    
    # Check the first part and append to appropriate list
    if parts[0] == "bern":
        bern_arrays.append(arr)
    elif parts[0] == "gaus":
        gaus_arrays.append(arr)

# Concatenate along axis 0
bern_predictions = np.concatenate(bern_arrays, axis=0) if bern_arrays else np.array([])
gaus_predictions = np.concatenate(gaus_arrays, axis=0) if gaus_arrays else np.array([])

print(f"bern shape: {bern_predictions.shape}\n"
    f"gaus shape: {gaus_predictions.shape}")

# ==============================================================================
# FIGURE 1 GENERATION - COMPLETE WORKFLOW
# ==============================================================================

print("=" * 80)
print("FIGURE 1 GENERATION - COMPLETE WORKFLOW")
print("=" * 80)
print()

# Step 1: Calculate neuron-level statistics (mean and variance across noise seeds - axis 1)
print("Step 1: Calculating neuron-level statistics...")
stochbin_meanv1 = np.mean(bern_predictions, axis=1)  # Shape: (num_images, num_neurons)
stochbin_varv1 = np.var(bern_predictions, axis=1)    # Shape: (num_images, num_neurons)
sigma10_meanv1 = np.mean(gaus_predictions, axis=1)   # Shape: (num_images, num_neurons)
sigma10_varv1 = np.var(gaus_predictions, axis=1)     # Shape: (num_images, num_neurons)

print(f"\nNeuron data shapes:")
print(f"  Stochbin mean: {stochbin_meanv1.shape}")
print(f"  Sigma10 mean: {sigma10_meanv1.shape}")
print()

# Step 2: Calculate sums for pixel-space statistics
print("Step 2: Collecting pixel-space statistics...")
stochbin_sum = np.sum(bern_predictions, axis=1)  # Shape: (num_images, num_neurons)
sigma10_sum = np.sum(gaus_predictions, axis=1)   # Shape: (num_images, num_neurons)

pixel_data = figure1_collection(
    stochbin_sum=stochbin_sum,
    sigma10_sum=sigma10_sum,
    num_imgs=5,
    num_noise_seeds=100,
    num_frames=15,
    img_height=144,
    img_width=256,
    relu_threshold=128,
    image_stack_path = "image_stacks/final_image_stack/image_stack(12-24-2025).npy"
)
print()

# Step 3: Generate the figure (fits computed internally)
print("Step 3: Generating Figure 1...")
print("  (This may take a moment...)")

# Note: Skipping region filtering for this test since we don't have the proper label mapping
# In production, you would load and filter region_labels_df to match the neurons in predictions

fig = plot_figure1(f"{file_name}(region {test_region})",
    stochbin_meanv1, stochbin_varv1,
    sigma10_meanv1, sigma10_varv1,
    pixel_data['gnoise_mean'], pixel_data['gnoise_var'],
    pixel_data['bnoise_mean'], pixel_data['bnoise_var'],
    pixel_data['fg_mean_relu'], pixel_data['fg_var_relu'],
    pixel_data['fb_mean_relu'], pixel_data['fb_var_relu'],
    None, test_region,  # region_labels_df=None to skip filtering
    ve_thresh=0.1, neuron_idx=2, savefig=True
)

print("\n✓ Figure 1 generated successfully!")
print()
print("=" * 80)
print("WORKFLOW COMPLETE!")
print("=" * 80)

