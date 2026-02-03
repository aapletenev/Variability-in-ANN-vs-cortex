import numpy as np
import pandas as pd
from pathlib import Path
from figure1_utils import figure1_collection, filter_x_region
from plot_figure1 import plot_figure1


# ==============================================================================
# LOAD/PROCESS DATA - COMPLETE WORKFLOW
# ==============================================================================

# Path where predictions are stored
#path = "predictions/test_predictions(1-8-26)"
path = "predictions/final predictions"
test_region = 1
ADD_Poisson = True  # Whether to add Poisson on top

# Initialize empty lists to collect arrays
bern_arrays = []
gaus_arrays = []


# Iterate through all .npy files in the path
for file in Path(path).glob("*.npy"):
    # Get filename and split on underscore
    filename = file.stem  # Gets filename without extension
    parts = filename.split("_")
    
    # Only process sum arrays (skip mean, var, label)
    if "sum" not in filename:
        continue
    
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

if ADD_Poisson:
    print("\nAdding Poisson noise to ...")
    bern_predictions = np.random.poisson(lam=bern_predictions)
    gaus_predictions = np.random.poisson(lam=gaus_predictions)


    print("Poisson noise added.\n")



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

#nan for all if mean > 100
stochbin_meanv1[stochbin_meanv1>100] = np.nan
stochbin_varv1 [stochbin_meanv1>100] = np.nan
sigma10_meanv1[sigma10_meanv1>100] = np.nan
sigma10_varv1 [sigma10_meanv1>100] = np.nan



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

# Load brain region labels from microns_area_labels.csv
path_regions_csv = "microns_area_labels.csv"
regions_df = pd.read_csv(path_regions_csv)

# Region mapping: {'V1':1, 'LM':2, 'AL':3, 'RL':4}
# Test with region 2 = LM
region_name_map = {1: 'V1', 2: 'LM', 3: 'AL', 4: 'RL'}
print(f"\nFiltering for region {test_region} ({region_name_map[test_region]})")

##find the topk neurons with highest median of the mean
filtered = filter_x_region(test_region, stochbin_meanv1, stochbin_varv1, regions_df,
                              sigma10_meanv1, sigma10_varv1
                              )
median_Bern = np.nanmedian(filtered['v1mean_stochbin'], axis=0)
median_Sigma10 = np.nanmedian(filtered['v1mean_sigma10'], axis=0)
indices = np.intersect1d(np.argsort(median_Bern)[-100:], np.argsort(median_Sigma10)[-100:])
index = indices[-1]
#V1 index 3350


plot_p = {"Bern_spike_limit":150, "Gaus_spike_limit":150,
        "Gaus_log_limit":(0.3,300), "Bern_log_limit":(0.3,300),
        "Kernel_b_limit":(0.8,1.4), "Kernel_r2_ind_xlim":0.6,
          "Kernel_r2_all_xlim":0.6, "Kernel_r2_all_ylim":30}

fig = plot_figure1(f"testjosh_jan21(region_{region_name_map[test_region]})",
    stochbin_meanv1, stochbin_varv1,
    sigma10_meanv1, sigma10_varv1,
    pixel_data['gnoise_mean'], pixel_data['gnoise_var'],
    pixel_data['bnoise_mean'], pixel_data['bnoise_var'],
    pixel_data['fg_mean_relu'], pixel_data['fg_var_relu'],
    pixel_data['fb_mean_relu'], pixel_data['fb_var_relu'],
    regions_df, test_region,
    ve_thresh=0.1, neuron_idx=index, savefig=True,
    plot_parameters=plot_p
)

print("\n[SUCCESS] Figure 1 generated successfully!")
print()
print("=" * 80)
print("WORKFLOW COMPLETE!")
print("=" * 80)

