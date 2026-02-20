#here we do similar analisys as for the whole trial but individually for each frame
%matplotlib qt
import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
from functions_Anton import *
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from pathlib import Path
from figure1_utils import  filter_x_region
import pandas as pd




############Start#################################
##load the data
wd = os.getcwd()


# Path where predictions are stored
# path = "predictions/test_predictions(1-8-26)"
path = "predictions/final predictions"
test_region = 1
ADD_Poisson = True  # Whether to add Poisson on top

if ADD_Poisson:
    file_name_add = "_with_Poisson"
else:
    file_name_add = ""

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


# Step 1: Calculate neuron-level statistics (mean and variance across noise seeds - axis 1)
print("Step 1: Calculating neuron-level statistics...")
stochbin_meanv1 = np.mean(bern_predictions, axis=1)  # Shape: (num_images, num_neurons)
stochbin_varv1 = np.var(bern_predictions, axis=1)  # Shape: (num_images, num_neurons)
sigma10_meanv1 = np.mean(gaus_predictions, axis=1)  # Shape: (num_images, num_neurons)
sigma10_varv1 = np.var(gaus_predictions, axis=1)  # Shape: (num_images, num_neurons)

# nan for all if mean > 100
stochbin_meanv1[stochbin_meanv1 > 100] = np.nan
stochbin_varv1[stochbin_meanv1 > 100] = np.nan
sigma10_meanv1[sigma10_meanv1 > 100] = np.nan
sigma10_varv1[sigma10_meanv1 > 100] = np.nan

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


#for sum over frames
slopes_Bern_sum, r2_Bern_sum = compute_slope_var_mean(filtered['v1mean_stochbin'] [:, np.newaxis, :],
                                                      filtered['v1var_stochbin'][:, np.newaxis, :])
slopes_Gaus_sum, r2_Gaus_sum = compute_slope_var_mean(filtered['v1mean_sigma10'] [:, np.newaxis, :],
                                                      filtered['v1var_sigma10'][:, np.newaxis, :])




###plot fano factor slopes only
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
plot_dual_axis_boxplot(axes, slopes_Bern_sum, slopes_Gaus_sum,
                       title="Fano Factor",
                       ylim=0.9, whisk=(25, 50), scale = 2, xlabel = "", ylabel_suffix = "", use_dual_axis= not ADD_Poisson,
                       show_xticks=False)
plt.tight_layout()
plt.savefig(wd + '/plots/Anton/FF'+ file_name_add + '.pdf')

#summary of slopes_Bern_sum - mean, median,1 and 3rd quartiles
print(f"\nSummary of Fano Factor slopes (Bernoulli noise):\n"
      f"  Mean: {np.nanmean(slopes_Bern_sum):.4f}\n"
      f"  Median: {np.nanmedian(slopes_Bern_sum):.4f}\n"
      f"  1st Quartile: {np.nanpercentile(slopes_Bern_sum, 25):.4f}\n"
      f"  3rd Quartile: {np.nanpercentile(slopes_Bern_sum, 75):.4f}")