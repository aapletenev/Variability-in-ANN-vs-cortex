import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "predictions/final predictions"

# Initialize empty lists to collect arrays
bern_arrays = []
gaus_arrays = []

# Iterate through all .npy files in the path
for file in Path(path).glob("*.npy"):
    # Get filename and split on underscore
    filename = file.stem  # Gets filename without extension
    parts = filename.split("_")
    
    # Load the array
    arr = np.load(file)
    
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

# Calculate mean and variance on axis 1
bern_mean = np.mean(bern_predictions, axis=1)
bern_var = np.var(bern_predictions, axis=1)
gaus_mean = np.mean(gaus_predictions, axis=1)
gaus_var = np.var(gaus_predictions, axis=1)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Regular plot
axes[0].scatter(bern_mean, bern_var, alpha=0.3, label='Bern', marker=".", color = "tab:orange")
axes[0].scatter(gaus_mean, gaus_var, alpha=0.3, label='Gaus', marker=".", color = "tab:blue")
axes[0].set_xlabel('Mean')
axes[0].set_ylabel('Variance')
axes[0].set_title('Mean vs Variance (Regular Scale)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log-log plot
axes[1].scatter(bern_mean, bern_var, alpha=0.3, label='Bern', marker=".", color = "tab:orange")
axes[1].scatter(gaus_mean, gaus_var, alpha=0.3, label='Gaus', marker=".", color = "tab:blue")
axes[1].set_xlabel('Mean')
axes[1].set_ylabel('Variance')
axes[1].set_title('Mean vs Variance (Log-Log Scale)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

