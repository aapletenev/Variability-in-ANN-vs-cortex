import numpy as np
import os


def load_predictions_by_type(pred_dir, array_shape=(5, 100, 7493)):
    """
    Load prediction files from a directory and separate them by type (gaus/bern).
    
    Args:
        pred_dir: Directory containing prediction .npy files
        array_shape: Shape of the arrays to create (default: (5, 100, 7493))
    
    Returns:
        tuple: (gaus_array, bern_array) containing the loaded predictions
    """
    # Initialize arrays
    gaus_arr = np.empty(array_shape)
    bern_arr = np.empty(array_shape)
    
    # Get all .npy files from the directory
    files = [f for f in os.listdir(pred_dir) if f.endswith('.npy')]

    # Load predictions into corresponding arrays
    for filename in files:
        # Split filename on '_' to check the first part
        first_part = filename.split('_')[0]
        
        # Extract index from filename, e.g., 'gaus_sum(0).npy' -> 0
        index = int(filename.split('(')[1].split(')')[0])
        
        if first_part == 'gaus':
            gaus_arr[index] = np.load(os.path.join(pred_dir, filename))
        elif first_part == 'bern':
            bern_arr[index] = np.load(os.path.join(pred_dir, filename))
    
    return gaus_arr, bern_arr


# Load predictions
pred_dir = "predictions/test_predictions(1-8-26)"
jan_gaus, jan_bern = load_predictions_by_type(pred_dir)

final_gaus, final_bern = np.load("predictions/final predictions/bern_sum(0).npy")[:5], np.load("predictions/final predictions/gaus_sum(0).npy")[:5]

print(f"gaus shapes\njan gaus: {jan_gaus.shape}, final gaus: {final_gaus.shape}")

print(f"gaus: {np.unique(jan_gaus==final_gaus, return_counts = True)}, bern: {np.unique(jan_bern==final_bern, return_counts=True)}")