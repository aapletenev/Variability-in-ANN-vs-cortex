"""
This module contains all the data loading and preprocessing functions needed
to reproduce the analysis.

Example usage at line 460.
"""

import numpy as np
from processing import filter_region
from noise_seeds import noise_iterations


def load_arrs(arr_type: str, folder_path: str) -> np.ndarray:
    """
    Load prediction arrays from saved numpy files.
    
    Parameters:
    -----------
    arr_type : str
        Type of array to load: 'sum', 'mean', 'var', or 'label'
    folder_path : str
        Path to folder containing the prediction arrays
        
    Returns:
    --------
    np.ndarray
        Loaded array with shape (110, 100, 7493) for 'sum' or (110, 7493) for others
        
    Raises:
    -------
    ValueError
        If arr_type is not one of the valid options
    """
    if arr_type not in ['sum', 'mean', 'var', 'label']:
        raise ValueError('---Insert "sum" or "mean" "var" or "label" as input---')
    
    return_arr = np.empty((110, 100, 7493)) if arr_type == 'sum' else np.empty((110, 7493))
    
    for arr_index in range(110):
        arr_temp = np.load(f'{folder_path}image{arr_index}/{arr_type}_arr_image{arr_index}.npy')
        return_arr[arr_index] = arr_temp
    
    return return_arr


def load_prediction_data(folder_path_stochbin: str = 'predictions/8-27-2025/',
                         folder_path_sigma10: str = 'predictions/sigma10') -> dict:
    """
    Load all prediction arrays for both stochastic binary and sigma10 noise.
    
    Parameters:
    -----------
    folder_path_stochbin : str, default='predictions/8-27-2025/'
        Path to stochastic binary prediction folder
    folder_path_sigma10 : str, default='predictions/sigma10'
        Path to sigma10 prediction folder
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'stochbin_sum': Sum array for stochastic binary
        - 'stochbin_mean': Mean array for stochastic binary
        - 'stochbin_var': Variance array for stochastic binary
        - 'stochbin_label': Label array for stochastic binary
        - 'sigma10_sum': Sum array for sigma10
        - 'sigma10_mean': Mean array for sigma10
        - 'sigma10_var': Variance array for sigma10
        - 'sigma10_label': Label array for sigma10
    """
    # Load stochastic binary arrays
    stochbin_sum = load_arrs('sum', folder_path_stochbin)
    stochbin_mean = load_arrs('mean', folder_path_stochbin)
    stochbin_var = load_arrs('var', folder_path_stochbin)
    stochbin_label = load_arrs('label', folder_path_stochbin)
    
    # Load sigma10 arrays
    sigma10_sum = np.load(folder_path_sigma10 + "/sum_sigma10.npy")
    sigma10_mean = np.load(folder_path_sigma10 + "/mean_sigma10.npy")
    sigma10_var = np.load(folder_path_sigma10 + "/var_sigma10.npy")
    sigma10_label = np.load(folder_path_sigma10 + "/labels_sigma10.npy")
    
    return {
        'stochbin_sum': stochbin_sum,
        'stochbin_mean': stochbin_mean,
        'stochbin_var': stochbin_var,
        'stochbin_label': stochbin_label,
        'sigma10_sum': sigma10_sum,
        'sigma10_mean': sigma10_mean,
        'sigma10_var': sigma10_var,
        'sigma10_label': sigma10_label
    }


def filter_x_region(region, stochbin_mean: np.ndarray, stochbin_var: np.ndarray,
                     region_labels_df, sigma10_mean: np.ndarray,
                     sigma10_var: np.ndarray, 
                     encoding: dict = {'V1':1, 'LM':2, 'AL':3, 'RL':4}) -> dict:
    """
    Filter V1 region (region 1) from the prediction arrays.
    
    Parameters:
    -----------
    region : int
        Region number to filter (1=V1, 2=LM, 3=AL, 4=RL)
    stochbin_mean : np.ndarray
        Mean array for stochastic binary
    stochbin_var : np.ndarray
        Variance array for stochastic binary
    region_labels_df : pandas.DataFrame
        DataFrame containing region labels for all neurons with 'brain_area' column
    sigma10_mean : np.ndarray
        Mean array for sigma10
    sigma10_var : np.ndarray
        Variance array for sigma10
    encoding : dict, default={'V1':1, 'LM':2, 'AL':3, 'RL':4}
        Mapping from brain area names to numeric labels
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'v1mean_stochbin': V1 mean for stochastic binary
        - 'v1var_stochbin': V1 variance for stochastic binary
        - 'v1mean_sigma10': V1 mean for sigma10
        - 'v1var_sigma10': V1 variance for sigma10
    """
    # Extract brain_area column and encode to numeric labels
    if hasattr(region_labels_df, 'brain_area'):
        # It's a DataFrame - extract and encode the brain_area column
        stochbin_label = region_labels_df['brain_area'].replace(encoding).values
    elif hasattr(region_labels_df, 'replace'):
        # It's a Series - encode it directly
        stochbin_label = region_labels_df.replace(encoding).values
    else:
        # It's already a numpy array
        stochbin_label = region_labels_df
    
    print("^"*80)
    print(f"TEST DIAGNOSTICS\nregion: {region}, label shape: {stochbin_label.shape}, arr shape: {stochbin_mean.shape}")
    print("^"*80)
    v1mean_stochbin = filter_region(region, stochbin_label, stochbin_mean)
    v1var_stochbin = filter_region(region, stochbin_label, stochbin_var)
    
    # Using labels from stochbin for sigma10 as well (same neurons)
    v1mean_sigma10 = filter_region(region, stochbin_label, sigma10_mean)
    v1var_sigma10 = filter_region(region, stochbin_label, sigma10_var)
    
    return {
        'v1mean_stochbin': v1mean_stochbin,
        'v1var_stochbin': v1var_stochbin,
        'v1mean_sigma10': v1mean_sigma10,
        'v1var_sigma10': v1var_sigma10
    }


def remove_grey_images(v1mean_stochbin: np.ndarray, v1var_stochbin: np.ndarray,
                       v1mean_sigma10: np.ndarray, v1var_sigma10: np.ndarray,
                       num_grey_images: int = 9) -> dict:
    """
    Remove grey images from the beginning of the arrays.
    
    Parameters:
    -----------
    v1mean_stochbin : np.ndarray
        V1 mean for stochastic binary
    v1var_stochbin : np.ndarray
        V1 variance for stochastic binary
    v1mean_sigma10 : np.ndarray
        V1 mean for sigma10
    v1var_sigma10 : np.ndarray
        V1 variance for sigma10
    num_grey_images : int, default=9
        Number of grey images to remove from the beginning
        
    Returns:
    --------
    dict
        Dictionary containing filtered arrays with grey images removed
    """
    return {
        'v1mean_stochbin': v1mean_stochbin[num_grey_images:, ...],
        'v1var_stochbin': v1var_stochbin[num_grey_images:, ...],
        'v1mean_sigma10': v1mean_sigma10[num_grey_images:, ...],
        'v1var_sigma10': v1var_sigma10[num_grey_images:, ...]
    }


def apply_threshold_masking(v1mean_stochbin: np.ndarray, v1var_stochbin: np.ndarray,
                            v1mean_sigma10: np.ndarray, v1var_sigma10: np.ndarray,
                            threshold: float = 100.0) -> dict:
    """
    Apply threshold masking to remove values above the specified threshold.
    Values above threshold are replaced with NaN.
    
    Parameters:
    -----------
    v1mean_stochbin : np.ndarray
        V1 mean for stochastic binary
    v1var_stochbin : np.ndarray
        V1 variance for stochastic binary
    v1mean_sigma10 : np.ndarray
        V1 mean for sigma10
    v1var_sigma10 : np.ndarray
        V1 variance for sigma10
    threshold : float, default=100.0
        Threshold value (values > threshold are set to NaN)
        
    Returns:
    --------
    dict
        Dictionary containing masked arrays
    """
    # Create copies to avoid modifying original arrays
    v1mean_stochbin = v1mean_stochbin.copy()
    v1var_stochbin = v1var_stochbin.copy()
    v1mean_sigma10 = v1mean_sigma10.copy()
    v1var_sigma10 = v1var_sigma10.copy()
    
    # Create masks
    mask_gaus = v1mean_sigma10 <= threshold
    mask_bernoulli = v1mean_stochbin <= threshold
    
    # Apply masks
    v1mean_sigma10[~mask_gaus] = np.nan
    v1var_sigma10[~mask_gaus] = np.nan
    v1mean_stochbin[~mask_bernoulli] = np.nan
    v1var_stochbin[~mask_bernoulli] = np.nan
    
    return {
        'v1mean_stochbin': v1mean_stochbin,
        'v1var_stochbin': v1var_stochbin,
        'v1mean_sigma10': v1mean_sigma10,
        'v1var_sigma10': v1var_sigma10
    }


def sort_by_median(v1mean_stochbin: np.ndarray, v1var_stochbin: np.ndarray,
                   v1mean_sigma10: np.ndarray, v1var_sigma10: np.ndarray) -> dict:
    """
    Sort neurons by median activity across images.
    
    Parameters:
    -----------
    v1mean_stochbin : np.ndarray
        V1 mean for stochastic binary
    v1var_stochbin : np.ndarray
        V1 variance for stochastic binary
    v1mean_sigma10 : np.ndarray
        V1 mean for sigma10
    v1var_sigma10 : np.ndarray
        V1 variance for sigma10
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'stochbin_meanv1': Sorted stochastic binary means
        - 'stochbin_varv1': Sorted stochastic binary variances
        - 'sigma10_meanv1': Sorted sigma10 means
        - 'sigma10_varv1': Sorted sigma10 variances
        - 'stochbin_sort_indices': Sort indices for stochastic binary
        - 'sigma10_sort_indices': Sort indices for sigma10
    """
    # Compute median activity
    stochbin_med = np.nanmedian(v1mean_stochbin, axis=0)
    sigma10_med = np.nanmedian(v1mean_sigma10, axis=0)
    
    # Get sort indices (descending order)
    stochbin_sort_indices = np.argsort(stochbin_med)[::-1]
    sigma10_sort_indices = np.argsort(sigma10_med)[::-1]
    
    # Apply sorting
    stochbin_meanv1 = v1mean_stochbin[..., stochbin_sort_indices]
    stochbin_varv1 = v1var_stochbin[..., stochbin_sort_indices]
    sigma10_meanv1 = v1mean_sigma10[..., sigma10_sort_indices]
    sigma10_varv1 = v1var_sigma10[..., sigma10_sort_indices]
    
    return {
        'stochbin_meanv1': stochbin_meanv1,
        'stochbin_varv1': stochbin_varv1,
        'sigma10_meanv1': sigma10_meanv1,
        'sigma10_varv1': sigma10_varv1,
        'stochbin_sort_indices': stochbin_sort_indices,
        'sigma10_sort_indices': sigma10_sort_indices
    }


def preprocess_neuron_data(folder_path_stochbin: str,
                          folder_path_sigma10: str,
                          threshold: float = 100.0,
                          num_grey_images: int = 0) -> dict:
    """
    Complete preprocessing pipeline for neuron data.
    
    This function combines all preprocessing steps:
    1. Load prediction data
    2. Filter V1 region
    3. Remove grey images
    4. Apply threshold masking
    5. Sort by median activity
    
    Parameters:
    -----------
    folder_path_stochbin : str, default='predictions/8-27-2025/'
        Path to stochastic binary prediction folder
    folder_path_sigma10 : str, default='predictions/sigma10'
        Path to sigma10 prediction folder
    threshold : float, default=100.0 (biologically unplausible spike rate)
        Threshold value for masking (values > threshold are set to NaN)
    num_grey_images : int, default=0 (for our analysis we needed this at one point, should be irrelevant for most reproduction)
        Number of grey images to remove from the beginning
        
    Returns:
    --------
    dict
        Dictionary containing all preprocessed arrays and metadata:
        - Raw data: stochbin_sum, stochbin_mean, stochbin_var, stochbin_label,
                    sigma10_sum, sigma10_mean, sigma10_var, sigma10_label
        - Sorted data: stochbin_meanv1, stochbin_varv1, sigma10_meanv1, sigma10_varv1
        - Sort indices: stochbin_sort_indices, sigma10_sort_indices
    """
    # Step 1: Load data
    print("Loading prediction data...")
    data = load_prediction_data(folder_path_stochbin, folder_path_sigma10)
    
    # Step 2: Filter V1 region
    print("Filtering V1 region...")
    v1_data = filter_v1_region(
        data['stochbin_mean'], data['stochbin_var'], data['stochbin_label'],
        data['sigma10_mean'], data['sigma10_var']
    )
    
    # Step 3: Remove grey images
    print(f"Removing {num_grey_images} grey images...")
    v1_data = remove_grey_images(
        v1_data['v1mean_stochbin'], v1_data['v1var_stochbin'],
        v1_data['v1mean_sigma10'], v1_data['v1var_sigma10'],
        num_grey_images
    )
    
    # Step 4: Apply threshold masking
    print(f"Applying threshold masking (threshold={threshold})...")
    v1_data = apply_threshold_masking(
        v1_data['v1mean_stochbin'], v1_data['v1var_stochbin'],
        v1_data['v1mean_sigma10'], v1_data['v1var_sigma10'],
        threshold
    )
    
    # Step 5: Sort by median
    print("Sorting by median activity...")
    sorted_data = sort_by_median(
        v1_data['v1mean_stochbin'], v1_data['v1var_stochbin'],
        v1_data['v1mean_sigma10'], v1_data['v1var_sigma10']
    )
    
    # Combine all data
    result = {**data, **sorted_data}
    
    print("Preprocessing complete!")
    print(f"  Stochbin mean shape: {result['stochbin_meanv1'].shape}")
    print(f"  Sigma10 mean shape: {result['sigma10_meanv1'].shape}")
    
    return result


def figure1_collection(stochbin_sum: np.ndarray, sigma10_sum: np.ndarray, 
                       num_imgs: int = 100, num_noise_seeds: int = 10, num_frames: int = 15,
                       img_height: int = 144, img_width: int = 256, relu_threshold: int = 128,
                       image_stack_path: str = "image_stacks/imagestack_nov232025(first100).npy") -> dict:
    """
    Collect and compute pixel-space statistics for Figure 1.
    
    Parameters:
    -----------
    stochbin_sum : np.ndarray
        Sum array for stochastic binary (Bernoulli) noise, shape (num_imgs, num_neurons)
    sigma10_sum : np.ndarray
        Sum array for Gaussian noise (sigma=10), shape (num_imgs, num_neurons)
    num_imgs : int, default=100
        Number of images
    num_noise_seeds : int, default=10
        Number of noise seeds
    num_frames : int, default=15
        Number of frames to sum over
    img_height : int, default=144
        Height of images in pixels
    img_width : int, default=256
        Width of images in pixels
    relu_threshold : int, default=128
        Threshold for ReLU operation
    image_stack_path : str, default="image_stacks/imagestack_nov232025(first100).npy"
        Path to the image stack file
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'gnoise_mean': Gaussian noise mean (flattened)
        - 'gnoise_var': Gaussian noise variance (flattened)
        - 'bnoise_mean': Bernoulli noise mean (flattened)
        - 'bnoise_var': Bernoulli noise variance (flattened)
        - 'fg_mean_relu': Gaussian ReLU mean (flattened)
        - 'fg_var_relu': Gaussian ReLU variance (flattened)
        - 'fb_mean_relu': Bernoulli ReLU mean (flattened)
        - 'fb_var_relu': Bernoulli ReLU variance (flattened)
    """
    total_pixels = img_height * img_width
    
    # Load image data
    print(f"Loading image stack from {image_stack_path}...")
    image_df = np.load(image_stack_path)
    
    def get_noise_per_image(img, num_frames, num_noise_seeds, noise_type: str, sigma: int, stoch_bin_param):
        """Generate noise for a single image across multiple frames and noise seeds."""
        if img.ndim != 3:
            img = np.array(img).reshape((1, img.shape[0], img.shape[1]))
        tiled_img = np.tile(img, (num_frames, 1, 1))
        return np.array(noise_iterations(
            "_", "_", noise_type, num_noise_seeds, tiled_img, sigma,
            stochastic_bin_param=stoch_bin_param,
            scans=[[4, 7]], 
            num_frames=num_frames, 
            return_noise=True
        ))
    
    # Generate 15-frame noise arrays
    print("Generating noise arrays...")
    noise_arr_gaus = np.empty((num_imgs, num_noise_seeds, num_frames, img_height, img_width))
    noise_arr_bern = np.empty((num_imgs, num_noise_seeds, num_frames, img_height, img_width))
    
    for i in range(num_imgs):
        if i % 10 == 0:
            print(f"  Processing image {i}/{num_imgs}...")
        noise_arr_gaus[i] = get_noise_per_image(image_df[i], num_frames, num_noise_seeds, "dynamic", 10, False)
        noise_arr_bern[i] = get_noise_per_image(image_df[i], num_frames, num_noise_seeds, "dynamic", 10, True)
    
    # Reshape to flatten spatial dimensions
    print("Computing statistics...")
    noise_arr_fg = noise_arr_gaus.reshape(num_imgs, num_noise_seeds, num_frames, total_pixels).astype(np.float32)
    noise_arr_bern = noise_arr_bern.reshape(num_imgs, num_noise_seeds, num_frames, total_pixels).astype(np.float32)
    
    # Compute mean and variance - Regular data (15 frames summed)
    gaus_summed = np.sum(noise_arr_fg, axis=2)
    gnoise_mean = np.mean(gaus_summed, axis=1).flatten()
    gnoise_var = np.var(gaus_summed, axis=1).flatten()
    
    bern_summed = np.sum(noise_arr_bern, axis=2)
    bnoise_mean = np.mean(bern_summed, axis=1).flatten()
    bnoise_var = np.var(bern_summed, axis=1).flatten()
    
    # Compute mean and variance - ReLU data (15 frames summed)
    fg_relu = np.where(noise_arr_fg >= relu_threshold, noise_arr_fg - relu_threshold, 0)
    grelu_sum = np.sum(fg_relu, axis=2)
    fg_mean_relu = np.mean(grelu_sum, axis=1).flatten()
    fg_var_relu = np.var(grelu_sum, axis=1).flatten()
    
    fb_relu = np.where(noise_arr_bern >= relu_threshold, noise_arr_bern - relu_threshold, 0)
    fb_relu_sum = np.sum(fb_relu, axis=2)
    fb_mean_relu = np.mean(fb_relu_sum, axis=1).flatten()
    fb_var_relu = np.var(fb_relu_sum, axis=1).flatten()
    
    print(f"Pixel-space data collection complete:")
    print(f"  Gaussian noise: {gnoise_mean.shape}")
    print(f"  Bernoulli noise: {bnoise_mean.shape}")
    print(f"  Gaussian ReLU: {fg_mean_relu.shape}")
    print(f"  Bernoulli ReLU: {fb_mean_relu.shape}")
    
    return {
        'gnoise_mean': gnoise_mean,
        'gnoise_var': gnoise_var,
        'bnoise_mean': bnoise_mean,
        'bnoise_var': bnoise_var,
        'fg_mean_relu': fg_mean_relu,
        'fg_var_relu': fg_var_relu,
        'fb_mean_relu': fb_mean_relu,
        'fb_var_relu': fb_var_relu
    }


if __name__ == "__main__":
    """
    Example usage of the preprocessing pipeline.
    """
    # Preprocess neuron data
    neuron_data = preprocess_neuron_data(
        folder_path_stochbin='predictions/8-27-2025/',
        folder_path_sigma10='predictions/sigma10',
        threshold=100.0,
        num_grey_images=9
    )
    
    # Collect pixel-level statistics
    pixel_data = figure1_collection(
        stochbin_sum=neuron_data['stochbin_sum'],
        sigma10_sum=neuron_data['sigma10_sum'],
        num_imgs=100,
        num_noise_seeds=10,
        num_frames=15,
        img_height=144,
        img_width=256,
        relu_threshold=128
    )
    
    print("\nAll preprocessing and data collection complete!")
