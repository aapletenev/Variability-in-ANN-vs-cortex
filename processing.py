import numpy as np

def remove_x(arr: list or np.array, remove: str, k_percent: int):
    """
    Parameters
    ----------
    arr: list or np.array
        object that we are removing from
    remove: str
        where to remove values from
    k_percent: int
        what percent to cut off (enter as int, ie if you want to remove 10% enter 10)

    Returns
    -------
    arr
        sorted arr with top/bottom values removed
    indices
        indices for relevant portion of data
    """
    
    if type(arr) == list: arr = np.array(arr)

    indices = np.argsort(arr)
    arr = arr[indices]

    n_remove = int(k_percent / 100 * len(arr))
    n_keep = max(1, len(arr) - n_remove)
    
    if remove == 'top':
        return arr[:n_keep], indices[:n_keep]
    elif remove == 'bottom': 
        return arr[-n_keep:], indices[-n_keep]
    else: raise ValueError("Please enter 'top' or 'bottom' as an argument for remove.")

def filter_region(region_input: int, label_mapping: list | np.ndarray, array):
    """
    Parameters
    ----------
    region_input: int
        specified region to output
    label_mapping: label
        mapping output from scans() - should be 1D array with one entry per neuron
    array: np.array
        array to filter - shape (num_images, num_neurons)

    Returns
    -------
    array
        filtered array where brain region is equal to region_input
    """
    filtered = [array[i][label_mapping == region_input] for i in range(array.shape[0])]
    return np.stack(filtered, axis=0)

def remove_rate(arr, spikes: int, frames: int = 15, seconds: int = 1, fill_value = np.nan):
    """
    remove values that exceed a specified rate, default replace with np.nan
    ie anything above 200 spikes/second becomes nan
    note: 30 frames is one second, function will default to x spikes per second (30 frames)
    
    Parameters
    ----------
    arr: np.array
        array to filter
    spikes: int
        total number of spikes within rate to filter
    frames: int
        frames to filter based off of, defaults to 15 as that is what was used for our predictions
    seconds: int
        seconds to filter based off of for rate threshold
    fill_value: int, double or np.nan
        what to fill new array with, double int or nan will suffice, defaults to nan
    
    Returns
    -------
    arr
        array where any value within arr that EXCEEDS the specified rate is changed to fill_value

    """
    rate_per_frame = (spikes / seconds) * (1 / 30)
    frame_thresh = rate_per_frame * frames
    return np.where(arr > frame_thresh, fill_value, arr)

def sort_max_arr(arr, ascending: bool = True, axis = 1):
    """
    sort multiple rows based on average value for each

    parameters
    ----------
    arr: np.ndarray
        array to sort
    ascending: bool
        whether to sort ascending or not
    axis: int
        axis to sort on, defaults to 1 for 2d array
    
    returns
    -------
    list
        list of indices to sort array
    """

    arr_averaged = np.nanmax(arr, axis = axis)
    if ascending: return np.argsort(arr_averaged)
    else: return np.argsort(arr_averaged)[::-1]
    