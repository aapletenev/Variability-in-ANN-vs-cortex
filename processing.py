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
        mapping output from scans()
    array: np.array
        array to filter

    Returns
    -------
    array
        filtered array where brain region is equal to region_input
    """
    filtered = [array[i][label_mapping[i] == region_input] for i in range(array.shape[0])]
    return np.stack(filtered, axis=0)

def sort_avg_arr(arr, ascending: bool = True, axis = 1):
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

    arr_averaged = np.nanmean(arr, axis = axis)
    if ascending: return np.argsort(arr_averaged)
    else: return np.argsort(arr_averaged)[::-1]


def sort_within_arr(arr, ascending = True):
    """
    sort row of data within array
    
    parameters
    ----------
    arr: np.ndarray
        array to sort
    ascending: bool
        whether to sort ascending or not
    axis: 
        defaults to -1 since input will likely be row of data
    returns
    -------
    list
        list of indices to sort row of data
    """
    if ascending: return np.argsort(arr)
    else: return np.argsort(arr)[::-1]