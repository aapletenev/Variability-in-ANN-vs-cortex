import os 
import numpy as np
import pandas as pd

def remove_x(arr, remove: str, k_percent: int):
    """
    Parameters
    ----------
    arr: 
        object to remove x% from
    remove: str
        for now only top used, removes top k % if "top" otherwise bottom k % 
    k_percent: int
        k % to take off top/bottom %

    Returns
    list
        indices of neurons to keep
    """
    if len(arr) <= 1:
        return np.arange(len(arr))  

        
    n_remove = int(k_percent / 100 * len(arr))
    n_keep = max(1, len(arr) - n_remove)
    
    if remove == 'top':
        return arr[:n_keep]
    else:  # remove == 'bottom'
        return arr[-n_keep:]

def ensure_2d_list(scans) -> list:
    """
    Parameters
    ----------
    scans: object
        scans is going to be list with session and scan_id, ie scans = [4,7]
    
    Returns
    -------
    list
        scans as 2d list if it is not one already

    """
    if not isinstance(scans, list) or len(scans) == 0:
        return [scans]
    if not isinstance(scans[0], (list, tuple)):
        return [scans]
    return scans

def get_neuron_units(scans) -> int: # MAKE ADDING PATH TO SCANS PARAMETER
    """
    Parameters
    ----------
    scans: list
        takes scans as input
    
    Returns
    -------
    int
        number of neurons from scans object, needed to preallocate array size for optimal performance. data taken from scans.csv
    """
    scans_df = pd.read_csv('scans.csv')
    final_df = pd.DataFrame({}, columns = ['session', 'scan_idx', 'units', 'data_id'])
    for scan in scans:
        row = scans_df[(scans_df['session'] == scan[0]) & (scans_df['scan_idx'] == scan[1])]
        final_df = pd.concat([final_df, row])
    return final_df['units'].sum()

def get_brain_region(ids: pd.DataFrame, num_neurons: int, encoding : dict = {'V1':1, 'LM':2, 'AL':3, 'RL':4}, # why num_neurons here?
                     brain_region_path: str = 'brain_region_files//microns_area_labels.csv') -> np.array:
    """
    Parameters
    ----------
    ids: DataFrame
        mapping for readout to unit ids from scan()
    num_neurons: int
        number of neurons taken from scan()
    encoding: dictionary
        default dictionary provided for brain regions V1, LM, AL and RL
    brain_region_path: str
        default path provided for microns_area_labels.csv

    Returns
    array
        array of predictions with brain region column added
    """
    brain_regions = pd.read_csv('microns_area_labels.csv') # make input

    ids_matched = pd.merge(ids, brain_regions, how = 'left', on = ['session', 'scan_idx', 'unit_id'])['brain_area']
    ids_matched = ids_matched.map(encoding)

    # output is a list with one pd.Series in it, keep for now
    return ids_matched
