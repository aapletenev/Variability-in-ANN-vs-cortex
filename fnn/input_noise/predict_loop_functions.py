import os 
import numpy as np
import pandas as pd
from fnn.microns.__init__ import scan

def visual_prediction(session, scan_idx, stimuli_noise):
    """
    Parameters 
    ----------
    session: int
        session number 
    scan_idx: int
        scan id
    stimuli_noise: array
        array/dataframe with added noise    

    Returns
    -------
    array
        neuron predictions
    """
    pred_model, table = scan(session, scan_idx, directory = os.path.join(os.getcwd(), "data","microns")) # look at data/microns/scans.csv for numbers that work
    results = pred_model.predict(stimuli = stimuli_noise)
    return np.array(results) # should be array regardless   

def generate_noise(noise_type, num_frames, sigma, mean = 0): # mean always equal to 0
    """
    Parameters
    ----------
    noise_type: string
        type of noise to produce; dynamic, constant or no noise
    num_frames: int
        number of frames in image input, just first value in input.shape
    sigma: int
        sigma value, std. for gaussian noise distribution and must be non-negative
    mean: int
        mean for gaussian noise distribution

    Returns
    -------
    np.array
        array of specified noise
    """
    if noise_type == "constant":   
        noise = np.random.normal(loc = mean, scale = sigma, size = (144,256))
        return np.stack([noise.copy() for i in range(num_frames)], axis = 0)
    
    elif noise_type == "dynamic": return np.random.normal(loc = mean, scale = sigma, size = (num_frames, 144, 256))
    elif noise_type == "no noise": return np.zeros(shape = (num_frames, 144, 256))
    else: print("\n---Noise input not recognized, please try again---")

def ensure_2d_list(scans):
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

def get_neuron_units(scans):
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
    scans_csv = os.path.join(os.getcwd(),"data\\microns\\scans.csv")
    scans_df = pd.read_csv(scans_csv)
    final_df = pd.DataFrame({}, columns = ['session', 'scan_idx', 'units', 'data_id'])
    for scan in scans:
        row = scans_df[(scans_df['session'] == scan[0]) & (scans_df['scan_idx'] == scan[1])]
        final_df = pd.concat([final_df, row])
    return final_df['units'].sum()
