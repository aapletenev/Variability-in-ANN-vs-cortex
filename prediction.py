import os 
import numpy as np
from numpy import argpartition
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import sys
from fnn.microns.main import scan
from scipy.stats import shapiro, linregress
from PIL import Image
from data_management import ensure_2d_list, get_neuron_units, get_brain_region
from noise_seeds import noise_iterations


def visual_prediction(session: int, scan_idx: int, stimuli_noise) -> np.array: # make model input here, move scan() to beginning of code
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
    array
        ids with corresponding unit_id for each scan
    """
    pred_model, ids = scan(session, scan_idx, directory = os.path.join(os.getcwd(), "data","microns")) # look at data/microns/scans.csv for numbers that work
    results = pred_model.predict(stimuli = stimuli_noise)
    return results, ids   

def make_predictions(noise_type: str, images: np.ndarray, sigma: int, scans, stochastic_bin_param = False, noise_seeds: int = 100, 
                     num_frames: int = 15, return_before_sum: bool = False):
    
    """
    Parameters
    ----------
    noise_type: string
        dynamic, constant, or no noise
    images: object
        images (usually object from np.stack()) that we are adding noise to and then predicting on
    sigma: int
        standard deviation for gaussian noise distribution
    scans: list/array
        should be 2d array with pairs of sessions and scan ids taken from scans.csv
    stochastic_bin_param: bool
        if stochastic binarization should be performed for noise, default False 
    noise_seeds: int
        how many times we add noise to the prediction
    num_frames: int
        number of frames per image, defaults to 30
    return_before_sum: bool
        used for analysis, returns original prediction object
    Returns
    -------
    stack_sum
        array with predictions for each image, noise_seed, and neuron
    mean_array
        returns array with each mean neuron prediction for all images entered
    var_array
        returns array with each var. in neuron prediction for all images entered
    regions
        DataFrame with brain region mappings for all neurons 
        
    ex. input predict_loop("constant", 100, image, 3, [[4,6], [5,7]])
    """
    scans = ensure_2d_list(scans)
    noise_type = noise_type.lower()

    models_list, ids_list = [], []

    for session_scan in scans:
        pred_model, ids = scan(session_scan[0], session_scan[1], directory = os.path.join(os.getcwd(), "data","microns")) # look at data/microns/scans.csv for numbers that work
        models_list.append(pred_model)
        ids_list.append(ids)

    def process_image(i: int):
        predict_stack = np.repeat(images[i][np.newaxis, :], num_frames, axis=0)
        return noise_iterations(models_list, ids_list, noise_type, noise_seeds, predict_stack, sigma, scans, stochastic_bin_param, num_frames)
        
    final_array = np.array([process_image(i) for i in range(len(images))])
    if return_before_sum: return final_array # use for Anton analysis

    else:
        final_stack_sum = np.sum(final_array, axis=2)
        final_mean, final_var = np.mean(final_stack_sum, axis=1), np.var(final_stack_sum, axis=1)

        num_neurons = get_neuron_units(scans)
        regions = [get_brain_region(mapping, num_neurons) for mapping in ids_list] # this is why output is in list
        return final_stack_sum, final_mean, final_var, regions
