import os 
import numpy as np
from numpy import argpartition
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import sys
from fnn.microns.__init__ import scan
from scipy.stats import shapiro, linregress
from PIL import Image

"""
relevant functions for iterating through predictions
"""
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

def generate_noise(noise_type: str, num_frames: int, sigma: int, mean = 0) -> np.array: # mean always equal to 0
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

def get_neuron_units(scans) -> int:
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

def stochastic_binarization(image_object: np.array) -> np.array:
    """
    Parameters
    ----------
    image_object: np.array
        either a single image transformed into an array, or stack of different images transformed into arrays
    
    Returns
    -------
    np.array
        array with stochasic binarization applied to input
    """
    image_prob = image_object / 256
    prob_results = np.random.binomial(1, image_prob)
    image = (prob_results * 255).astype('uint8')
    return image

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
    brain_regions = pd.read_csv(brain_region_path) # make input

    ids_matched = pd.merge(ids, brain_regions, how = 'left', on = ['session', 'scan_idx', 'unit_id'])['brain_area']
    ids_matched = ids_matched.map(encoding)

    # output is a list with one pd.Series in it, keep for now
    return ids_matched

def noise_iterations(model_list, id_list, noise_type: str, noise_seeds: int, image, sigma: int, scans, stochastic_bin_param: bool, num_frames: int = 30) -> np.array:
    num_neurons = get_neuron_units(scans)
    noise_results = np.empty((noise_seeds, num_frames, num_neurons))
    """
    Parameters
    ----------
    model_list: list
        list with predictive model weights from scans, all models are visual models
    id_list: list
        list with neuron ids from scan()
    noise_type: string
        dynamic, constant, or no noise
    noise_seeds: int
        how many times we add noise to the prediction
    image: object
        image that we are adding noise to and then predicting on, ie like the frames object below
    sigma: int
        standard deviation for gaussian noise distribution
    scans: list/array
        should be 2d array with pairs of sessions and scan ids taken from scans.csv
    num_neurons: int
        total number of neurons from all scans, inputted from predict_loop()

    Returns
    -------
    array
        returns array with predictions for all neurons
    
    ex. input predict_loop("constant", 100, image, 3, [[4,6], [5,7]])
    """

    def process_noise_seed(noise_type: str, image) -> np.array:
        """
        Parameters
        ----------
        noise_seeds: int
            noise seeds, same as entered into inner_predict_loop() to iterate for each noise seed
        
        Returns
        -------
        array
            concatenated object of all predictions for every scan and noise seed
        """
        if stochastic_bin_param: 
            # constant stochastic binarization
            if noise_type == 'constant': new_image = stochastic_binarization(image)
            # dynamic stochastic binarization
            elif noise_type == 'dynamic': new_image = np.array([stochastic_binarization(frame) for frame in image])       
            else:
                print('Please specify the correct type of noise for stochastic binarization, either constant or dynamic')
                return

            for model, ids in zip(model_list, id_list):
                prediction = model.predict(new_image)
            return prediction

        else: noise_type_process = noise_type # case: no stochastic bin. 
        
        new_noise = generate_noise(noise_type_process, num_frames, sigma)
        new_image = (image + new_noise).astype('uint8')
        
        for model, ids in zip(model_list, id_list):
            prediction = model.predict(new_image)
        
        return prediction
    
    """
    with ThreadPoolExecutor(max_workers=None) as executor:
        for i, result in enumerate(executor.map(
            lambda i: process_noise_seed(noise_type, image),
            range(noise_seeds)
        )):
            noise_results[i] = result
"""
    for i in range(noise_seeds):
        result = process_noise_seed(noise_type, image)
        noise_results[i] = result

    return noise_results

def predict_loop(noise_type: str, images: np.ndarray, sigma: int, scans, stochastic_bin_param = False, noise_seeds: int = 100, num_frames: int = 15):
    """
    Parameters
    ----------
    noise_type: string
        dynamic, constant, stochastic binarization, or no noise
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

    final_stack_sum = np.sum(final_array, axis=2)
    final_mean, final_var = np.mean(final_stack_sum, axis=1), np.var(final_stack_sum, axis=1)

    num_neurons = get_neuron_units(scans)
    regions = [get_brain_region(mapping, num_neurons) for mapping in ids_list] # this is why output is in list
    return final_stack_sum, final_mean, final_var, regions

"""
helper functions
"""
def remove_outliers_topk(arr: list | np.ndarray, removal: str, k_percent: float):
    """   
    Parameters
    ----------
    arr : list or np.ndarray
        Array or list to filter from
    removal : str
        Section of dataset to remove: 'top', 'bottom', or 'both'
        - 'top' removes the top k_percent of the dataset
        - 'bottom' removes the bottom k_percent of the dataset
        - 'both' removes k_percent from both top and bottom, returning the middle
    k_percent : float
        Percentage (0 to 100) of elements to remove from each end
    
    Returns
    -------
    np.ndarray
        Array with outliers removed, preserving original order
    """
    if isinstance(arr, np.ndarray):
        arr = arr.flatten()
    else:
        arr = np.array(arr)

    k = round(k_percent / 100 * len(arr))
    k = max(1, min(k, len(arr) // 2)) 

    if removal == 'top':
        partition_indices = np.argpartition(arr, kth=k)
        indices = partition_indices[:-k]  
    elif removal == 'bottom':
        partition_indices = np.argpartition(arr, kth=len(arr) - k - 1)
        indices = partition_indices[k:] 
    elif removal == 'both':
        top_k_indices = np.argpartition(arr, -k)[-k:]
        bottom_k_indices = np.argpartition(arr, k)[:k]
        exclude_indices = np.concatenate([top_k_indices, bottom_k_indices])
        indices = np.setdiff1d(np.arange(len(arr)), exclude_indices)
    else:
        raise ValueError("removal must be 'top', 'bottom', or 'both'")

    indices = np.unique(indices)
    return arr[indices]



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

def random_images(num, train = True, path = os.path.join("//imagenet-mini")):
    """
    NOTE: directory structure for subfolers as follows
    --imagenet-mini
        -->train
            -->folder1
                -->picture1-1
                -->picture2-1.....
            -->folder2...
            -->folder3...
                -->picture1-3
                -->pictuer2-3
                -->picture....
        -->validation


    Parameters
    ----------
    num: int
        number of images 
    frames: int
        number of frames
    train: bool
        using training or validation set, defaults to train
    path: string
        defaults to cwd for training folder, otherwise path to train/val. folders needed

    Returns
    -------
    folder_id
        folder # used
    image_ids
        image #'s used
    np.stack(final_array)
        stack of num DIFFERENT images in one object, shape (num, 144, 256)
    """

    if train == True: train_val = '//train' 
    else: train_val = '//val'

    folder_path = path + train_val # get path to train/val folders
    initial_folders = os.listdir(folder_path)

    folder_id = np.random.randint(0, len(initial_folders) - 1) # get path to random folder within train/val
    image_folder_path = folder_path + '//' + initial_folders[folder_id]

    image_folder = os.listdir(image_folder_path) # path to folder in train/val with images

    image_ids = np.random.randint(0,len(image_folder) - 1, size = num)
    final_array = np.empty(shape = (num, 144, 256))

    for i in range(num):
        image_name = '//' + image_folder[i]
        image_path = image_folder_path + image_name
        image = Image.open(image_path)
        image = image.convert('L')
        if image.size != (256, 144): image = image.resize((256, 144))
        image = np.array(image)
        final_array[i] = image

    return folder_id, image_ids, np.stack(final_array, axis = 0)

"""
make mask generic for different brain regions 
use for loop with list of regions
"""
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
    mask=label_mapping==region_input
    return array[..., np.squeeze(mask)]

"""
visualization functions
"""

def plot_select30_hist(array, title, neurons, color = 'b'): # plots histogram for first image in stack object, specified neurons

    neuron_pvalue = pd.DataFrame({
    'neuron' : [],
    'p_value' :[]
    })

    fig, axes = plt.subplots(5,6, figsize = (20,6))

    array = array[0] # 3d array is going to be inputted, for now just take first image

    p_values = np.empty((neurons.size))
    test_statistics = np.empty((neurons.size))
    for id, neuron in enumerate(neurons):
        row = id // 6
        col = id % 6
        ax = axes[row, col]
        plot = array[:,neuron]
        ax.hist(plot, color = color)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{round(x,3)}"))

        # hypothesis test
        p_value, test_statistic = shapiro(plot)
        p_values[id] = p_value
        test_statistics[id] = test_statistic
        neuron_pvalue.loc[id] = neuron, p_value

    # display histogram
    fig.suptitle(f"Neuron Outputs: {title}")
    fig.supxlabel('Predicted Firing Rate')
    fig.supylabel('Counts')
    plt.tight_layout()
    plt.show()

    # print dataframe with p values
    neuron_pvalue['neuron'] = neuron_pvalue['neuron'].astype(int)
    neuron_pvalue['p_value'] = neuron_pvalue['p_value'].round(3)
    print(neuron_pvalue)

def line_plot_mean(array: np.array, neurons: list | np.ndarray = [i for i in range(100)],
                   specified_image_index: int = None): # input mean/var output from predict_loop()
    """
    Parameters
    ----------
    array: np.array
        mean/var output from predict loop
    neurons: list or np.array
        list of specified neurons to pick, defaults to first 100
    specified_image_index: int
        if inputted will format line plot data for specified image

    Returns
    -------
    final_array
        average output for all images for specified neurons
    """
    final_array = np.empty(shape = len(neurons))
    for i, neuron in enumerate(neurons):
        if specified_image_index is not None: final_array[i] =  array[specified_image_index, neuron] # if input is just one image
        else: final_array[i] = np.mean(array[:, neuron]) # if input is for all images
    return final_array

"""
This function assumes three sigmas values of 3,15, and 30.
Thus, both inputted lists must have three different arrays of means/variance predictions from prediction loop.
"""
def spike_plot(dynamic_array: list, constant_array: list, mean_plot: bool, # PLOT FOR GRAY IMAGE ONLY
               normalized: bool, region: int, sigmas: list = [3, 15, 30], num_neurons: int = 100):
    """
    Parameters
    ----------
    dynamic_array: list
        list of dynamic noise mean/var arrays from prediction loop
    constant_array: list
        list of constant noise mean/var arrays from prediction loop
    mean_plot: bool
        if plot is for mean values or not (variance plot)
    normalized: bool
        to normalize values or not
    region: int
        input region from brain for title
    sigmas: list
        defaults to list of sigma values for noise used for initial runs
    num_neurons: int
        number of neurons to plot, defaults to 100
    
    Returns
    -------
    matplotlib plot
        line plot of sigma versus mean or var spike counts
    """
    fig, axes = plt.subplots(1,2, figsize = (8,6), sharey = True)

    sigmas = np.array([3, 15, 30])

    colors = plt.cm.tab20(np.linspace(0, 1, 100))  # 100 colors for 100 neurons

    dynamic_means_y = np.empty((num_neurons, len(sigmas)))
    constant_means_y = np.empty((num_neurons, len(sigmas)))

    for i in range(num_neurons):
        dynamic_y = np.array([dynamic_array[0][i], dynamic_array[1][i], dynamic_array[2][i]]) / dynamic_array[0][i] if normalized else np.array(
            [dynamic_array[0][i], dynamic_array[1][i], dynamic_array[2][i]])
        constant_y = np.array([constant_array[0][i], constant_array[1][i], constant_array[2][i]]) / constant_array[0][i] if normalized else np.array(
            [constant_array[0][i], constant_array[1][i], constant_array[2][i]])

        axes[0].plot(sigmas, dynamic_y, marker='o', linestyle='-', alpha=0.5, color = colors[i])

        axes[1].plot(sigmas, constant_y, marker='o', linestyle='-', alpha=0.5, color = colors[i])
        
        dynamic_means_y[i] = dynamic_y
        constant_means_y[i] = constant_y

    dynamic_mean_plot = np.mean(dynamic_means_y, axis = 0) / np.mean(dynamic_means_y[0], axis = 0) if normalized else np.mean(dynamic_means_y, axis = 0)
    constant_mean_plot = np.mean(dynamic_means_y, axis=0) / np.mean(constant_means_y[0], axis=0) if normalized else np.mean(constant_means_y, axis = 0)

    slope_dyn, intercept_dyn, r_value_dyn, p_value_dyn, std_err_dyn = linregress(sigmas, dynamic_mean_plot)
    y_pred_dyn = intercept_dyn + slope_dyn * sigmas
    slope_con, intercept_con, r_value_con, p_value_con, std_err_con = linregress(sigmas, constant_mean_plot)
    y_pred_con = intercept_con + slope_con * sigmas


    axes[0].plot(sigmas, y_pred_dyn, 'r-', label='Regression line, Dynamic Noise', color = 'red', linewidth = 2)

    axes[1].plot(sigmas, y_pred_con, 'r-', label='Regression line, Constant Noise', color = 'red', linewidth = 2)

    # error band/SE
    dyn_se = std_err_dyn * np.sqrt(1/len(sigmas) + (sigmas - np.mean(sigmas))**2 / np.sum((sigmas - np.mean(sigmas))**2))
    con_se = std_err_con * np.sqrt(1/len(sigmas) + (sigmas - np.mean(sigmas))**2 / np.sum((sigmas - np.mean(sigmas))**2))
    axes[0].fill_between(sigmas, y_pred_dyn - 1.96*dyn_se, y_pred_dyn + 1.96*dyn_se, color='red',  alpha = 0.3)
    axes[1].fill_between(sigmas, y_pred_con - 1.96*con_se, y_pred_con + 1.96*con_se, color='red', alpha = 0.3)

    #axes[0].text(1, -2, f'95% CI for Slope: [{slope_dyn - 1.96 * std_err_dyn:.3f}, {slope_dyn + 1.96 * std_err_dyn:.3f}]', fontsize = 7) # x1.96 for 95% CI
    #axes[1].text(1, -2, f'95% CI for Slope = [{slope_con - 1.96 * std_err_con:.3f}, {slope_con + 1.96 * std_err_con:.3f}]', fontsize = 7)
    # plotting text is difficult for different values, for now print
    print(f'\033[1m\033[4mDynamic Noise 95% Confidence Interval for Slope: [{slope_dyn - 1.96 * std_err_dyn:.3f}, {slope_dyn + 1.96 * std_err_dyn:.3f}]')
    print(f'\033[4m\033[1mConstant Noise 95% Confidence Interval for Slope: [{slope_con - 1.96 * std_err_con:.3f}, {slope_con + 1.96 * std_err_con:.3f}]')
    
    if not normalized: 
        axes[0].text(2, axes[0].get_ylim()[1]-4.5, f'Slope: {slope_dyn:.3f}')
        axes[1].text(2, axes[1].get_ylim()[1]-4.5, f'Slope: {slope_con:.3f}')

    axes[0].axhline(y=1, color='black', linestyle='-', linewidth=1)
    axes[1].axhline(y=1, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Sigma for Input Noise')
    axes[0].set_title(f'Dynamic Noise ({num_neurons} neurons, Region={region})\nNormalized by Response at Sigma=3' if normalized 
                    else f'Dynamic Noise ({num_neurons} neurons, Region={region})', fontsize = 11)
    axes[1].set_title(f'Constant Noise (100 neurons, Region={region})\nNormalized by Response at Sigma=3' if normalized
                      else f'Constant Noise (100 neurons, Region={region})', fontsize = 11)
    axes[0].set_ylabel('Mean Prediction\nSpike Count' if mean_plot else 'Variance in\nPredicted Spike Count')
    axes[1].set_xlabel('Sigma for Input Noise')
    axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def five_turning_curves(arrays: list | np.ndarray, titles: list, ten_neurons: list = [i for i in range(10)]):
    """
    Parameters
    ----------
    arrays: list
        list of output arrys to plot (sum arrays from predict loop)
    titles: list
        list of corresponding titles for each plot
    ten_neurons: list
        list of ten neurons to use, defaults to first ten 
    """
    fig, axes = plt.subplots(5, 1, figsize=(4, 20), sharex=True)

    image_ids = np.array([i for i in range(5)], dtype=int)

    for ax, array, title in zip(axes, arrays, titles):
        mean_array = np.mean(array, axis=1) 
        variance = np.var(array, axis=1)
        std_dev = np.sqrt(variance)

        for neuron in ten_neurons:
            neuron_predictions = mean_array[:, neuron]
            ax.errorbar(image_ids, neuron_predictions, yerr=std_dev[:, neuron],
                        marker='o', linestyle='-', capsize=5)

        ax.set_title(f"{title}", fontsize=10)
        ax.set_ylabel("Mean Predicted Spike Count")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True)
    plt.xlabel('Image ID')
    plt.tight_layout()
    plt.show()

def spike_plot_4x2(dynamic_array: list, constant_array: list, mean_plot: bool, 
                   normalized: bool, sigmas: list = [3, 15, 30], num_neurons: int = 100, num_regions: int = 4):
    """
    Parameters
    ----------
    dynamic_array: list
        List of lists where each sublist contains arrays of mean/var spike counts 
        for different sigmas across neurons for a region
    constant_array: list
        List of lists where each sublist contains arrays of mean/var spike counts 
        for different sigmas across neurons for a region
    mean_plot: bool
        If True, plot mean spike counts; if False, plot variance
    normalized: bool
        If True, normalize values by the response at the first sigma
    sigmas: list
        List of sigma values for noise, defaults to [3, 15, 30]
    num_neurons: int
        Number of neurons to plot, defaults to 100
    num_regions: int
        Number of brain regions from data, defaults to 4
    
    Returns
    -------
    matplotlib plot
        4x2 grid of line plots, each row for a region, with dynamic noise in column 0 
        and constant noise in column 1
    """
    fig, axes = plt.subplots(4, 2, figsize=(8, 12), sharey=True, sharex=True)
    sigmas = np.array(sigmas)
    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))

    for region in range(num_regions):
        ax_dyn = axes[region, 0]
        ax_con = axes[region, 1]
        dynamic_data = dynamic_array[region]  # List of arrays for sigmas in this region
        constant_data = constant_array[region]

        # Dynamic noise plot
        dynamic_means_y = np.empty((num_neurons, len(sigmas)))
        for i in range(num_neurons):
            dynamic_y = np.array([dynamic_data[s_idx][i] for s_idx in range(len(sigmas))])
            if normalized and dynamic_data[0][i] != 0:
                dynamic_y = dynamic_y / dynamic_data[0][i]
            dynamic_means_y[i] = dynamic_y
            ax_dyn.plot(sigmas, dynamic_y, marker='o', linestyle='-', alpha=0.5, color=colors[i])

        dynamic_mean_plot = np.mean(dynamic_means_y, axis=0)
        slope_dyn, intercept_dyn, _, _, std_err_dyn = linregress(sigmas, dynamic_mean_plot)
        y_pred_dyn = intercept_dyn + slope_dyn * sigmas
        ax_dyn.plot(sigmas, y_pred_dyn, 'r-', label='Regression line', linewidth=2)
        dyn_se = std_err_dyn * np.sqrt(1/len(sigmas) + (sigmas - np.mean(sigmas))**2 / np.sum((sigmas - np.mean(sigmas))**2))
        ax_dyn.fill_between(sigmas, y_pred_dyn - 1.96*dyn_se, y_pred_dyn + 1.96*dyn_se, color='red', alpha=0.3)
        print(f"Region {region+1}, Dynamic Noise 95% CI for Slope: [{slope_dyn - 1.96 * std_err_dyn:.3f}, {slope_dyn + 1.96 * std_err_dyn:.3f}]")
        if not normalized:
            ax_dyn.text(0.05, 0.8, f'Slope: {slope_dyn:.3f}', transform=ax_dyn.transAxes, ha='left', va='top')
        if normalized:
            ax_dyn.axhline(y=1, color='black', linestyle='-', linewidth=1)
        ax_dyn.set_title(f'Dynamic Noise, Region {region}' + 
                         (f'\nNormalized by Response at Sigma={sigmas[0]}' if normalized else ''), 
                         fontsize=11) if region == 0 else ''

        # Constant noise plot
        constant_means_y = np.empty((num_neurons, len(sigmas)))
        for i in range(num_neurons):
            constant_y = np.array([constant_data[s_idx][i] for s_idx in range(len(sigmas))])
            if normalized and constant_data[0][i] != 0:
                constant_y = constant_y / constant_data[0][i]
            constant_means_y[i] = constant_y
            ax_con.plot(sigmas, constant_y, marker='o', linestyle='-', alpha=0.5, color=colors[i])

        constant_mean_plot = np.mean(constant_means_y, axis=0)
        slope_con, intercept_con, _, _, std_err_con = linregress(sigmas, constant_mean_plot)
        y_pred_con = intercept_con + slope_con * sigmas
        ax_con.plot(sigmas, y_pred_con, 'r-', label='Regression line', linewidth=2)
        con_se = std_err_con * np.sqrt(1/len(sigmas) + (sigmas - np.mean(sigmas))**2 / np.sum((sigmas - np.mean(sigmas))**2))
        ax_con.fill_between(sigmas, y_pred_con - 1.96*con_se, y_pred_con + 1.96*con_se, color='red', alpha=0.3)
        print(f"Region {region+1}, Constant Noise 95% CI for Slope: [{slope_con - 1.96 * std_err_con:.3f}, {slope_con + 1.96 * std_err_con:.3f}]")
        if not normalized:
            ax_con.text(0.05, 0.8, f'Slope: {slope_con:.3f}', transform=ax_con.transAxes, ha='left', va='top')
        if normalized:
            ax_con.axhline(y=1, color='black', linestyle='-', linewidth=1)
        ax_con.set_title(f'Constant Noise' + 
                         (f'\nNormalized by Response at Sigma={sigmas[0]}' if normalized else ''), 
                         fontsize=11) if region == 0 else ''

        # Y-axis label for each region (left column only)
        ax_dyn.set_ylabel(f'Mean Prediction\nSpike Count\n(Region {region+1})' if mean_plot else f'Variance in\nPredicted Spike Count\n(Region {region+1})')

    # X-axis labels for bottom row only
    axes[3, 0].set_xlabel('Sigma for Input Noise')
    axes[3, 1].set_xlabel('Sigma for Input Noise')

    # Add legend to each subplot
    for ax in axes.flat:
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

