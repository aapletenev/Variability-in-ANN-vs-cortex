import os 
import numpy as np
from numpy import full
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    noise_results = np.empty((noise_seeds, num_frames, num_neurons)) # check with anton, last dim should always be equal to 2
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

def filter_region(region_input: int, num_images: int, label_mapping, array):
    """
    Parameters
    ----------
    region_input: int
        specified region to output
    num_images: int
        number of images used for predict_loop()
    label_mapping: label
        mapping output from scans()
    array: np.array
        array to filter

    Returns
    -------
    array
        filtered array where brain region is equal to region_input
    """
    
    concat_array = np.concatenate((array, label_mapping), axis=0)
    region_indices = np.where(concat_array[num_images] == region_input)

    return array[..., region_indices[0]]

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

def line_plot_mean(array: np.array, neurons: list | np.ndarray = [i for i in range(100)]): # input mean/var output from predict_loop()
    """
    Parameters
    ----------
    array: np.array
        mean/var output from predict loop
    neurons: list or np.array
        list of specified neurons to pick, defaults to first 100

    Returns
    -------
    final_array
        average output for all images for specified neurons
    """
    final_array = np.empty(shape = len(neurons))
    for i, neuron in enumerate(neurons):
        avg = np.mean(array[:,neuron])
        final_array[i] = avg
    return final_array

"""
This function assumes three sigmas values of 3,15,30 and then a stochastic binarization which is plotted last.
Thus, both inputted lists must have four different arrays of means/variance predictions from prediction loop.
"""
def spike_plot(dynamic_array: list, constant_array: list, mean_plot: bool, 
               normalized: bool, region: int, sigmas: list = [3, 15, 30, 50]):
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
    
    Returns
    -------
    matplotlib plot
        line plot of sigma versus mean or var spike counts
    """
    fig, axes = plt.subplots(1,2, figsize = (8,6), sharey = True)

    sigmas = np.array([3, 15, 30, 50]) # making stochastic bin. 50 here, what should it be?

    colors = plt.cm.tab20(np.linspace(0, 1, 100))  # 100 colors for 100 neurons

    for i in range(100):
        dynamic_y = ([dynamic_array[0][i], dynamic_array[1][i], dynamic_array[2][i], dynamic_array[3][i]] / dynamic_array[0][i] if normalized
        else [dynamic_array[0][i], dynamic_array[1][i], dynamic_array[2][i], dynamic_array[3][i]])
        constant_y = ([constant_array[0][i], constant_array[1][i], constant_array[2][i], constant_array[3][i]] / constant_array[0][i] if normalized
                      else [constant_array[0][i], constant_array[1][i], constant_array[2][i], constant_array[3][i]])

        axes[0].plot(sigmas, dynamic_y, marker='o', linestyle='-', alpha=0.5, color = colors[i])

        axes[1].plot(sigmas, constant_y, marker='o', linestyle='-', alpha=0.5, color = colors[i])
        
    dynamic_mean_y = np.mean(dynamic_array, axis=1) / np.mean(dynamic_array[0])  
    constant_mean_y = np.mean(constant_array, axis=1) / np.mean(constant_array[0])

    slope_dyn, intercept_dyn, r_value_dyn, p_value_dyn, std_err_dyn = linregress(sigmas, dynamic_mean_y)
    y_pred_dyn = intercept_dyn + slope_dyn * sigmas

    slope_con, intercept_con, r_value_con, p_value_con, std_err_con = linregress(sigmas, constant_mean_y)
    y_pred_con = intercept_con + slope_con * sigmas


    axes[0].plot(sigmas, y_pred_dyn, 'o')
    axes[0].plot(sigmas, y_pred_dyn, 'r-', label='Regression line, Dynamic Noise', color = 'red', linewidth = 2)

    axes[1].plot(sigmas, y_pred_con, 'o')
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
    print(f'\033[4m\033[1mConstant Noise 95% Confidence Interval for Slope: [{slope_con - 1.96 * std_err_con:.3f}, {slope_con + 1.96 * std_err_con:.3f}')
    
    axes[0].axhline(y=1, color='black', linestyle='-', linewidth=1)
    axes[1].axhline(y=1, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Sigma for Input Noise')
    axes[0].set_title(f'Dynamic Noise (100 neurons, Region={region})\nNormalized by Mean Response at Sigma=3\nStochastic Binarization Plotted at Point 50' if normalized 
                    else f'Dynamic Noise (100 neurons, Region={region})\nStochastic Binarization Plotted at Point 50', fontsize = 11)
    axes[1].set_title(f'Constant Noise (100 neurons, Region={region})\nNormalized by Mean Response at Sigma=3\nStochastic Binarization Plotted at Point 50' if normalized
                      else f'Constant Noise (100 neurons, Region={region})\nStochastic Binarization Plotted at Point 50', fontsize = 11)
    axes[0].set_ylabel('Mean Prediction\nSpike Count' if mean_plot else 'Variance in\nPredicted Spike Count')
    axes[1].set_xlabel('Sigma for Input Noise')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()
