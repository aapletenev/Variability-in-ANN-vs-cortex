import os 
import numpy as np
from numpy import concatenate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from fnn.microns.__init__ import scan
from scipy.stats import shapiro
from concurrent.futures import ThreadPoolExecutor

"""
relevant functions for iterating through predictions, visualizations as well
"""
def visual_prediction(session: int, scan_idx: int, stimuli_noise) -> np.array:
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
    return results # should be array regardless   

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

def inner_predict_loop(noise_type: str, noise_seeds: int, image, sigma: int, scans, num_neurons: int) -> np.array:
    """
    Parameters
    ----------
    noise_type: string
        dynamic, constant, or no noise NOTE: make stochastic binarization a parameter here
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
    num_frames = image.shape[0]
    noise_results = np.empty((noise_seeds, num_frames, num_neurons))

    def process_noise_seed(noise_seeds: int):
        """
        Parameters
        ----------
        noise_seeds: int
            noise seeds, same as entered into inner_predict_loop() to iterate for each noise seed in threadpoolexecutor
        
        Returns
        -------
        array
            concatenated object of all predictions for every scan and noise seed
        """
        new_noise = generate_noise(noise_type, num_frames, sigma)

        new_image = (image + new_noise).astype('uint8')
        
        prediction1_array = [visual_prediction(pair[0], pair[1], new_image) for pair in scans]
        # ralf's comment about pixel_value/256 being chance of pixel value being changed to 255, add code here? should be per noise seed
        return np.concatenate(prediction1_array, axis = 1)
            
    with ThreadPoolExecutor() as executor:
        noise_results = np.array(list(executor.map(process_noise_seed, range(noise_seeds))))
    return noise_results


def predict_loop(noise_type: str, noise_seeds: int, images: np.ndarray, sigma: int, scans, num_frames: int = 30):
    """
    Parameters
    ----------
    noise_type: string
        dynamic, constant, or no noise NOTE: make stochastic binarization a parameter here
    noise_seeds: int
        how many times we add noise to the prediction
    images: object
        images (usually object from np.stack()) that we are adding noise to and then predicting on
    sigma: int
        standard deviation for gaussian noise distribution
    scans: list/array
        should be 2d array with pairs of sessions and scan ids taken from scans.csv
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
        
    ex. input predict_loop("constant", 100, image, 3, [[4,6], [5,7]])
    """
    scans = ensure_2d_list(scans)
    num_neurons = get_neuron_units(scans) 

    def process_image(i: int):
        predict_stack = np.repeat(images[i][np.newaxis, :], num_frames, axis=0)
        return inner_predict_loop(noise_type, noise_seeds, predict_stack, sigma, scans, num_neurons)
        
    """
    for i, element in enumerate(images):
        predict_stack = np.stack([images[i]] * num_frames, axis = 0)

        prediction1 = inner_predict_loop(noise_type, noise_seeds, predict_stack, sigma, scans, num_neurons)

        try:
            if (prediction_stack.shape != prediction1.shape): 
                prediction1 = np.reshape(prediction1, (1, prediction1.shape[0], prediction1.shape[1], prediction1.shape[2]))
                prediction_stack = np.stack((prediction1, prediction_stack), axis = 0)
        except NameError:
            prediction_stack = prediction1
        
        final_array[i] = prediction_stack
        del prediction_stack
    """

    """
    for i in range(len(images)):
        final_array[i] = process_image(i)
    """
    final_array = [process_image(i) for i in range(len(images))]

    final_stack_sum = np.sum(final_array, axis=2)
    return final_stack_sum, np.mean(final_stack_sum, axis=1), np.var(final_stack_sum, axis=1)
            
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




