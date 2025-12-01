from data_management import get_neuron_units
import numpy as np
from PIL import Image

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
    else: raise ValueError ("\n---Noise input not recognized, please try again---")
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

def noise_iterations(model_list, id_list, noise_type: str, noise_seeds: int, image, sigma: int, scans, stochastic_bin_param: bool, num_frames: int = 30, return_noise=False) -> np.array:
    if isinstance(scans, list): num_neurons = get_neuron_units(scans)
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
    return_noise: bool
        return noise instances, defaults to height and width equal to 144 and 256

    Returns
    -------
    array
        returns array with predictions for all neurons
    
    ex. input predict_loop("constant", 100, image, 3, [[4,6], [5,7]])
    """

    if return_noise: return_noise_list = np.empty((noise_seeds, num_frames, 144, 256))
    def process_noise_seed(noise_type: str, image, return_noise) -> np.array:
        """
        Parameters
        ----------
        
        Returns
        -------
        array
            concatenated object of all predictions for every scan and noise seed
        """
        if stochastic_bin_param == True: 
            # constant stochastic binarization
            if noise_type == 'constant': new_image = np.clip(np.round(stochastic_binarization(image)), a_min = 0, a_max = 255).astype('uint8')
            # dynamic stochastic binarization
            elif noise_type == 'dynamic': new_image = np.clip(np.round(np.array([stochastic_binarization(frame) for frame in image])),
                                                              a_min = 0, a_max = 255).astype('uint8')
                                                 
            else:
                print('Please specify the correct type of noise for stochastic binarization, either constant or dynamic')
                return
            if return_noise: 
                return new_image
            for model, ids in zip(model_list, id_list):
                prediction = model.predict(new_image)
            return prediction

        elif stochastic_bin_param == False:  # case: no stochastic bin. 
            
            new_noise = generate_noise(noise_type, num_frames, sigma)
            new_image = np.clip(np.round((image + new_noise)),  a_min = 0, a_max = 255).astype('uint8') # no need to clip values with this dtype

            if return_noise: 
                return new_image
            for model, ids in zip(model_list, id_list):
                prediction = model.predict(new_image)
            return prediction
        else: raise ValueError ('---Enter "true" or "false" for stchastic_bin_param.---')
    
    for i in range(noise_seeds):
        result = process_noise_seed(noise_type, image, return_noise=return_noise)
        if return_noise: 
            return_noise_list[i]=result
        else:
            noise_results[i] = result
    # store result as array in directory
    # delete previous result from memory, re run
    return return_noise_list if return_noise else noise_results
