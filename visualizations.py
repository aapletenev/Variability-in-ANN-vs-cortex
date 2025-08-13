import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from scipy.stats import shapiro, linregress
from data_management import remove_x

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

ALL INPUT MUST BE 2D LIST OF ARRAYS (for plots from presentation): input arrays must be inputted as follows to plot 16 constant noise tuning curves.
------------------------------------
sixteen_turning_curves(arrays = [[c3_sum_r1, c15_sum_r1, c30_sum_r1, cbin_sum_r1],
            [c3_sum_r2, c15_sum_r2, c30_sum_r2, cbin_sum_r2],
            [c3_sum_r3, c15_sum_r3, c30_sum_r3, cbin_sum_r3],
            [c3_sum_r4, c15_sum_r4, c30_sum_r4, cbin_sum_r4]]...) 
"""


#code to efficiently load predicted arrays if they are in the same directory as .npy files
"""
# in real code we now d3_labels from np.unique() on region label output from predictions
d3_labels = [1,2,3,4]

string_path = 'fnn//input_noise//saved_inputs_outputs//predictions_july21//'

for noise in ['c3', 'c15', 'c30', 'd3', 'd15', 'd30', 'dbin', 'cbin', 'no_noise']:
    for type_noise in ['_sum', '_mean', '_var', '_labels']:
        file_name = string_path + noise + type_noise + '.npy'
        var_name = noise + type_noise
        globals()[var_name] = np.load(file_name)

# load predictions by region
for region in np.unique(d3_labels):
    for noise in ['c3', 'c15', 'c30', 'd3', 'd15', 'd30', 'dbin', 'cbin', 'no_noise']:
        for type_noise in ['_sum', '_mean', '_var']:
            array_name = string_path + noise + type_noise + '_r' + str(region) + '.npy'
            globals()[array_name] = np.load(array_name)

"""



def spike_plot_4x2(main_title: str, dynamic_array: list, constant_array: list, mean_plot: bool, # first plot in presentation results
                   normalized: bool, sigmas: list = [3, 15, 30], num_neurons: int = 100, num_regions: int = 4, savefig: bool = False):
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
    fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharey=True, sharex=True)
    fig.text(0.5, 1.02, main_title, ha='center', va='top', fontsize=14, fontweight = 'bold', bbox=dict(facecolor='white'))
    sigmas = np.array(sigmas)
    colors = plt.cm.tab20(np.linspace(0, 1, num_neurons))

    for region in range(num_regions):
        ax_dyn = axes[0, region]
        ax_con = axes[1, region]
        dynamic_data = dynamic_array[region]  # List of arrays for sigmas in this region
        constant_data = constant_array[region]

        if region == 0: title_region = 'V1' 
        elif region == 1: title_region = 'LM'
        elif region == 2: title_region = 'AL'
        else: title_region = 'RL'

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
        ax_dyn.plot(sigmas, y_pred_dyn, 'r--', label='Regression line', linewidth=1)
        dyn_se = std_err_dyn * np.sqrt(1/len(sigmas) + (sigmas - np.mean(sigmas))**2 / np.sum((sigmas - np.mean(sigmas))**2))
        ax_dyn.fill_between(sigmas, y_pred_dyn - 1.96*dyn_se, y_pred_dyn + 1.96*dyn_se, color='blue', alpha=1)
        #print(f"Region: {title_region}, Dynamic Noise 95% CI for Slope: [{slope_dyn - 1.96 * std_err_dyn:.3f}, {slope_dyn + 1.96 * std_err_dyn:.3f}]")
        
        ci_dyn = f'[{slope_dyn - 1.96 * std_err_dyn:.3f}, {slope_dyn + 1.96 * std_err_dyn:.3f}]'
        ax_dyn.text(0.05, 0.85, f'Slope: {slope_dyn:.3f}\nCI: {ci_dyn}', transform=ax_dyn.transAxes, ha='left', va='top')
        if normalized: ax_dyn.axhline(y=1, color='black', linestyle='-', linewidth=1)
        
        ax_dyn.set_ylabel(f'Dynamic Noise' + 
                         (f'\n(Normalized by Response at Sigma={sigmas[0]})' if normalized else '') +
                          f'\n\nMean Prediction\nSpike Count', 
                         fontsize=12) if region == 0 else ''
        
        # Constant noise plot
        constant_means_y = np.empty((num_neurons, len(sigmas)))
        for i in range(num_neurons):
            constant_y = np.array([constant_data[s_idx][i] for s_idx in range(len(sigmas))])
            if normalized and constant_data[0][i] != 0:
                constant_y = constant_y / constant_data[0][i]
            constant_means_y[i] = constant_y
            ax_con.plot(sigmas, constant_y, marker='o', linestyle='-', alpha=0.5, color=colors[i])

        constant_mean_plot = np.mean(constant_means_y, axis=0)
        linregress_con = linregress(sigmas, constant_mean_plot) # changing to linregress object here
        y_pred_con = linregress_con.intercept + linregress_con.slope * sigmas
        ax_con.plot(sigmas, y_pred_con, 'r--', label='Regression line', linewidth=1)
        con_se = linregress_con.stderr * np.sqrt(1/len(sigmas) + (sigmas - np.mean(sigmas))**2 / np.sum((sigmas - np.mean(sigmas))**2))
        #print(f'lower bound: {y_pred_con - 1.96*con_se}; upper bound: { y_pred_con + 1.96*con_se}')
        ax_con.fill_between(sigmas, y_pred_con - 1.96*con_se, y_pred_con + 1.96*con_se, color='blue', alpha=1)
        #print(f'Region: {title_region}, Constant Noise 95% CI for Slope: [{linregress_con.slope - 1.96 * linregress_con.stderr:.3f}, '
        #      f'{linregress_con.slope + 1.96 * linregress_con.stderr:.3f}]')
        
        ci_con = f'[{linregress_con.slope - 1.96 * linregress_con.stderr:.3f}, {linregress_con.slope + 1.96 * linregress_con.stderr:.3f}]'
        ax_con.text(0.05, 0.85, f'Slope: {linregress_con.slope:.3f}\nCI: {ci_con}', transform=ax_con.transAxes, ha='left', va='top')
        if normalized: ax_con.axhline(y=1, color='black', linestyle='-', linewidth=1)

        ax_con.set_ylabel(f'Constant Noise' + 
                         (f'\n(Normalized by Response at Sigma={sigmas[0]})' if normalized else '') +
                          f'\n\nMean Prediction\nSpike Count', 
                         fontsize=12) if region == 0 else ''

        #ax_dyn.set_ylabel(f'Mean Prediction\nSpike Count\n(Region: {title_region})' if mean_plot else f'Variance in\nPredicted Spike Count\n(Region: {title_region})')


    for col in range(4): axes[1, col].set_xlabel('Sigma for Input Noise')

    # Add legend to each subplot
    for ax in axes.flat:
        ax.legend(loc='upper left')

    plt.tight_layout()
    if savefig: 
        try:
            date = datetime.now()
            file_date = f'({str(date.month)}-{str(date.day)}-{str(date.year)})'
            path_name = f'{main_title}_{file_date}.pdf'
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name), 
                        bbox_inches = 'tight', pad_inches = 0.3)
        except FileNotFoundError:
            os.makedirs(f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', exist_ok = True)
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name),
                        bbox_inches = 'tight', pad_inches = 0.3)
            plt.close()
    else: plt.show()

def sixteen_tuning_curves(arrays: list | np.ndarray, main_title: str, titles: list, ten_neurons: list = [i for i in range(10)], savefig: bool = False): # second plot in presentation results
    """
    Parameters
    ----------
    arrays: list
        list of output arrys to plot (sum arrays from predict loop)
    main_title: str
        main title at top of plot
    titles: list
        list of corresponding titles for each plot
    ten_neurons: list
        list of ten neurons to use, defaults to first ten 
    savefig: bool
        whether or not to save the figure
    
    Returns
        4x4 tuning curves figure

    """
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
    image_ids = np.array([i for i in range(5)], dtype=int)
    fig.text(0.5, 1.02, main_title, ha='center', va='top', fontsize=14, fontweight = 'bold', bbox=dict(facecolor='white'))
    fig.text(0.5, -.02, 'Image ID', ha='center', va='bottom', fontsize=12, fontweight = 'bold')

    for i in range(len(arrays)):
        for j in range(len(arrays[0])):
            ax = axes[i,j]
            
            mean_array = np.mean(arrays[i][j], axis=1) 
            variance = np.var(arrays[i][j], axis=1)
            std_dev = np.sqrt(variance)

            for neuron in ten_neurons:
                neuron_predictions = mean_array[:, neuron]
                ax.errorbar(image_ids, neuron_predictions, yerr=std_dev[:, neuron],
                            marker='o', linestyle='-', capsize=5)

            ax.set_title(f"{titles[j]}" if i == 0 else '', fontsize=10)
            ax.set_ylabel("Mean Predicted Spike Count" if j == 0 else '', fontweight = 'bold')
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.grid(True)
    plt.tight_layout()
    if savefig: 
        try:
            date = datetime.now()
            file_date = f'({str(date.month)}-{str(date.day)}-{str(date.year)})'
            path_name = f'{main_title}_{file_date}.pdf'
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name), 
                        bbox_inches = 'tight', pad_inches = 0.3)
        except FileNotFoundError:
            os.makedirs(f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', exist_ok = True)
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name),
                        bbox_inches = 'tight', pad_inches = 0.3)
            plt.close()
    else: plt.show()

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

def mean_var_scatter_4x4(means, vars, main_title: str, savefig: bool = False, remove_x_str: str = None, x_percent: int = None): # third plot in presentation results

    fig, axes = plt.subplots(4, 4, figsize=(10, 8), sharex=False, sharey=False)
    fig.text(0.5, 1.02, main_title, ha='center', va='top', fontsize=14, fontweight = 'bold', bbox=dict(facecolor='white'))
    sigmas = ['Sigma=3', 'Sigma=15', 'Sigma=30', 'Stochastic Binarization']
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            ax = axes[i, j] # first iterate j through different sigmas
            
            x = means[i][j]
            y = vars[i][j]

            x = x.flatten()
            y = y.flatten()

            linregress_result = linregress(x, y)
            x_fit = np.linspace(np.min(x), np.max(x), len(x))
            y_fit = linregress_result.intercept + linregress_result.slope * x_fit

            std_err = linregress_result.stderr * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            if remove_x_str is not None and x_percent is not None:                  
                y, _ = remove_x(y, remove_x_str, x_percent)
                x = x[_]
                ax.set_ylim(0, np.max(y))
                ax.set_xlim(0,  np.max(x))
            cmap = plt.get_cmap('plasma')
            colors = cmap(np.linspace(0, .95, len(x)))         
            
            ax.scatter(x, y, marker='.', color = colors, alpha = 0.7)
            ax.plot(x_fit, y_fit, 'r--', label='Regression line', linewidth=2)           
            
            ax.fill_between(x_fit, y_fit - 1.96 * std_err, y_fit + 1.96 * std_err, color = 'blue', alpha = 1)

            confidence_interval = f'[{linregress_result.slope - 1.96*linregress_result.stderr:.3f}, {linregress_result.slope + 1.96*linregress_result.stderr:.3f}]'
            ax.text(0, 1, f'Slope: {linregress_result.slope:.3f}\nCI: {confidence_interval}\nR-Squared: {linregress_result.rvalue**2:.3f}',
                    fontsize = '6', ha = 'left',  va = 'top', transform=ax.transAxes)

            ax.grid(True)

            if i == 0: title_region = 'V1' 
            elif i == 1: title_region = 'LM'
            elif i == 2: title_region = 'AL'
            else: title_region = 'RL'

            if j == 0: ax.set_ylabel(f'Variance in\nPredicted Spike Count', fontsize = 11)
            
            if i == 0: ax.set_title(f'({sigmas[j]})\n', fontsize = 10)
            elif i == 3: ax.set_xlabel('Mean Predicted\nSpike Count', fontsize = 12)  
    
    plt.tight_layout()
    if savefig: 
        try:
            date = datetime.now()
            file_date = f'({str(date.month)}-{str(date.day)}-{str(date.year)})'
            path_name = f'{main_title}_{file_date}.pdf'
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name), 
                        bbox_inches = 'tight', pad_inches = 0.3)
        except FileNotFoundError:
            os.makedirs(f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', exist_ok = True)
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name),
                        bbox_inches = 'tight', pad_inches = 0.3)
            plt.close()
    else: plt.show()

def mean_var_scatter_4x4_regression(means, vars, neurons, main_title: str, savefig: bool = False): # fourth plot in presentation results
    """
    Parameters
    ----------
    means: list
        list of mean arrays, must be 2d list
    vars: list
        list of varriance arrays, must be 2d list
    neurons: list
        ten neurons to plot
    main_title: str
        title for plot
    savefig: bool
        whether or not to save figure, defaults image to same directory

    Returns
        4x4 mean versus variance lineplot for selected neurons 
    """
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=False, sharey=False)
    fig.text(0.5, 1.02, main_title, ha='center', va='top', fontsize=12, fontweight = 'bold', bbox=dict(facecolor='white'))
    sigmas = ['Sigma=3', 'Sigma=15', 'Sigma=30', 'Stochastic Binarization']
    
    for i in range(4):
        neuron_arr = neurons[i]
        for j in range(4):
            idx = i * 4 + j
            ax = axes[i, j] # first iterate j through different sigmas
            

            x = np.array(means[i][j])
            y = np.array(vars[i][j])
            for neuron in neuron_arr:
                x_neuron = x[:, neuron]
                y_neuron = y[:, neuron]
                
                cmap = plt.get_cmap('plasma')

                color_i = neuron_arr.index(neuron) / (len(neuron_arr) - 1)
                colors = cmap(color_i)
                               
                ax.scatter(x_neuron, y_neuron, marker='.', color = colors, alpha = 0.4)

                linregress_result = linregress(x_neuron, y_neuron)
                x_fit = np.linspace(np.min(x_neuron), np.max(x_neuron), len(x_neuron))
                y_fit = linregress_result.intercept + linregress_result.slope * x_fit

                ax.plot(x_fit, y_fit, color=colors, label='Regression line', linewidth=2)           
                
                ax.grid(True)

                if i == 0: title_region = 'V1' 
                elif i == 1: title_region = 'LM'
                elif i == 2: title_region = 'AL'
                else: title_region = 'RL'

                if j == 0: ax.set_ylabel(f'Variance in\nPredicted Spike Count', fontsize = 11)
                
                if i == 0: ax.set_title(f'({sigmas[j]})', fontsize = 10)
                elif i == 3: ax.set_xlabel('Mean Predicted\nSpike Count', fontsize = 12)  
    
    fig.text(0, -0.01, 'Note: Colors represent same neuron within each region.', 
             ha='left', va='bottom', fontsize=12)


    plt.tight_layout()
    if savefig: 
        try:
            date = datetime.now()
            file_date = f'({str(date.month)}-{str(date.day)}-{str(date.year)})'
            path_name = f'{main_title}_{file_date}.pdf'
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name), 
                        bbox_inches = 'tight', pad_inches = 0.3)
        except FileNotFoundError:
            os.makedirs(f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', exist_ok = True)
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name),
                        bbox_inches = 'tight', pad_inches = 0.3)
            plt.close()
    else: plt.show()

def violin_4x4(means, vars, main_title: str, savefig: bool = False, remove_x_str: str = None, remove_x_percent: int = None): # first violin plot in presentation results
    """
    Parameters
    ----------
    means: list
        list of mean arrays, must be 2d list
    vars: list
        list of varriance arrays, must be 2d list
    main_title: str
        main title for plot
    savefig: bool
        to save figure in current directory, defaults to False
    remove_x_str: str
        to remove top, bottom or None % of data, defaults to None
    Remove_x_percent: int
        to remove x % of data, defaults to None

    Returns
    -------
    plot
        4x4 violin plot for each region and sigma/binarization (region 3 excluded)

    """
    
    fig, axes = plt.subplots(len(means), 1, figsize=(10, 8), sharex=True, sharey=False)
    fig.text(0.5, 1.02, main_title, ha='center', va='top', fontsize=12, fontweight = 'bold', bbox=dict(facecolor='white'))
    violin_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    violin_colors = [violin_colors[i] for i in range(len(means))]
    for i in range(len(means)):
        ax = axes[i] 
        
        region_x = means[i]
        region_y = vars[i]
        slope_list = []

        for mean_arr, var_arr in zip(region_x, region_y):
            slope_temp = []
            for neuron in range(mean_arr.shape[1]):
                neuron_x = mean_arr[:, neuron]
                neuron_y = var_arr[:, neuron]
                linregress_neuron = linregress(neuron_x, neuron_y)
                slope_temp.append(linregress_neuron.slope)
            slope_list.append(slope_temp)
        
        # method 1, remove x
        if remove_x_str is not None and remove_x_percent is not None:
            for region_slopes in slope_list:
                region_slopes = remove_x(region_slopes, remove_x_str, remove_x_percent)

        vp = ax.violinplot(slope_list, showmeans = False, showmedians = True, showextrema = False,
                           quantiles = [[0.25, 0.75] for _ in range(len(slope_list))])


        vp['cquantiles'].set_color('black')  
        vp['cquantiles'].set_linewidth(0.25)

        for body in vp['bodies']:
            body.set_facecolor(violin_colors[i])
        for element in ['cmedians']:
            vp[element].set_color('black')
                
        for element in ['cquantiles', 'cmedians']: vp[element].set_linewidth(0.5)

        if i == 0: title_region = 'V1' 
        elif i == 1: title_region = 'LM'
        elif i == 2: title_region = 'AL'
        else: title_region = 'RL'

        ax.set_ylabel(f'(Region: {title_region})\nRegression Slope', fontsize = 10)
        
        tick_labels = ['15', '30', 'Stochastic\nBinarization']
        if i == 3: ax.set_xticks([i+1 for i, _ in enumerate(tick_labels)])
        if i == 3: ax.set_xticklabels(tick_labels)  
        if i == 3: ax.set_xlabel('Sigma + Binarization')
        
        # method 2, set y_lim (adjusting differently for D and C plots)
        if i == 0 or i == 1: ax.set_ylim( min([np.min(arr) for arr in slope_list]), 0.05 * max([np.max(arr) for arr in slope_list]))
        elif i == 2: ax.set_ylim(min([np.min(arr) for arr in slope_list]), 0.2 * max([np.max(arr) for arr in slope_list]))
        else: ax.set_ylim(.1 * min([np.min(arr) for arr in slope_list]), 0.2 * max([np.max(arr) for arr in slope_list]))
    
    plt.text(0, -.45, 'Note: lines within each plot show the interquartile range and median.')
    plt.tight_layout()
    if savefig: 
        try:
            date = datetime.now()
            file_date = f'({str(date.month)}-{str(date.day)}-{str(date.year)})'
            path_name = f'{main_title}_{file_date}.pdf'
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name), 
                        bbox_inches = 'tight', pad_inches = 0.3)
        except FileNotFoundError:
            os.makedirs(f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', exist_ok = True)
            plt.savefig(os.path.join(os.getcwd(), f'plots//({str(date.month)}-{str(date.day)}-{str(date.year)})', path_name),
                        bbox_inches = 'tight', pad_inches = 0.3)
            plt.close()
    else: plt.show()

def violin_combined(means, vars, main_title: str, savefig: bool = False): # second violin plot in presentation
    """
    Parameters
    ----------
    means: list
        list of mean arrays, must be 2d list
    vars: list
        list of varriance arrays, must be 2d list
    main_title: str
        main title for plot
    savefig: bool
        to save figure in current directory, defaults to False
    
    Returns
    -------
    plot
        combined violin plot for each region and sigma/binarization with each region on one figure

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regions = ['V1', 'LM', 'AL', 'RL']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    all_slope_lists = []
    for i in range(len(means)):
        region_x = means[i]
        region_y = vars[i]
        slope_list = []
        
        for mean_arr, var_arr in zip(region_x, region_y):
            slope_temp = []
            for neuron in range(mean_arr.shape[1]):
                neuron_x = mean_arr[:, neuron]
                neuron_y = var_arr[:, neuron]
                linregress_neuron = linregress(neuron_x, neuron_y)
                slope_temp.append(linregress_neuron.slope)
            slope_list.append(slope_temp)
        all_slope_lists.append(slope_list)
    
    num_conditions = len(all_slope_lists[0])
    violin_data = []
    positions = []
    violin_colors = []
    for cond in range(num_conditions):
        for region_idx in range(len(regions)):
            violin_data.append(all_slope_lists[region_idx][cond])
            positions.append(cond)
            violin_colors.append(colors[region_idx])
    
    vp = ax.violinplot(violin_data, positions=positions, showextrema = False, quantiles = [[0.25, 0.75] for _ in range(len(violin_data))],
                       showmeans=True, showmedians=True, widths=0.2, bw_method = 0.8)
    
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(violin_colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    
    vp['cmeans'].set_color('yellow')
    vp['cmedians'].set_color('white')
    
    ax.set_xticks(range(1, num_conditions + 1))
    ax.set_xticklabels(['3', '15', '30', 'Stochastic\nBinarization'])
    ax.set_xlabel('Sigma + Binarization', fontsize=10)
    
    ax.set_ylabel('Regression Slope', fontsize=10)
    
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=4, label=regions[i]) for i in range(len(regions))]
    ax.legend(handles=legend_elements, title='Regions', loc='upper right')
    
    ax.set_title(main_title, fontsize=12, fontweight='bold', pad=10)
    
    plt.ylim(0, .08 * max([np.max(x) for x in violin_data])) # for reference max is a little under 10 here

    plt.tight_layout()
    
    if savefig:
        date = datetime.now()
        file_date = f'({date.year}-{date.month}-{date.day})'
        path_name = f'{main_title}_{file_date}.pdf'
        plt.savefig(os.path.join(os.getcwd(), 'plots', path_name))
    else:
        plt.show()