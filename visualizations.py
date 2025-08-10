import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from scipy.stats import shapiro, linregress

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

