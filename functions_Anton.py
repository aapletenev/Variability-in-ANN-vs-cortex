import numpy as np
from scipy import linalg
from typing import Optional, Union, Tuple
import os
from matplotlib.ticker import MaxNLocator


#####functions
#function to return neurons of specific area
def get_neurons_of_area(data, labels, area_code = 1):
    return data[...,:,labels[0,:] == area_code]

def compute_corr(data):
    correlations = []
    for i in range(data.shape[0]):
        corr_matrix = np.ma.corrcoef(data[i, :, :].T)
        correlations.append(corr_matrix)
    return np.array(correlations)


def get_lower_triangle(corr_matrix):
    # 1. Get the size of the last dimension (assuming square matrix in last 2 dims)
    n = corr_matrix.shape[-1]

    # 2. Get the indices for the lower triangle (rows and columns)
    rows, cols = np.tril_indices(n, k=-1)

    # 3. Use Ellipsis (...) to keep all leading dimensions intact
    # This broadcasts the indices over the last two dimensions
    return corr_matrix[..., rows, cols]


def get_cor_zscored(data):
    #z-score data across noise levels for each image
    data_zscored = (data - np.nanmean(data, axis=1, keepdims=True)) / np.nanstd(data, axis=1, keepdims=True)
    data_zscored = data_zscored.reshape(-1, data_zscored.shape[2])
    corr_matrix_zscored = np.ma.corrcoef(data_zscored.T)
    return get_lower_triangle(corr_matrix_zscored), data_zscored

def get_signal_correlation(data_mean):
    data_mean_reshaped = data_mean.reshape(-1, data_mean.shape[1])
    signal_corr_matrix = np.ma.corrcoef(data_mean_reshaped.T)
    return get_lower_triangle(signal_corr_matrix)

def get_cor_topk(data_z, data_mean, k=0.1):
    num_neurons_to_select = int(k * data_mean.shape[1])
    data_mean_reshaped = data_mean.reshape(-1, data_mean.shape[1])
    median_mean_responses = np.nanmedian(data_mean_reshaped, axis=0)
    top_neuron_indices = np.argsort(median_mean_responses)[-num_neurons_to_select:]
    #compute noise correlation matrix for these neurons
    data_z_top = data_z[:, top_neuron_indices]
    corr_matrix_zscored_top = np.ma.corrcoef(data_z_top.T)
    lower_triangle_values_zscored_top = get_lower_triangle(corr_matrix_zscored_top)
    #compute signal correlation matrix for these neurons
    data_mean_top = data_mean_reshaped[:, top_neuron_indices]
    signal_corr_matrix_top = np.ma.corrcoef(data_mean_top.T)
    lower_triangle_values_signal_top = get_lower_triangle(signal_corr_matrix_top)
    return lower_triangle_values_zscored_top, lower_triangle_values_signal_top, top_neuron_indices


def get_correlations_all(Spikes, Mean):
    cor = compute_corr(Spikes)
    print(f'cor NaN count: {np.isnan(cor).sum()}')
    cor_z, Spike_z = get_cor_zscored(Spikes)
    print(f'cor_z NaN count: {np.isnan(cor_z).sum()}')
    signal_cor = get_signal_correlation(Mean)
    print (f'signal_cor NaN count: {np.isnan(signal_cor).sum()}')
    cor_z_top, signal_cor_top, top_neuron_indices = get_cor_topk(Spike_z, Mean, k=0.1)
    return cor, cor_z, signal_cor, cor_z_top, signal_cor_top, top_neuron_indices


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_scatter_noise_vs_signal(ax, noise_cor, signal_cor, noise_cor_top, signal_cor_top,
                                 color='powderblue', color_top='tab:blue',
                                 point_size=0.3, point_alpha=0.01,
                                 title='Noise Correlation vs Signal Correlation',
                                 xlabel='Signal Correlation', ylabel='Noise Correlation',
                                 xline=0, yline=0, xlim=None, ylim=None,
                                 trendline='regression', window_size=500,
                                 yaxis_step=None):  # <--- NEW PARAMETER
    """
    Plots noise correlation vs signal correlation.

    Parameters:
    -----------
    yaxis_step : float, optional
        If provided (e.g., 0.2), forces the y-axis ticks to be at this interval.
        Useful for enforcing consistent decimal places across plots.
    """

    # --- INNER HELPER: Just Clean Data ---
    def get_clean_data(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        return x_clean, y_clean

    # --- INNER HELPER: Plotting Logic ---
    def add_trendline(ax_obj, x, y, label_prefix, line_color, is_top=False):
        stats_text = None
        if len(x) < 2: return None

        if trendline == 'regression':
            res = stats.linregress(x, y)
            m, b = res.slope, res.intercept
            r_squared = res.rvalue ** 2
            x_range = np.array([np.min(x), np.max(x)])
            linestyle = 'dotted' if not is_top else 'solid'
            ax_obj.plot(x_range, m * x_range + b, color='black', linestyle=linestyle,
                        label=f'Fit {label_prefix}')
            stats_text = [f'R² ({label_prefix}): {r_squared:.2f}',
                          f'Slope ({label_prefix}): {m:.2f}']

        elif trendline == 'moving_average' or trendline == 'smooth':
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            if len(x_sorted) > window_size:
                y_smooth = np.convolve(y_sorted, np.ones(window_size) / window_size, mode='valid')
                diff = len(x_sorted) - len(y_smooth)
                start_idx = diff // 2
                end_idx = start_idx + len(y_smooth)
                x_smooth = x_sorted[start_idx:end_idx]
                linestyle = 'dotted' if not is_top else 'solid'
                ax_obj.plot(x_smooth, y_smooth, color='black', linestyle=linestyle,
                            linewidth=1.5, label=f'Smooth {label_prefix}')
            else:
                print(f"Warning: Not enough points ({len(x)}) for window_size ({window_size})")
        return stats_text

    # -----------------------------

    stats_lines = []

    if signal_cor is not None and noise_cor is not None:
        ax.scatter(signal_cor, noise_cor, s=point_size, alpha=point_alpha,
                   label='All Neurons', color=color, facecolors='none')
        x_clean, y_clean = get_clean_data(signal_cor, noise_cor)
        s_txt = add_trendline(ax, x_clean, y_clean, "All", 'black', is_top=False)
        if s_txt: stats_lines.extend(s_txt)

    if signal_cor_top is not None and noise_cor_top is not None:
        ax.scatter(signal_cor_top, noise_cor_top, s=point_size * 3, alpha=point_alpha * 10,
                   label='Top 10% Neurons', color=color_top)
        x_clean_top, y_clean_top = get_clean_data(signal_cor_top, noise_cor_top)
        s_txt_top = add_trendline(ax, x_clean_top, y_clean_top, "Top 10%", 'black', is_top=True)
        if s_txt_top: stats_lines.extend(s_txt_top)

    # --- 3. Standard Plot Formatting ---
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # === NEW OPTION: Manual Y-Axis Steps ===
    if yaxis_step is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yaxis_step))
    # =======================================

    # --- Limits Logic ---
    if xlim is not None:
        if isinstance(xlim, (tuple, list, np.ndarray)):
            ax.set_xlim(left=xlim[0], right=xlim[1])
        else:
            ax.set_xlim(left=xlim)
    if ylim is not None:
        if isinstance(ylim, (tuple, list, np.ndarray)):
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        else:
            ax.set_ylim(bottom=ylim)

    # Lines
    if xline is not None:
        ax.axhline(xline, color='lightgray', linestyle='dashed', linewidth=1)
    if yline is not None:
        ax.axvline(yline, color='lightgray', linestyle='dashed', linewidth=1)

    # --- 4. Legend & Text Box ---
    leg = ax.legend(loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
        if hasattr(lh, 'set_sizes'):
            lh.set_sizes([20])

    if stats_lines:
        textstr = "\n".join(stats_lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

# 2. MODIFIED HISTOGRAM COMPARISON FUNCTION

def plot_hist_comparison(ax, noise_cor, noise_cor_top, nbins=30,
                         color='tab:blue',  # Controls the TOP neurons
                         title='Noise Correlation Distribution',
                         xlabel='Noise Correlation',
                         xlim=None):
    """
    Plots a dual-axis histogram onto a specific axis.
    Automatically creates the secondary y-axis (twinx) inside the function.
    """

    # 1. Setup shared bins
    min_val = min(np.min(noise_cor), np.min(noise_cor_top))
    max_val = max(np.max(noise_cor), np.max(noise_cor_top))
    bins = np.linspace(min_val, max_val, nbins + 1)

    # 2. Plot "All Neurons" on the Main Axis (ax)
    # Background: White with Black Edge
    ax.hist(noise_cor, bins=bins, color='white', edgecolor='black',
            linewidth=1.2, label='All Neurons', zorder=1)

    # 3. Create Twin Axis internally
    ax2 = ax.twinx()

    # 4. Plot "Top Neurons" on Twin Axis (ax2)
    # Foreground: Colored
    ax2.hist(noise_cor_top, bins=bins, color=color, edgecolor=None,
             alpha=0.6, label='Top 10% Neurons', zorder=2)

    # 5. Add Vertical Mean Lines
    mean_all = np.nanmean(noise_cor)
    mean_top = np.nanmean(noise_cor_top)

    ax.axvline(mean_all, color='black', linestyle='dashed', linewidth=1.5)
    ax2.axvline(mean_top, color=color, linestyle='dashed', linewidth=1.5)

    # 6. Labels and Axis Coloring
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count (All Neurons)', color='black')

    # Color the Right Axis (Top Neurons)
    ax2.set_ylabel('Count (Top 10% Neurons)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.spines['right'].set_color(color)
    ax2.spines['right'].set_linewidth(2)

    # 7. Apply Limits and Title
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_title(title)

    # 8. Unified Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='All Neurons'),
        Patch(facecolor=color, alpha=0.6, label='Top 10% Neurons')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # 9. Stats Text Box
    textstr = f'Mean (All): {mean_all:.3f}\nMean (Top 10%): {mean_top:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

def geometric_mean_pairwise(means):
    pairwise_products = np.einsum('ki, kj -> kij', means, means)
    return np.sqrt(pairwise_products)


def plot_dual_axis_boxplot(ax, data1, data2,
                           label1='Bernoulli', label2='Gaussian',
                           color1='tab:orange', color2='tab:blue',
                           title='Fano Factor Comparison',
                           xlabel='Frame', ylabel_suffix='Fano Factor',
                           show_outliers=False, whisk=(25, 75),
                           ylim=None, xline=None, scale=1.0):
    """
    Plots two datasets on a specific axis (ax) using dual Y-axes.
    Adapts to 3D (Samples, Frames, Neurons) or 2D (Frames, Neurons) inputs.
    """

    # --- Font Scaling ---
    s_title = 14 * scale
    s_label = 12 * scale
    s_tick = 10 * scale

    # --- HELPER: Extract and Clean Data ---
    def prepare_data(data):
        def clean_nans(arr):
            flat = arr.flatten()
            return flat[~np.isnan(flat)]

        if data.ndim == 3:
            n_frames = data.shape[1]
            return [clean_nans(data[:, i, :]) for i in range(n_frames)]

        elif data.ndim == 2:
            n_frames = data.shape[0]
            return [clean_nans(data[i, :]) for i in range(n_frames)]

        else:
            raise ValueError(f"Data must be 2D or 3D, but got {data.ndim}D")

    # 1. Prepare Data Lists
    plot_data1 = prepare_data(data1)
    plot_data2 = prepare_data(data2)

    # Ensure both datasets have the same number of frames
    num_frames = len(plot_data1)
    if len(plot_data2) != num_frames:
        print(
            f"Warning: Datasets have different frame counts ({len(plot_data1)} vs {len(plot_data2)}). Using {num_frames}.")

    # 2. Setup Dual Axis
    ax1 = ax
    ax2 = ax1.twinx()

    # 3. Define Positions
    pos1 = np.arange(num_frames) - 0.15
    pos2 = np.arange(num_frames) + 0.15

    # 4. Plot Boxplots
    box1 = ax1.boxplot(plot_data1, positions=pos1, widths=0.3,
                       patch_artist=True, showfliers=show_outliers,
                       whis=whisk)

    box2 = ax2.boxplot(plot_data2, positions=pos2, widths=0.3,
                       patch_artist=True, showfliers=show_outliers,
                       whis=whisk)

    # --- STYLING ---
    def style_boxplot(box_handle, fill_color):
        for item in ['boxes', 'whiskers', 'fliers', 'caps']:
            plt.setp(box_handle[item], color=fill_color)
        plt.setp(box_handle["boxes"], facecolor=fill_color, alpha=0.5)
        plt.setp(box_handle["medians"], color="black", linewidth=1.5)

    style_boxplot(box1, color1)
    style_boxplot(box2, color2)

    # --- AXIS FORMATTING ---
    ax1.set_xlabel(xlabel, fontsize=s_label)

    # Create the full range of ticks and labels
    all_ticks = np.arange(num_frames)
    all_labels = np.arange(1, num_frames + 1)

    # Apply slicing [::2] to show every second tick/label
    ax1.set_xticks(all_ticks[::2])
    ax1.set_xticklabels(all_labels[::2], fontsize=s_tick)

    ax1.set_title(title, fontsize=s_title)

    # Left Axis (Dataset 1)
    ax1.set_ylabel(f'{ylabel_suffix} ({label1})', color=color1, fontweight='bold', fontsize=s_label)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=s_tick)
    ax1.spines['left'].set_color(color1)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_visible(False)

    # Right Axis (Dataset 2)
    ax2.set_ylabel(f'{ylabel_suffix} ({label2})', color=color2, fontweight='bold', fontsize=s_label)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=s_tick)
    ax2.spines['right'].set_color(color2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_visible(False)

    if ylim is not None:
        ax1.set_ylim(bottom=ylim)
        ax2.set_ylim(bottom=ylim)
    if xline is not None:
        ax1.axvline(xline, color='lightgray', linestyle='dashed', linewidth=1)
        ax2.axvline(xline, color='lightgray', linestyle='dashed', linewidth=1)

    return ax1, ax2


def plot_single_axis_boxplot(ax, data1, data2,
                             label1='Bernoulli', label2='Gaussian',
                             color1='tab:orange', color2='tab:blue',
                             title='Fano Factor Comparison',
                             xlabel='Frame', ylabel_suffix='Fano Factor',
                             show_outliers=False, ylim=None,
                             whisk=(25, 75),
                             legend_loc='upper right', xline=None, scale=1.0):
    """
    Plots two datasets on the SAME axis (ax) side-by-side.
    Handles the case where data2 is None.
    """

    # --- Font Scaling ---
    s_title = 14 * scale
    s_label = 12 * scale
    s_tick = 10 * scale
    s_legend = 10 * scale

    # --- HELPER: Extract and Clean Data ---
    def prepare_data(data):
        def clean_nans(arr):
            flat = arr.flatten()
            return flat[~np.isnan(flat)]

        if data.ndim == 3:
            n_frames = data.shape[1]
            return [clean_nans(data[:, i, :]) for i in range(n_frames)]
        elif data.ndim == 2:
            n_frames = data.shape[0]
            return [clean_nans(data[i, :]) for i in range(n_frames)]
        else:
            raise ValueError(f"Data must be 2D or 3D, but got {data.ndim}D")

    # 1. Prepare Data Lists
    plot_data1 = prepare_data(data1)
    num_frames = len(plot_data1)

    # 2. Determine Logic based on data2 presence
    if data2 is not None:
        plot_data2 = prepare_data(data2)
        pos1 = np.arange(num_frames) - 0.15
        pos2 = np.arange(num_frames) + 0.15

        box1 = ax.boxplot(plot_data1, positions=pos1, widths=0.3,
                          patch_artist=True, showfliers=show_outliers,
                          whis=whisk)
        box2 = ax.boxplot(plot_data2, positions=pos2, widths=0.3,
                          patch_artist=True, showfliers=show_outliers,
                          whis=whisk)

        legend_handles = [box1["boxes"][0], box2["boxes"][0]]
        legend_labels = [label1, label2]

    else:
        pos1 = np.arange(num_frames)
        box1 = ax.boxplot(plot_data1, positions=pos1, widths=0.4,
                          patch_artist=True, showfliers=show_outliers,
                          whis=whisk)
        box2 = None
        legend_handles = [box1["boxes"][0]]
        legend_labels = [label1]

    # --- STYLING ---
    def style_boxplot(box_handle, fill_color):
        for item in ['boxes', 'whiskers', 'fliers', 'caps']:
            plt.setp(box_handle[item], color=fill_color)
        plt.setp(box_handle["boxes"], facecolor=fill_color, alpha=0.5)
        plt.setp(box_handle["medians"], color="black", linewidth=1.5)

    style_boxplot(box1, color1)
    if data2 is not None:
        style_boxplot(box2, color2)

    # --- AXIS FORMATTING ---
    ax.set_xlabel(xlabel, fontsize=s_label)

    # Create the full range of ticks and labels
    all_ticks = np.arange(num_frames)
    all_labels = np.arange(1, num_frames + 1)

    # Apply slicing [::2] to show every second tick/label
    ax.set_xticks(all_ticks[::2])
    ax.set_xticklabels(all_labels[::2], fontsize=s_tick)

    ax.set_title(title, fontsize=s_title)
    ax.set_ylabel(ylabel_suffix, fontweight='bold', fontsize=s_label)
    ax.tick_params(axis='y', labelsize=s_tick)

    # Apply Legend with Scaled Font
    ax.legend(legend_handles, legend_labels, loc=legend_loc, fontsize=s_legend)

    if ylim is not None:
        ax.set_ylim(bottom=ylim)
    if xline is not None:
        ax.axvline(xline, color='lightgray', linestyle='dashed', linewidth=1)

    return ax


#fast function to compute slopes using vectorized operations

def compute_slope_var_mean(Mean, Var):
    """
    Computes slope and R^2 for Var = slope * Mean (regression through origin).
    Vectorized over images.

    Returns:
    --------
    slopes : np.ndarray (num_frames, num_neurons)
    r2_scores : np.ndarray (num_frames, num_neurons)
    """
    # 1. Create a mask of valid data
    valid_mask = np.isfinite(Mean) & np.isfinite(Var)

    # 2. Clean data (0s are placeholders, ignored via mask later)
    M_clean = np.where(valid_mask, Mean, 0.0)
    V_clean = np.where(valid_mask, Var, 0.0)

    # --- PART A: Calculate Slope ---

    # Sum(x*y) and Sum(x*x)
    numerator = np.sum(M_clean * V_clean, axis=0)
    denominator = np.sum(M_clean ** 2, axis=0)

    slopes = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=slopes, where=denominator != 0)

    # --- PART B: Calculate R^2 ---

    # 1. Calculate SS_res (Residual Sum of Squares)
    # Prediction: y_hat = slope * x
    # We broadcast slope (F, N) to match M_clean (I, F, N)
    y_pred = slopes[np.newaxis, :, :] * M_clean

    # Residuals: (y - y_hat)^2
    residuals_sq = (V_clean - y_pred) ** 2

    # Zero out invalid residuals before summing
    residuals_sq = np.where(valid_mask, residuals_sq, 0.0)
    ss_res = np.sum(residuals_sq, axis=0)

    # 2. Calculate SS_tot (Total Sum of Squares)
    # Formula: Sum((y - mean_y)^2)
    # Optimized Formula: Sum(y^2) - (Sum(y)^2 / N)

    valid_counts = np.sum(valid_mask, axis=0)
    sum_y = np.sum(V_clean, axis=0)
    sum_y_sq = np.sum(V_clean ** 2, axis=0)

    # Initialize ss_tot
    ss_tot = np.zeros_like(sum_y)

    # Calculate SS_tot only where we have valid data to avoid div/0
    mask_counts = valid_counts > 0
    term2 = np.zeros_like(sum_y)
    np.divide(sum_y[mask_counts] ** 2, valid_counts[mask_counts], out=term2[mask_counts])
    ss_tot = sum_y_sq - term2

    # 3. Calculate R^2 = 1 - (SS_res / SS_tot)
    r2_scores = np.full(slopes.shape, np.nan)

    # Avoid division by zero if SS_tot is 0 (perfect flat line) or counts < 2
    # Note: R^2 can be negative for regression through origin
    calc_mask = (ss_tot > 0) & (valid_counts > 1)

    np.divide(ss_res, ss_tot, out=r2_scores, where=calc_mask)
    r2_scores = 1 - r2_scores

    # --- PART C: Final Cleanup ---

    # Enforce NaN where insufficient data (<= 1 point)
    invalid_final = valid_counts <= 1
    slopes[invalid_final] = np.nan
    r2_scores[invalid_final] = np.nan

    # Also mask R2 where SS_tot was 0 (undefined R2, usually constant variance)
    r2_scores[ss_tot == 0] = np.nan

    return slopes, r2_scores



def smooth_func(x, a, b):
    return a * (x**b)
def fit_lin_power(x, y, a_initial = 1e-3, b_initial = 1e-1):
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan  # a_lin, b_lin, r2_lin
    xm = x[mask]; ym = y[mask]
    try:
        params, _ = curve_fit(smooth_func, xm, ym, maxfev=1_000_000, p0=[a_initial, b_initial])
        a_lin, b_lin = params
        y_hat = smooth_func(xm, a_lin, b_lin)
        ss_res = np.sum((ym - y_hat)**2)
        ss_tot = np.sum((ym - np.mean(ym))**2)
        r2_lin = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        return a_lin, b_lin, r2_lin
    except Exception:
        return np.nan, np.nan, np.nan


def compute_global_frame_fits(mean_data, var_data):
    """
    Aggregates all neurons and images per frame to find a global 'a' and 'b'.
    Returns dictionary of lists (length = num_frames).
    """
    num_frames = mean_data.shape[1]
    results = {'a': [], 'b': [], 'r2': []}

    for frame_idx in range(num_frames):
        # Flatten across images AND neurons
        x_flat = mean_data[:, frame_idx, :].flatten()
        y_flat = var_data[:, frame_idx, :].flatten()

        a, b, r2 = fit_lin_power(x_flat, y_flat)

        results['a'].append(a)
        results['b'].append(b)
        results['r2'].append(r2)

    return results


def compute_neuron_fits(mean_data, var_data, global_init_params=None):
    """
    Computes fit for each neuron individually per frame.

    Returns:
    --------
    a_mat, b_mat, r2_mat : np.ndarray
        Shape (num_frames, num_neurons)  <-- CHANGED to Frame x Neuron
    """
    num_images, num_frames, num_neurons = mean_data.shape

    # Pre-allocate arrays (Frames x Neurons)
    a_mat = np.zeros((num_frames, num_neurons))
    b_mat = np.zeros((num_frames, num_neurons))
    r2_mat = np.zeros((num_frames, num_neurons))

    for frame_idx in range(num_frames):

        # Determine initial guesses for this frame
        p0_a = 1e-3
        p0_b = 1e-3

        if global_init_params:
            if not np.isnan(global_init_params['a'][frame_idx]):
                p0_a = global_init_params['a'][frame_idx]
                p0_b = global_init_params['b'][frame_idx]

        for neuron_idx in range(num_neurons):
            x_neuron = mean_data[:, frame_idx, neuron_idx]
            y_neuron = var_data[:, frame_idx, neuron_idx]

            a, b, r2 = fit_lin_power(x_neuron, y_neuron,
                                     a_initial=p0_a,
                                     b_initial=p0_b)

            # Index by [frame, neuron]
            a_mat[frame_idx, neuron_idx] = a
            b_mat[frame_idx, neuron_idx] = b
            r2_mat[frame_idx, neuron_idx] = r2

    return a_mat, b_mat, r2_mat


###compute linear fisher info with kanitscheider correction


def compute_fisher_info(spike_counts, d_theta=1, method='cholesky', reg=1e-3, nan_fill_value=100.0, shuffle=False):
    """
    Computes Fisher Information.

    If shuffle=False: Returns Bias-Corrected Linear Fisher Information (I_bc).
    If shuffle=True:  Returns Naive Linear Fisher Information (I_naive) to avoid
                      over-penalizing diagonal matrices.

    Parameters
    ----------
    spike_counts : np.ndarray
        Shape (2, n_trials, n_neurons).
    d_theta : float
        Difference in stimulus parameter.
    method : str
        'simple' or 'cholesky'.
    reg : float
        Regularization term for stability.
    nan_fill_value : float
        Value to replace NaNs with.
    shuffle : bool
        If True, computes 'Shuffled Fisher Information' (Is) by removing
        noise correlations (diagonalizing the covariance matrix).

    Returns
    -------
    I_out, var_out : (float, float)
        I_out is I_bc (if shuffle=False) or I_naive (if shuffle=True).
        var_out is var_I_bc (if shuffle=False) or 0.0 (if shuffle=True).
    """
    # 0. Pre-processing: Handle NaNs
    X = np.array(spike_counts, dtype=float, copy=True)
    if np.isnan(X).any():
        X[np.isnan(X)] = nan_fill_value

    n_conds, T, N = X.shape

    # Correction constraint check
    v = (2 * T) - 2
    if v - N - 3 <= 0:
        raise ValueError(f"Insufficient trials. Requirement: 2*T > N + 5. (Got T={T}, N={N})")

    # 1. Compute Statistics
    X1, X2 = X[0], X[1]

    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)
    d_mu = mu1 - mu2
    d_mu_d_theta = d_mu / d_theta

    # --- COVARIANCE CALCULATION ---
    if method == 'simple':
        cov1 = np.cov(X1, rowvar=False)
        cov2 = np.cov(X2, rowvar=False)
        S = (cov1 + cov2) / 2.0

        if shuffle:
            S = np.diag(np.diag(S))

        if N == 1:
            S = np.array([[S.item()]])

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        I_naive = d_mu_d_theta.T @ S_inv @ d_mu_d_theta

        if isinstance(I_naive, np.ndarray):
            I_naive = I_naive.item()

    elif method == 'cholesky':
        X1_c = X1 - mu1
        X2_c = X2 - mu2
        X_combined = np.concatenate([X1_c, X2_c], axis=0)

        S = (X_combined.T @ X_combined) / (2 * T - 2)

        if shuffle:
            S = np.diag(np.diag(S))

        S.flat[::N + 1] += reg

        try:
            c, lower = linalg.cho_factor(S, lower=True)
            x = linalg.cho_solve((c, lower), d_mu_d_theta)
            I_naive = np.dot(d_mu_d_theta, x)
        except linalg.LinAlgError:
            I_naive = d_mu_d_theta @ np.linalg.pinv(S) @ d_mu_d_theta

    else:
        raise ValueError("Method must be 'simple' or 'cholesky'")

    # --- MODIFICATION START ---
    # If shuffled, return Naive estimate directly to avoid invalid bias correction
    # on diagonal matrices.
    if shuffle:
        return I_naive, 0.0
    # --- MODIFICATION END ---

    # 3. Bias Correction (Only for Real Data)
    numerator = 2 * T - N - 3
    denominator = 2 * T - 2
    factor = numerator / denominator
    subtraction = (2 * T * N) / (T * T * (d_theta ** 2))

    I_bc = I_naive * factor - subtraction

    # 4. Variance Calculation (Only for Real Data)
    denom_common = (v - N) * (v - N - 3)
    alpha = 2 / denom_common
    beta = (v - N - 1) / denom_common
    gamma = (2 * T) / (T * T * (d_theta ** 2))

    var_I_bc = (alpha + 2 * beta) * (I_bc ** 2) + \
               (6 * alpha + 12 * beta + 4) * gamma * I_bc + \
               (3 * alpha + 6 * beta + 2) * (gamma ** 2) * N

    return I_bc, var_I_bc


def compute_pixel_fisher_info(image_stack, d_theta=1.0):
    """
    Computes Unbiased Pixel Fisher Information using Split-Half Cross-Validation.

    This method has ZERO bias by construction, because noise in Split A
    is uncorrelated with noise in Split B.

    Parameters
    ----------
    image_stack : np.ndarray
        Shape (2, n_trials, n_pixels).

    Returns
    -------
    I_cv : float
        The unbiased Fisher Information estimate.
    """
    n_conds, n_trials, n_pixels = image_stack.shape

    # 1. Split Data into Halves (Train / Test)
    # We use integer division to ensure equal splits
    mid = n_trials // 2

    # Split A
    imgs_A = image_stack[:, :mid, :]
    mu1_A = np.mean(imgs_A[0], axis=0)
    mu2_A = np.mean(imgs_A[1], axis=0)
    diff_A = mu1_A - mu2_A

    # Split B
    imgs_B = image_stack[:, mid:2 * mid, :]
    mu1_B = np.mean(imgs_B[0], axis=0)
    mu2_B = np.mean(imgs_B[1], axis=0)
    diff_B = mu1_B - mu2_B

    # 2. Compute Signal Energy via Dot Product (Cross-Validated)
    # E[diff_A . diff_B] = True_Signal^2 + 0 (Noise cancels out)
    signal_energy = np.dot(diff_A, diff_B)

    # 3. Normalize by Variance (using all data for better stability)
    var1 = np.var(image_stack[0], axis=0, ddof=1)
    var2 = np.var(image_stack[1], axis=0, ddof=1)
    sigma_sq_pooled = (np.mean(var1) + np.mean(var2)) / 2.0

    # Fisher Info = Signal / (Variance * dTheta^2)
    I_cv = signal_energy / (sigma_sq_pooled * d_theta ** 2)

    return I_cv


def compute_fisher_info_all(data_tensor, n_image_pairs=10, n_repeats=50, n_list=None,
                            d_theta=1.0, image_pairs=None, mode='neurons'):
    """
    Computes Fisher Information.

    Modes
    -----
    'neurons': Hierarchical scaling (bootstraps neurons).
    'pixels' : High-dimensional Ideal Observer (no subsampling).

    Parameters
    ----------
    data_tensor : np.ndarray
        Shape (n_images, n_trials, n_features).
        Features are Neurons (if mode='neurons') or Pixels (if mode='pixels').
    n_image_pairs : int
        Number of random image pairs (ignored if image_pairs is provided).
    n_repeats : int
        Number of subsamples per N (only for mode='neurons').
    n_list : list, optional
        List of population sizes N (only for mode='neurons').
    d_theta : float
        Stimulus difference (default 1.0).
    image_pairs : list, optional
        Specific pairs to test.
    mode : str
        'neurons' or 'pixels'.

    Returns
    -------
    If mode='neurons':
        (FI_real, FI_shuf, n_list, used_image_pairs)
    If mode='pixels':
        (FI_pixels, None, None, used_image_pairs)
    """

    # 1. Setup Image Pairs
    total_images = data_tensor.shape[0]

    if image_pairs is not None:
        used_image_pairs = np.array(image_pairs, dtype=int)
        actual_n_pairs = len(used_image_pairs)
        print(f"Using {actual_n_pairs} pre-defined image pairs.")
    else:
        actual_n_pairs = n_image_pairs
        used_image_pairs = np.zeros((actual_n_pairs, 2), dtype=int)
        for k in range(actual_n_pairs):
            used_image_pairs[k] = np.random.choice(total_images, 2, replace=False)
        print(f"Generated {actual_n_pairs} random image pairs.")

    print(f"Starting analysis (Mode: {mode})...")

    # --- MODE: PIXELS ---
    if mode == 'pixels':
        # Output shape: (n_pairs,)
        FI_pixels = np.zeros(actual_n_pairs)

        for p in range(actual_n_pairs):
            img_indices = used_image_pairs[p]

            # Slice pair: (2, n_trials, n_pixels)
            current_pair_data = data_tensor[img_indices]

            # Compute Ideal Observer FI
            FI_pixels[p] = compute_pixel_fisher_info(current_pair_data, d_theta=d_theta)

            if (p + 1) % 10 == 0:
                print(f"Finished Pair {p + 1}/{actual_n_pairs}")

        return FI_pixels, None, None, used_image_pairs

    # --- MODE: NEURONS ---
    elif mode == 'neurons':
        # Setup N list
        total_neurons = data_tensor.shape[2]
        if n_list is None:
            n_list = np.linspace(2, 50, 10, dtype=int)
            n_list = np.unique(n_list)
        n_sizes = len(n_list)

        FI_real = np.zeros((actual_n_pairs, n_repeats, n_sizes))
        FI_shuf = np.zeros((actual_n_pairs, n_repeats, n_sizes))

        print(f"For each pair, sampling {n_repeats} repeats for N in {n_list}")

        for p in range(actual_n_pairs):
            img_indices = used_image_pairs[p]

            # Optimization: Copy slice once
            current_pair_data = data_tensor[img_indices].copy()

            for i, N in enumerate(n_list):
                for r in range(n_repeats):
                    neuron_indices = np.random.choice(total_neurons, N, replace=False)
                    sub_data = current_pair_data[:, :, neuron_indices]

                    # Real FI
                    val_real, _ = compute_fisher_info(
                        sub_data, d_theta=d_theta, method='cholesky', shuffle=False
                    )
                    # Shuffled FI
                    val_shuf, _ = compute_fisher_info(
                        sub_data, d_theta=d_theta, method='simple', shuffle=True
                    )

                    FI_real[p, r, i] = val_real
                    FI_shuf[p, r, i] = val_shuf

            if actual_n_pairs < 20 or (p + 1) % 10 == 0:
                print(f"Finished Pair {p + 1}/{actual_n_pairs} (Idx: {img_indices})")

        return FI_real, FI_shuf, n_list, used_image_pairs

    else:
        raise ValueError("Mode must be 'neurons' or 'pixels'")


def plot_scaling_metric(data, n_list, aggregation='median', title="Fisher Information Scaling",
                        ylabel="Fisher Information", color='gray', line_width=1.5,
                        yline=None, ylim=None, xlim = 0, FI_pixel=None, scale=1.0, ax=None):
    """
    Plots scaling curves for multiple image pairs with optional Pixel Information bounds
    and font scaling.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_image_pairs, n_repeats, n_sizes).
    n_list : array-like
        List of N values (x-axis).
    aggregation : str
        'mean' or 'median' over repeats.
    title : str
        Title of the plot.
    ylabel : str
        Label for the y-axis.
    color : str
        Color of the lines (default 'gray').
    line_width : float
        Width of the solid lines.
    yline : float, optional
        A single global horizontal reference line (e.g., 0).
    ylim : tuple, optional
        (ymin, ymax) limits for the y-axis.
    FI_pixel : np.ndarray, optional
        Array of shape (n_image_pairs,).
        If provided, plots a dashed horizontal line for each image pair.
    scale : float
        Scaling factor for font sizes (default 1.0).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Base font sizes
    base_label_size = 12
    base_title_size = 14
    base_tick_size = 10

    # Calculate scaled sizes
    label_size = base_label_size * scale
    title_size = base_title_size * scale
    tick_size = base_tick_size * scale

    # Check input shape
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D (pairs, repeats, sizes). Got shape {data.shape}")

    # Aggregate over repeats (axis 1)
    if aggregation == 'mean':
        y_values = np.mean(data, axis=1)
    elif aggregation == 'median':
        y_values = np.median(data, axis=1)
    else:
        raise ValueError("Aggregation must be 'mean' or 'median'")

    n_pairs = y_values.shape[0]

    # Validation for Pixel FI
    if FI_pixel is not None:
        if len(FI_pixel) != n_pairs:
            print(f"Warning: FI_pixel length ({len(FI_pixel)}) does not match data pairs ({n_pairs}).")

    # Plot each pair
    for i in range(n_pairs):
        # 1. Neural Scaling (Solid Line)
        ax.plot(n_list, y_values[i],
                linestyle='-',
                linewidth=line_width,
                marker=None,
                color=color,
                alpha=0.6)

        # 2. Pixel Limit (Dashed Line)
        if FI_pixel is not None:
            ax.plot(n_list, [FI_pixel[i]] * len(n_list),
                    linestyle='--',
                    linewidth=line_width,
                    color=color,
                    alpha=0.4)

    # Set Labels with scaled font sizes
    ax.set_xlabel("Number of Neurons", fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_title(f"{title}", fontsize=title_size)

    # Scale Tick Labels
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # Clean formatting
    ax.grid(False)

    if yline is not None:
        ax.axhline(yline, color='black', linestyle='dashed', linewidth=1)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(left = xlim)

    return ax


def stochastic_binarization_vectorized(image_stack: np.ndarray) -> np.ndarray:
    """Vectorized Stochastic Binarization."""
    probs = image_stack / 255.0
    binarized = np.random.binomial(1, probs)
    return (binarized * 255).astype('uint8')


def generate_noise_chunk(
        image_input: np.ndarray,
        num_trials: int,
        num_frames: int,
        noise_type: str,
        stochastic_bin_param: bool,
        sigma: float
) -> np.ndarray:
    """
    Generates noise for a SINGLE image expanded over trials and frames.
    Input image_input shape: (H, W)
    Output shape: (num_trials, num_frames, H, W)
    """
    h, w = image_input.shape
    full_shape = (num_trials, num_frames, h, w)

    # Expand image to (Trials, Frames, H, W) for broadcasting
    # shape (1, 1, H, W) -> broadcast to full
    image_view = np.broadcast_to(
        image_input[np.newaxis, np.newaxis, :, :],
        full_shape
    )

    if stochastic_bin_param:
        if noise_type == 'constant':
            # 1. Generate noise for (Trials, 1, H, W)
            # We treat the first frame as the seed for the whole trial
            single_frame_view = image_view[:, 0:1, :, :]
            binarized_single = stochastic_binarization_vectorized(single_frame_view)

            # 2. Broadcast back to all frames (Trials, Frames, H, W)
            return np.broadcast_to(binarized_single, full_shape).copy()

        elif noise_type == 'dynamic':
            # Generate unique noise for every frame in every trial
            return stochastic_binarization_vectorized(image_view)

    else:
        # Gaussian Noise Logic
        if noise_type == 'constant':
            noise = np.random.normal(loc=0, scale=sigma, size=(num_trials, 1, h, w))
            noise = np.broadcast_to(noise, full_shape)
        elif noise_type == 'dynamic':
            noise = np.random.normal(loc=0, scale=sigma, size=full_shape)
        else:
            # No noise
            return image_view.copy()  # Return plain image copies

        return np.clip(image_view + noise, 0, 255).astype('uint8')


def process_images_batched(
        images: np.ndarray,
        num_trials: int,
        num_frames: int,
        noise_type: str,
        stochastic_bin_param: bool,
        sigma: float = 10,
        return_sum_over_frames: bool = False
) -> np.ndarray:
    """
    Main function that handles batching to prevent Memory Errors.

    Parameters
    ----------
    return_sum_over_frames : bool
        If True, sums the frames axis immediately to save memory.
        Output shape becomes (N, Trials, H, W) instead of (N, Trials, Frames, H, W).
    """
    results_list = []

    # Loop over images one by one to keep RAM usage low (approx 2GB peak per iter)
    for i in range(len(images)):

        # 1. Generate the heavy (1000, 15, H, W) array for just THIS image
        chunk_output = generate_noise_chunk(
            images[i],
            num_trials,
            num_frames,
            noise_type,
            stochastic_bin_param,
            sigma
        )

        # 2. Aggregation Step
        if return_sum_over_frames:
            # Sum over axis 1 (the frames axis in the chunk: Trials, Frames, H, W)
            # Result shape: (1000, H, W)
            # We use float32 to prevent overflow (uint8 sums max out at 255)
            chunk_sum = np.sum(chunk_output, axis=1, dtype=np.float32)
            results_list.append(chunk_sum)
        else:
            # Keep the full 4D chunk
            results_list.append(chunk_output)

    # 3. Stack all image results together
    # Final Shape: (19, 1000, H, W) if summed
    # Final Shape: (19, 1000, 15, H, W) if not summed
    return np.stack(results_list, axis=0)


def save_predictions(data, path, filename, dirs = ['frames', 'sum', 'mean', 'var', 'label']):
    for dir in dirs:
        os.makedirs(path + dir, exist_ok=True)
    for i,dir in enumerate(dirs):
        np.save(path + dir + '/' + filename, data[i])


def plot_spike_comparison(ref_array, title_text, stack_array=None, line_color='blue', font_scale=1.0, ylim = None, ax=None):
    """
    Plots a single reference line (black). Optionally plots a stack of background lines.
    Can be used as a standalone plot or as part of a subplot.
    """

    # Logic: If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        is_standalone = True
    else:
        is_standalone = False

    # Calculate scaled font sizes
    title_size = 14 * font_scale
    label_size = 12 * font_scale
    tick_size = 10 * font_scale

    # Define x-axis based on the reference array length
    x_axis = np.arange(ref_array.shape[0]) + 1

    # 1. Plot the 2D stack (if provided)
    if stack_array is not None:
        for i in range(stack_array.shape[0]):
            ax.plot(x_axis, stack_array[i, :], color=line_color, alpha=0.4, linewidth=1.5)

    # 2. Plot the 1D reference array
    ax.plot(x_axis, ref_array, color='black', linewidth=2.5, label='Reference')

    # 3. Formatting
    ax.set_title(title_text, fontsize=title_size)
    ax.set_xlabel("Frames", fontsize=label_size)
    ax.set_ylabel("Spike Count", fontsize=label_size)

    # Force Integer Ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', labelsize=tick_size)

    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. Only show/tight_layout if we created the figure
    if is_standalone:
        plt.tight_layout()
        plt.show()

    if ylim is not None:
        ax.set_ylim((0,ylim))
    return ax

def plot_mean_comparison(ax, mean_no_noise, mean_with_noise, title, color):
    ax.scatter(mean_no_noise.flatten(), mean_with_noise.flatten(),
               color=color,
               s=1,
               alpha=0.1)
    #add regression line
    mask = np.isfinite(mean_no_noise) & np.isfinite(mean_with_noise)
    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_no_noise[mask].flatten(), mean_with_noise[mask].flatten())
    x_vals = np.array([0, 100])
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='red', linestyle='-', label=f'Regression line (slope={slope:.2f})')
    ax.set_xlabel('Mean spike count (No Noise)', fontsize=16)
    ax.set_ylabel('Mean spike count (With Noise)', fontsize=16)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.plot([0, 100], [0, 100], color='black', linestyle='--')
    ax.set_title(title, fontsize=16)

