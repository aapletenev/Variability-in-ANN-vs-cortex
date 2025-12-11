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
                           show_outliers=False, ylim = None):
    """
    Plots two datasets on a specific axis (ax) using dual Y-axes.
    Adapts to 3D (Samples, Frames, Neurons) or 2D (Frames, Neurons) inputs.
    """

    # --- HELPER: Extract and Clean Data ---
    def prepare_data(data):
        """
        Converts 2D or 3D arrays into a list of 1D arrays (one per frame).
        Removes NaNs.
        """

        # Inner helper to flatten and clean NaNs
        def clean_nans(arr):
            flat = arr.flatten()
            return flat[~np.isnan(flat)]

        if data.ndim == 3:
            # ASSUMPTION: Shape is (Samples, Frames, Neurons)
            # We iterate over axis 1 (Frames)
            n_frames = data.shape[1]
            return [clean_nans(data[:, i, :]) for i in range(n_frames)]

        elif data.ndim == 2:
            # ASSUMPTION: Shape is (Frames, Neurons)
            # We iterate over axis 0 (Frames)
            n_frames = data.shape[0]
            return [clean_nans(data[i, :]) for i in range(n_frames)]

        else:
            raise ValueError(f"Data must be 2D or 3D, but got {data.ndim}D")

    # 1. Prepare Data Lists
    plot_data1 = prepare_data(data1)
    plot_data2 = prepare_data(data2)

    # Ensure both datasets have the same number of frames for plotting
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

    # 4. Plot Boxplots with 95% Whiskers
    box1 = ax1.boxplot(plot_data1, positions=pos1, widths=0.3,
                       patch_artist=True, showfliers=show_outliers,
                       whis=(2.5, 97.5))

    box2 = ax2.boxplot(plot_data2, positions=pos2, widths=0.3,
                       patch_artist=True, showfliers=show_outliers,
                       whis=(2.5, 97.5))

    # --- STYLING ---
    def style_boxplot(box_handle, fill_color):
        for item in ['boxes', 'whiskers', 'fliers', 'caps']:
            plt.setp(box_handle[item], color=fill_color)
        plt.setp(box_handle["boxes"], facecolor=fill_color, alpha=0.5)
        plt.setp(box_handle["medians"], color="black", linewidth=1.5)

    style_boxplot(box1, color1)
    style_boxplot(box2, color2)

    # --- AXIS FORMATTING ---
    ax1.set_xlabel(xlabel)
    ax1.set_xticks(np.arange(num_frames))
    ax1.set_xticklabels(np.arange(1, num_frames + 1))
    ax1.set_title(title)

    # Left Axis (Dataset 1)
    ax1.set_ylabel(f'{ylabel_suffix} ({label1})', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.spines['left'].set_color(color1)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_visible(False)

    # Right Axis (Dataset 2)
    ax2.set_ylabel(f'{ylabel_suffix} ({label2})', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.spines['right'].set_color(color2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_visible(False)

    if ylim is not None:
        ax1.set_ylim(bottom = ylim)
        ax2.set_ylim(bottom = ylim)

    return ax1, ax2


import numpy as np
import matplotlib.pyplot as plt


def plot_single_axis_boxplot(ax, data1, data2,
                             label1='Bernoulli', label2='Gaussian',
                             color1='tab:orange', color2='tab:blue',
                             title='Fano Factor Comparison',
                             xlabel='Frame', ylabel_suffix='Fano Factor',
                             show_outliers=False, ylim=None,
                             legend_loc='upper right'):  # <--- NEW PARAMETER
    """
    Plots two datasets on the SAME axis (ax) side-by-side.
    Adapts to 3D (Samples, Frames, Neurons) or 2D (Frames, Neurons) inputs.

    Parameters:
    -----------
    legend_loc : str or int
        Position of the legend (e.g., 'upper right', 'upper left', 'best', etc.)
    """

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

    num_frames = len(plot_data1)
    if len(plot_data2) != num_frames:
        print(
            f"Warning: Datasets have different frame counts ({len(plot_data1)} vs {len(plot_data2)}). Using {num_frames}.")

    # 2. Define Positions
    pos1 = np.arange(num_frames) - 0.15
    pos2 = np.arange(num_frames) + 0.15

    # 3. Plot Boxplots
    box1 = ax.boxplot(plot_data1, positions=pos1, widths=0.3,
                      patch_artist=True, showfliers=show_outliers,
                      whis=(2.5, 97.5))

    box2 = ax.boxplot(plot_data2, positions=pos2, widths=0.3,
                      patch_artist=True, showfliers=show_outliers,
                      whis=(2.5, 97.5))

    # --- STYLING ---
    def style_boxplot(box_handle, fill_color):
        for item in ['boxes', 'whiskers', 'fliers', 'caps']:
            plt.setp(box_handle[item], color=fill_color)
        plt.setp(box_handle["boxes"], facecolor=fill_color, alpha=0.5)
        plt.setp(box_handle["medians"], color="black", linewidth=1.5)

    style_boxplot(box1, color1)
    style_boxplot(box2, color2)

    # --- AXIS FORMATTING ---
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(num_frames))
    ax.set_xticklabels(np.arange(1, num_frames + 1))
    ax.set_title(title)

    ax.set_ylabel(ylabel_suffix, fontweight='bold')

    # --- LEGEND (Updated) ---
    ax.legend([box1["boxes"][0], box2["boxes"][0]],
              [label1, label2],
              loc=legend_loc)  # <--- USED HERE

    if ylim is not None:
        ax.set_ylim(bottom=ylim)

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
def fit_lin_power(x, y, a_initial = 1.0, b_initial = 1e-3):
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
        p0_a = 1.0
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
