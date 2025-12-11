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
