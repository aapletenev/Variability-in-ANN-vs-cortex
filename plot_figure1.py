"""
Plot Figure 1 - Complete Example Usage

This script demonstrates how to use the functions from figure1_utils.py to:
1. Load and preprocess neuron data
2. Collect pixel-space statistics
3. Fit power law models to both neuron and pixel data
4. Generate Figure 1 with all subpanels

Usage:
    python plot_figure1.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress

from figure1_utils import preprocess_neuron_data, figure1_collection, filter_x_region


# ==============================================================================
# FITTING FUNCTIONS
# ==============================================================================

def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * (x ** b)


def axb_noslope(x, a, b):
    """Power law function without slope: y = a * x^b"""
    return a * (x ** b)


def smooth_from_logfit(x, log_intercept, b):
    """Convert log-space fit to linear space"""
    a = np.exp(log_intercept)
    return a * (x ** b)


def fit_log_power(x, y):
    """Fit power law in log-log space"""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 1.0, np.nan, np.nan
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    model = LinearRegression(fit_intercept=False)
    model.fit(lx.reshape(-1, 1), ly)
    b = model.coef_[0]
    y_pred = model.predict(lx.reshape(-1, 1))
    ss_res = np.sum((ly - y_pred)**2)
    ss_tot = np.sum((ly - np.mean(ly))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return 1.0, b, r2


def fit_lin_power(x, y, a_initial=1.0, b_initial=1e-3):
    """Fit power law in linear space"""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    xm = x[mask]
    ym = y[mask]
    try:
        params, _ = curve_fit(axb_noslope, xm, ym, maxfev=1_000_000, p0=[a_initial, b_initial])
        a_lin, b_lin = params
        y_hat = axb_noslope(xm, a_lin, b_lin)
        ss_res = np.sum((ym - y_hat)**2)
        ss_tot = np.sum((ym - np.mean(ym))**2)
        r2_lin = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        return a_lin, b_lin, r2_lin
    except Exception:
        return np.nan, np.nan, np.nan


def fit_neuron_data(mean_data, var_data, a_initial=1.0, b_initial=1e-3):
    """
    Fit power law models to neuron-level data.
    
    Returns:
    --------
    dict with keys:
        - alist_lin: Linear-space 'a' parameters per neuron
        - blist_lin: Linear-space 'b' parameters per neuron  
        - rsqlist_lin: R² values from power law fits
        - rsqlist_lin_model: R² values from linear model fits
        - a_glob_lin: Global power law 'a' parameter
        - b_glob_lin: Global power law 'b' parameter
        - slope_glob: Global linear model slope
        - intercept_glob: Global linear model intercept
        - pooled_rsq_lin: Per-neuron R² using global power law params
        - pooled_rsq_lin_model: Per-neuron R² using global linear params
    """
    n_neurons = mean_data.shape[-1]
    
    # Initialize containers
    alist_lin, blist_lin, rsqlist_lin = [], [], []
    rsqlist_lin_model = []
    
    # Fit per-neuron
    for neuron in range(n_neurons):
        x = mean_data[..., neuron].astype(float)
        y = var_data[..., neuron].astype(float)
        
        # Linear-space power law fit
        a_lin, b_lin, r2_lin = fit_lin_power(x, y, a_initial=a_initial, b_initial=b_initial)
        alist_lin.append(a_lin)
        blist_lin.append(b_lin)
        rsqlist_lin.append(r2_lin)
        
        # Simple linear model fit (y = mx + b)
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            rsqlist_lin_model.append(np.nan)
        else:
            slope, intercept, r_val, p_val, std_err = linregress(x[mask], y[mask])
            rsqlist_lin_model.append(r_val**2)
    
    # Convert to arrays
    alist_lin = np.array(alist_lin)
    blist_lin = np.array(blist_lin)
    rsqlist_lin = np.array(rsqlist_lin)
    rsqlist_lin_model = np.array(rsqlist_lin_model)
    
    # Global pooled fit (linear space)
    x_all = mean_data.flatten()
    y_all = var_data.flatten()
    mask_all = (x_all > 0) & (y_all > 0) & np.isfinite(x_all) & np.isfinite(y_all)
    
    try:
        params_glob, _ = curve_fit(axb_noslope, x_all[mask_all], y_all[mask_all],
                                   maxfev=1_000_000, p0=[1, 0.001])
        a_glob_lin, b_glob_lin = params_glob
    except:
        a_glob_lin, b_glob_lin = np.nan, np.nan
    
    # Global linear fit (y = mx + b)
    if mask_all.sum() >= 2:
        slope_glob, intercept_glob, _, _, _ = linregress(x_all[mask_all], y_all[mask_all])
    else:
        slope_glob, intercept_glob = np.nan, np.nan
    
    # Per-neuron R² using global parameters
    pooled_rsq_lin, pooled_rsq_lin_model = [], []
    
    for neuron in range(n_neurons):
        x = mean_data[..., neuron].astype(float)
        y = var_data[..., neuron].astype(float)
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        
        if mask.sum() < 2:
            pooled_rsq_lin.append(np.nan)
            pooled_rsq_lin_model.append(np.nan)
        else:
            x_m, y_m = x[mask], y[mask]
            
            # Power law R²
            y_hat = axb_noslope(x_m, a_glob_lin, b_glob_lin)
            ss_res = np.sum((y_m - y_hat)**2)
            ss_tot = np.sum((y_m - np.mean(y_m))**2)
            pooled_rsq_lin.append(1 - ss_res/ss_tot if ss_tot > 0 else np.nan)
            
            # Linear model R²
            y_hat_lin = slope_glob * x_m + intercept_glob
            ss_res_lin = np.sum((y_m - y_hat_lin)**2)
            pooled_rsq_lin_model.append(1 - ss_res_lin/ss_tot if ss_tot > 0 else np.nan)
    
    pooled_rsq_lin = np.array(pooled_rsq_lin)
    pooled_rsq_lin_model = np.array(pooled_rsq_lin_model)
    
    return {
        'alist_lin': alist_lin,
        'blist_lin': blist_lin,
        'rsqlist_lin': rsqlist_lin,
        'rsqlist_lin_model': rsqlist_lin_model,
        'a_glob_lin': a_glob_lin,
        'b_glob_lin': b_glob_lin,
        'slope_glob': slope_glob,
        'intercept_glob': intercept_glob,
        'pooled_rsq_lin': pooled_rsq_lin,
        'pooled_rsq_lin_model': pooled_rsq_lin_model
    }


def fit_pixel_data(mean_data, var_data, a_glob=None, b_glob=None):
    """
    Fit power law to pixel-level data.
    
    Returns:
    --------
    dict with keys:
        - params: (a, b) parameters from power law fit
        - x_fit: X values for plotting the fit
        - y_fit: Y values for plotting the fit
        - x_pooled: X values for plotting pooled regression (if a_glob, b_glob provided)
        - y_pooled: Y values for plotting pooled regression (if a_glob, b_glob provided)
    """
    mask = (mean_data > 0) & (var_data > 0) & np.isfinite(mean_data) & np.isfinite(var_data)
    
    if mask.sum() < 2:
        return {
            'params': None,
            'x_fit': None,
            'y_fit': None,
            'x_pooled': None,
            'y_pooled': None
        }
    
    # Fit power law
    params, _ = curve_fit(power_law, mean_data[mask], var_data[mask],
                         maxfev=1_000_000, p0=[1, 2])
    x_fit = np.linspace(mean_data[mask].min(), mean_data[mask].max(), 100)
    y_fit = power_law(x_fit, *params)
    
    # Pooled regression if parameters provided
    if a_glob is not None and b_glob is not None and np.isfinite(a_glob) and np.isfinite(b_glob):
        x_pooled = np.linspace(mean_data[mask].min(), mean_data[mask].max(), 100)
        y_pooled = power_law(x_pooled, a_glob, b_glob)
    else:
        x_pooled, y_pooled = None, None
    
    return {
        'params': params,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'x_pooled': x_pooled,
        'y_pooled': y_pooled
    }


# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================

def plot_figure1(figure_name, stochbin_meanv1, stochbin_varv1, sigma10_meanv1, sigma10_varv1,
                 gnoise_mean, gnoise_var, bnoise_mean, bnoise_var,
                 fg_mean_relu, fg_var_relu, fb_mean_relu, fb_var_relu,
                 region_labels_df=None, region_label=1, ve_thresh=0.1, 
                 neuron_idx=2, savefig=False,
                 plot_parameters = {"Bern_spike_limit":100, "Gaus_spike_limit":5,
                                    "Bern_log_limit":(10e-6,10e2), "Gaus_log_limit":(10e-7,10e2),
                                    "Kernel_b_limit":(-1,6), "Kernel_r2_ind_xlim":0, "Kernel_r2_all_xlim":0, "Kernel_r2_all_ylim":7}):
    """
    Create Figure 1 with all subpanels.
    
    Parameters:
    -----------
    stochbin_meanv1 : ndarray
        Mean spike counts for stochastic binary noise (neuron-level), shape (num_images, num_neurons)
    stochbin_varv1 : ndarray
        Variance in spike counts for stochastic binary noise (neuron-level), shape (num_images, num_neurons)
    sigma10_meanv1 : ndarray
        Mean spike counts for Gaussian noise (neuron-level), shape (num_images, num_neurons)
    sigma10_varv1 : ndarray
        Variance in spike counts for Gaussian noise (neuron-level), shape (num_images, num_neurons)
    gnoise_mean : ndarray
        Mean pixel values for Gaussian noise (pixel-level)
    gnoise_var : ndarray
        Variance in pixel values for Gaussian noise (pixel-level)
    bnoise_mean : ndarray
        Mean pixel values for Bernoulli noise (pixel-level)
    bnoise_var : ndarray
        Variance in pixel values for Bernoulli noise (pixel-level)
    fg_mean_relu : ndarray
        Mean pixel values for Gaussian noise with ReLU (pixel-level)
    fg_var_relu : ndarray
        Variance in pixel values for Gaussian noise with ReLU (pixel-level)
    fb_mean_relu : ndarray
        Mean pixel values for Bernoulli noise with ReLU (pixel-level)
    fb_var_relu : ndarray
        Variance in pixel values for Bernoulli noise with ReLU (pixel-level)
    region_label : ndarray, optional
        Label array for stochastic binary noise (needed if filter_v1=True)
    ve_thresh : float, default=0.1
        Variance explained threshold
    neuron_idx : int, default=2
        Index of neuron to highlight (Neuron 3 = index 2)
    savefig : bool, default=False
        Whether to save the figure as 'figure1.png'
    filter_v1 : bool, default=True
        Whether to filter data to V1 region only
    """
    # Apply brain region filtering if requested
    region_name_map = {1: 'V1', 2: 'LM', 3: 'AL', 4: 'RL'}
    if region_labels_df is not None:
        region_name = region_name_map.get(region_label, f"Region {region_label}")
        print(f"  Filtering to {region_name} region...")
        v1_filtered = filter_x_region(region_label, stochbin_meanv1, stochbin_varv1, region_labels_df,
            sigma10_meanv1, sigma10_varv1
        )
        stochbin_meanv1 = v1_filtered['v1mean_stochbin']
        stochbin_varv1 = v1_filtered['v1var_stochbin']
        sigma10_meanv1 = v1_filtered['v1mean_sigma10']
        sigma10_varv1 = v1_filtered['v1var_sigma10']
        print(f"  Filtered to {stochbin_meanv1.shape[1]} neurons in {region_name}")
    
    # Print Notice
    region_name = region_name_map.get(region_label, f"Region {region_label}")
    print(f"\nProcessing {region_name}\n-------------------\n"
        f"Shape of mean arrays: Binarization {stochbin_meanv1.shape}, Gaussian {sigma10_meanv1.shape}\n"
          f"Shape of variance arrays: Binarization {stochbin_varv1.shape}, Gaussian {sigma10_varv1.shape}")

    # Step 1: Compute all neuron-level fits
    print("  Computing neuron-level fits...")

    stochbin_fits = fit_neuron_data(
        stochbin_meanv1, stochbin_varv1,
        a_initial=2.5e-3, b_initial=2.23
    )
    sigma10_fits = fit_neuron_data(
        sigma10_meanv1, sigma10_varv1,
        a_initial=2.2e-4, b_initial=2.24
    )
    
    # Extract fit results
    stochbin_rsqlist_lin = stochbin_fits['rsqlist_lin']
    sigma10_rsqlist_lin = sigma10_fits['rsqlist_lin']
    stochbin_rsqlist_lin_model = stochbin_fits['rsqlist_lin_model']
    sigma10_rsqlist_lin_model = sigma10_fits['rsqlist_lin_model']
    pooled_stochbin_lin = stochbin_fits['pooled_rsq_lin']
    pooled_sigma10_lin = sigma10_fits['pooled_rsq_lin']
    pooled_stochbin_lin_model = stochbin_fits['pooled_rsq_lin_model']
    pooled_sigma10_lin_model = sigma10_fits['pooled_rsq_lin_model']
    stochbin_blist_lin = stochbin_fits['blist_lin']
    sigma10_blist_lin = sigma10_fits['blist_lin']
    a_glob_st_lin = stochbin_fits['a_glob_lin']
    b_glob_st_lin = stochbin_fits['b_glob_lin']
    a_glob_sg_lin = sigma10_fits['a_glob_lin']
    b_glob_sg_lin = sigma10_fits['b_glob_lin']
    
    # Step 2: Compute all pixel-level fits
    print("  Computing pixel-level fits...")
    bernoulli_fits = fit_pixel_data(
        bnoise_mean, bnoise_var,
        a_glob_st_lin, b_glob_st_lin
    )
    bernoulli_relu_fits = fit_pixel_data(
        fb_mean_relu, fb_var_relu,
        a_glob_st_lin, b_glob_st_lin
    )
    gaussian_fits = fit_pixel_data(
        gnoise_mean, gnoise_var,
        a_glob_sg_lin, b_glob_sg_lin
    )
    gaussian_relu_fits = fit_pixel_data(
        fg_mean_relu, fg_var_relu,
        a_glob_sg_lin, b_glob_sg_lin
    )
    
    # Extract pixel fit results
    params_b_pixel = bernoulli_fits['params']
    params_br_pixel = bernoulli_relu_fits['params']
    params_g_pixel = gaussian_fits['params']
    params_gr_pixel = gaussian_relu_fits['params']
    x_fit_b, y_fit_b = bernoulli_fits['x_fit'], bernoulli_fits['y_fit']
    x_fit_br, y_fit_br = bernoulli_relu_fits['x_fit'], bernoulli_relu_fits['y_fit']
    x_fit_g, y_fit_g = gaussian_fits['x_fit'], gaussian_fits['y_fit']
    x_fit_gr, y_fit_gr = gaussian_relu_fits['x_fit'], gaussian_relu_fits['y_fit']
    x_pooled_b, y_pooled_b = bernoulli_fits['x_pooled'], bernoulli_fits['y_pooled']
    x_pooled_br, y_pooled_br = bernoulli_relu_fits['x_pooled'], bernoulli_relu_fits['y_pooled']
    x_pooled_g, y_pooled_g = gaussian_fits['x_pooled'], gaussian_fits['y_pooled']
    x_pooled_gr, y_pooled_gr = gaussian_relu_fits['x_pooled'], gaussian_relu_fits['y_pooled']
    
    # Helper functions
    def safe_mask(x, y):
        """Return masked x and y arrays for valid (positive, finite) values"""
        xa = np.asarray(x).astype(float).flatten()
        ya = np.asarray(y).astype(float).flatten()
        mask = (xa > 0) & (ya > 0) & np.isfinite(xa) & np.isfinite(ya)
        return xa[mask], ya[mask]
    
    def axb_noslope(x, a, b):
        """Power law function: y = a * x^b"""
        return a * (x ** b)
    
    def smooth_from_logfit(x, log_intercept, b):
        """Convert log-space fit to linear space"""
        a = np.exp(log_intercept)
        return a * (x ** b)
    
    def fit_log_power(x, y):
        """Fit power law in log-log space with zero intercept"""
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return 1.0, np.nan, np.nan
        lx = np.log(x[mask])
        ly = np.log(y[mask])
        model = LinearRegression(fit_intercept=True)
        model.fit(lx.reshape(-1, 1), ly)
        b = model.coef_[0]
        return 1.0, b, np.nan
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
    
    # ==============================================================================
    # TOP LEFT: Selected Neuron (Neuron 3) - 2x2 nested grid
    # ==============================================================================
    inner_tl = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0], wspace=0.3, hspace=0.3)
    
    # Data for Neuron 3
    xg, yg = safe_mask(sigma10_meanv1[:, neuron_idx], sigma10_varv1[:, neuron_idx])
    xb, yb = safe_mask(stochbin_meanv1[:, neuron_idx], stochbin_varv1[:, neuron_idx])
    
    # Fits for Gaussian
    if xg.size > 1:
        logxg = np.log(xg)
        logyg = np.log(yg)
        if np.isfinite(logxg).all() and np.isfinite(logyg).all():
            model_g = LinearRegression(fit_intercept=True)
            model_g.fit(logxg.reshape(-1, 1), logyg)
            class LogFitResult:
                def __init__(self, slope, intercept):
                    self.slope = slope
                    self.intercept = intercept
            res_g = LogFitResult(model_g.coef_[0], model_g.intercept_)
            xg_fit_log = np.exp(np.linspace(np.log(xg.min()), np.log(xg.max()), 100))
            yg_fit_log = smooth_from_logfit(xg_fit_log, res_g.intercept, res_g.slope)
        else:
            res_g = None
            xg_fit_log, yg_fit_log = [], []
        
        params_g, _ = curve_fit(axb_noslope, xg, yg, maxfev=1_000_000, p0=[1, 0.001])
        xg_fit_lin = np.linspace(xg.min(), xg.max(), 100)
        yg_fit_lin = smooth_from_logfit(xg_fit_lin, np.log(params_g[0]), params_g[1])
    else:
        res_g = None
        xg_fit_log, yg_fit_log = [], []
        xg_fit_lin, yg_fit_lin = [], []
        params_g = [np.nan, np.nan]
    
    # Fits for Bernoulli
    if xb.size > 1:
        logxb = np.log(xb)
        logyb = np.log(yb)
        if np.isfinite(logxb).all() and np.isfinite(logyb).all():
            model_b = LinearRegression(fit_intercept=True)
            model_b.fit(logxb.reshape(-1, 1), logyb)
            res_b = LogFitResult(model_b.coef_[0], model_b.intercept_)
            xb_fit_log = np.exp(np.linspace(np.log(xb.min()), np.log(xb.max()), 100))
            yb_fit_log = smooth_from_logfit(xb_fit_log, res_b.intercept, res_b.slope)
        else:
            res_b = None
            xb_fit_log, yb_fit_log = [], []
        
        params_b, _ = curve_fit(axb_noslope, xb, yb, maxfev=1_000_000, p0=[1, 0.001])
        xb_fit_lin = np.linspace(xb.min(), xb.max(), 100)
        yb_fit_lin = smooth_from_logfit(xb_fit_lin, np.log(params_b[0]), params_b[1])
    else:
        res_b = None
        xb_fit_log, yb_fit_log = [], []
        xb_fit_lin, yb_fit_lin = [], []
        params_b = [np.nan, np.nan]
    
    # TL Subplots
    # 1. Gaussian Linear
    ax = fig.add_subplot(inner_tl[0, 1])
    ax.scatter(xg, yg, s=10, label='Data', marker=".", alpha=0.5)
    if len(xg_fit_lin): ax.plot(xg_fit_lin, yg_fit_lin, "r-", label='Fit')
    ax.plot([0, xg.max()], [0, xg.max()], 'k--', alpha=0.5)
    ax.set_title("Gaussian Noise")
    if len(xg_fit_lin):
        ax.text(0.01, 0.99, f"a={params_g[0]:.1e}\nb={params_g[1]:.1f}", 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, plot_parameters["Gaus_spike_limit"])
    
    bbox = ax.get_position()
    fig.text(bbox.x1 - .17, bbox.y1 + 0.02, f"Select V1 Neuron\na · mean^b",
             ha='center', va='bottom', fontsize=14, fontweight='bold', transform=fig.transFigure)
    
    # 2. Gaussian Log
    ax = fig.add_subplot(inner_tl[1, 1])
    ax.loglog(xg, yg, '.', label='Data', alpha=0.1)
    if len(xg_fit_log): ax.loglog(xg_fit_log, yg_fit_log, 'r-', label='Fit')
    ax.plot([0, xg.max()], [0, xg.max()], 'k--', alpha=0.5)
    ax.set_xlabel('Mean Spike Count')
    if res_g:
        ax.text(0.01, 0.99, f"a={np.exp(res_g.intercept):.1e}\nb={res_g.slope:.1f}", 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(plot_parameters["Gaus_log_limit"][0], plot_parameters["Gaus_log_limit"][1])
    
    # 3. Bernoulli Linear
    ax = fig.add_subplot(inner_tl[0, 0])
    ax.scatter(xb, yb, s=10, color='tab:orange', label='Data', marker=".", alpha=0.5)
    if len(xb_fit_lin): ax.plot(xb_fit_lin, yb_fit_lin, 'r-', label='Fit')
    ax.plot([0, xb.max()], [0, xb.max()], 'k--', alpha=0.5)
    ax.set_ylabel('Variance in Spike Counts')
    ax.set_title("Bernoulli Noise")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, plot_parameters["Bern_spike_limit"])
    if len(xb_fit_lin):
        ax.text(0.01, 0.99, f"a={params_b[0]:.1e}\nb={params_b[1]:.1f}", 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=10)
    
    # 4. Bernoulli Log
    ax = fig.add_subplot(inner_tl[1, 0])
    ax.loglog(xb, yb, '.', color='tab:orange', label='Data', alpha=0.1)
    if len(xb_fit_log): ax.loglog(xb_fit_log, yb_fit_log, 'r-', label='Fit')
    ax.plot([0, xb.max()], [0, xb.max()], 'k--', alpha=0.5)
    ax.set_xlabel('Mean Spike Count')
    ax.set_ylabel("Variance in Spike Counts")
    if res_b:
        ax.text(0.01, 0.99, f"a={np.exp(res_b.intercept):.1e}\nb={res_b.slope:.1f}", 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(plot_parameters["Bern_log_limit"][0], plot_parameters["Bern_log_limit"][1])
    
    # ==============================================================================
    # TOP RIGHT: All Neurons - 2x2 nested grid
    # ==============================================================================
    inner_tr = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 1], wspace=0.3, hspace=0.3)
    
    # Data for All Neurons
    xg_all, yg_all = safe_mask(sigma10_meanv1, sigma10_varv1)
    xb_all, yb_all = safe_mask(stochbin_meanv1, stochbin_varv1)
    
    # Fits for All Gaussian
    if xg_all.size > 1:
        logxg_all = np.log(xg_all)
        logyg_all = np.log(yg_all)
        model_ga = LinearRegression(fit_intercept=True)
        model_ga.fit(logxg_all.reshape(-1, 1), logyg_all)
        y_pred_ga = model_ga.predict(logxg_all.reshape(-1, 1))
        ss_res_ga = np.sum((logyg_all - y_pred_ga)**2)
        ss_tot_ga = np.sum((logyg_all - np.mean(logyg_all))**2)
        r2_ga = np.clip(1 - ss_res_ga/ss_tot_ga, 0, 1) if ss_tot_ga > 0 else 0
        r_value_ga = np.sqrt(r2_ga)
        class LogFitResult:
            def __init__(self, slope, intercept, rvalue):
                self.slope = slope
                self.intercept = intercept
                self.rvalue = rvalue
        res_ga = LogFitResult(model_ga.coef_[0], model_ga.intercept_, r_value_ga)
        xg_fit_a = np.exp(np.linspace(np.log(xg_all.min()), np.log(xg_all.max()), 100))
        yg_fit_a = smooth_from_logfit(xg_fit_a, res_ga.intercept, res_ga.slope)
        try:
            params_ga_lin, _ = curve_fit(axb_noslope, xg_all, yg_all, maxfev=1_000_000, p0=[1, 0.001])
        except Exception:
            params_ga_lin = [np.nan, np.nan]
    else:
        res_ga = None
        xg_fit_a, yg_fit_a = [], []
        params_ga_lin = [np.nan, np.nan]
    
    # Fits for All Bernoulli
    if xb_all.size > 1:
        logxb_all = np.log(xb_all)
        logyb_all = np.log(yb_all)
        model_ba = LinearRegression(fit_intercept=True)
        model_ba.fit(logxb_all.reshape(-1, 1), logyb_all)
        y_pred_ba = model_ba.predict(logxb_all.reshape(-1, 1))
        ss_res_ba = np.sum((logyb_all - y_pred_ba)**2)
        ss_tot_ba = np.sum((logyb_all - np.mean(logyb_all))**2)
        r2_ba = np.clip(1 - ss_res_ba/ss_tot_ba, 0, 1) if ss_tot_ba > 0 else 0
        r_value_ba = np.sqrt(r2_ba)
        res_ba = LogFitResult(model_ba.coef_[0], model_ba.intercept_, r_value_ba)
        xb_fit_a = np.exp(np.linspace(np.log(xb_all.min()), np.log(xb_all.max()), 100))
        yb_fit_a = smooth_from_logfit(xb_fit_a, res_ba.intercept, res_ba.slope)
        try:
            params_ba_lin, _ = curve_fit(axb_noslope, xb_all, yb_all, maxfev=1_000_000, p0=[1, 0.001])
        except Exception:
            params_ba_lin = [np.nan, np.nan]
    else:
        res_ba = None
        xb_fit_a, yb_fit_a = [], []
        params_ba_lin = [np.nan, np.nan]
    
    # TR Subplots
    # 1. Gaussian Linear
    ax = fig.add_subplot(inner_tr[0, 1])
    ax.scatter(xg_all, yg_all, s=1, alpha=0.1)
    if len(xg_fit_a): ax.plot(xg_fit_a, yg_fit_a, 'r-')
    ax.plot([0, xg_all.max()], [0, xg_all.max()], 'k--', alpha=0.5)
    ax.set_title('Gaussian Noise')
    a_lin_g = params_ga_lin[0]
    b_lin_g = params_ga_lin[1]
    ax.text(0.01, 0.99, f"a={a_lin_g:.1e}\nb={b_lin_g:.1f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, plot_parameters["Gaus_spike_limit"])
    
    bbox = ax.get_position()
    fig.text(bbox.x1 - .17, bbox.y1 + 0.02, f"All Neurons\na · mean^b",
             ha='center', va='bottom', fontsize=14, fontweight='bold', transform=fig.transFigure)
    
    # 2. Gaussian Log
    ax = fig.add_subplot(inner_tr[1, 1])
    ax.loglog(xg_all, yg_all, '.', markersize=1, alpha=0.1)
    if len(xg_fit_a): ax.loglog(xg_fit_a, yg_fit_a, 'r-')
    ax.plot([0.1, 100], [0.1, 100], 'k--', alpha=0.5)
    ax.set_xlabel("Mean Spike Count")
    if res_ga:
        ax.text(0.01, 0.99, f"a={np.exp(res_ga.intercept):.1e}\nb={res_ga.slope:.1f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(plot_parameters["Gaus_log_limit"][0], plot_parameters["Gaus_log_limit"][1])
    
    # 3. Bernoulli Linear
    ax = fig.add_subplot(inner_tr[0, 0])
    ax.scatter(xb_all, yb_all, s=1, alpha=0.1, color='tab:orange')
    if len(xb_fit_a): ax.plot(xb_fit_a, yb_fit_a, 'r-')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5)
    ax.set_title('Bernoulli Noise')
    ax.set_ylabel('Variance in Spike Counts')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, plot_parameters["Bern_spike_limit"])
    a_lin_b = params_ba_lin[0]
    b_lin_b = params_ba_lin[1]
    ax.text(0.01, 0.99, f"a={a_lin_b:.1e}\nb={b_lin_b:.1f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=10)
    
    # 4. Bernoulli Log
    ax = fig.add_subplot(inner_tr[1, 0])
    ax.loglog(xb_all, yb_all, '.', markersize=1, alpha=0.1, color='tab:orange')
    if len(xb_fit_a): ax.loglog(xb_fit_a, yb_fit_a, 'r-')
    ax.plot([0.1, 100], [0.1, 100], 'k--', alpha=0.5)
    ax.set_xlabel("Mean Spike Count")
    ax.set_ylabel("Variance in Spike Counts")
    if res_ba:
        ax.text(0.01, 0.99, f"a={np.exp(res_ba.intercept):.1e}\nb={res_ba.slope:.1f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(plot_parameters["Bern_log_limit"][0], plot_parameters["Bern_log_limit"][1])
    
    # ==============================================================================
    # BOTTOM LEFT: KDE and Statistics - 2x2 nested grid
    # ==============================================================================
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1, 0], wspace=0.3, hspace=0.5)
    
    # Extract pooled parameters from log-space fits
    pooled_a_gaussian = np.exp(res_ga.intercept) if res_ga else np.nan
    pooled_b_gaussian = res_ga.slope if res_ga else np.nan
    pooled_a_bernoulli = np.exp(res_ba.intercept) if res_ba else np.nan
    pooled_b_bernoulli = res_ba.slope if res_ba else np.nan
    
    # Get indices of neurons with variance explained > threshold
    def above_threshold(r2_array, thresh=0.1):
        r2 = np.asarray(r2_array, float)
        mask = np.isfinite(r2) & (r2 > thresh)
        return np.where(mask)[0]
    
    ve_idx_gaus = above_threshold(sigma10_rsqlist_lin, ve_thresh)
    ve_idx_bern = above_threshold(stochbin_rsqlist_lin, ve_thresh)
    
    # Compute log-space b values for the plot
    n_neurons = stochbin_meanv1.shape[-1]
    stochbin_blist = np.array([fit_log_power(stochbin_meanv1[..., i], stochbin_varv1[..., i])[1] for i in range(n_neurons)])
    sigma10_blist = np.array([fit_log_power(sigma10_meanv1[..., i], sigma10_varv1[..., i])[1] for i in range(n_neurons)])
    
    # [0,0] - B PLOT (slope distribution)
    ax = fig.add_subplot(inner[0, 0])
    sns.kdeplot(stochbin_blist[ve_idx_bern], label="Log Log Space", linestyle="--", color="tab:orange", ax=ax)
    sns.kdeplot(sigma10_blist[ve_idx_gaus], ax=ax, linestyle="--")
    """
    commenting out loglog plots
    """
    sns.kdeplot(stochbin_blist_lin[ve_idx_bern], label="Spike Count Space", color="tab:orange", ax=ax)
    sns.kdeplot(sigma10_blist_lin[ve_idx_gaus], color="tab:blue", ax=ax)
    ax.set_xlim(plot_parameters["Kernel_b_limit"][0], plot_parameters["Kernel_b_limit"][1])
    ax.set_title("Exponent (b)")
    
    ylim = ax.get_ylim()
    y_pos = ylim[0] + 0.99 * (ylim[1] - ylim[0])
    
    if np.isfinite(pooled_b_bernoulli):
        ax.scatter(pooled_b_bernoulli, y_pos, marker="v", s=70, edgecolor="k", color="tab:orange")
    if np.isfinite(pooled_b_gaussian):
        ax.scatter(pooled_b_gaussian, y_pos, marker="v", s=70, edgecolor="k", color="C0")
    if np.isfinite(b_glob_st_lin):
        ax.scatter(b_glob_st_lin, y_pos, marker="v", s=70, edgecolor="k", color="tab:orange")
    if np.isfinite(b_glob_sg_lin):
        ax.scatter(b_glob_sg_lin, y_pos, marker="v", s=70, edgecolor="k", color="C0")
    
    # [1,0] - R² PLOT (individual fits)
    ax = fig.add_subplot(inner[1, 0])
    
    def clean(arr, clip01=True):
        v = np.asarray(arr, float)
        v = v[np.isfinite(v)]
        if clip01:
            v = np.clip(v, 0, 1)
        return v
    
    def kde_r2(ax, arr, label, color, ls="-"):
        v = clean(arr, clip01=True)
        if v.size < 3:
            ax.text(0.05, 0.9, f"{label}: n={v.size}", transform=ax.transAxes)
            return
        sns.kdeplot(v, ax=ax, label=label, color=color, linestyle=ls,
                    bw_adjust=0.35, cut=0, clip=(0, 1), common_norm=False)
    
    kde_r2(ax, stochbin_rsqlist_lin, "Bernoulli: power law (linear space)", "tab:orange", "-")
    kde_r2(ax, stochbin_rsqlist_lin_model, "Bernoulli: linear model (y=mx+b)", "tab:orange", ":")
    kde_r2(ax, sigma10_rsqlist_lin, "Gaussian: power law (linear space)", "tab:blue", "-")
    kde_r2(ax, sigma10_rsqlist_lin_model, "Gaussian: linear model (y=mx+b)", "tab:blue", ":")
    
    ax.axvline(x=ve_thresh, color="0.5", linestyle="dashdot")
    ax.set_title("Variance Explained\nIndividual Neuron Model")
    ax.set_xlim(plot_parameters["Kernel_r2_ind_xlim"], 1)

    
    # [1,1] - R² PLOT (pooled/all neurons)
    ax = fig.add_subplot(inner[1, 1])
    
    kde_r2(ax, pooled_stochbin_lin, "Bernoulli: power law pooled (linear space)", "tab:orange", "-")
    kde_r2(ax, pooled_stochbin_lin_model, "Bernoulli: linear model pooled (y=mx+b)", "tab:orange", ":")
    kde_r2(ax, pooled_sigma10_lin, "Gaussian: power law pooled (linear space)", "tab:blue", "-")
    kde_r2(ax, pooled_sigma10_lin_model, "Gaussian: linear model pooled (y=mx+b)", "tab:blue", ":")
    
    ax.axvline(x=ve_thresh, color="0.5", linestyle="dashdot")
    ax.set_title("Variance Explained\nAll Neuron Model")
    ax.set_xlim(plot_parameters["Kernel_r2_all_xlim"], 1)
    ax.set_ylim(0,plot_parameters["Kernel_r2_all_ylim"])
    
    # [0,1] - Legend
    ax = fig.add_subplot(inner[0, 1])
    ax.axis('off')
    legend_elements = [
        Line2D([0], [0], marker='v', color='k', label='1 Model for All Neurons',
               markerfacecolor='k', markersize=8, linestyle='None', markeredgecolor='k'),
        Line2D([0], [0], color='k', label='Log-log Space', linestyle='--'),
        Line2D([0], [0], color='k', label='Spike Count Space', linestyle='-'),
        Line2D([0], [0], color='k', label='Linear Fit', linestyle=':')
    ]
    ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=10)
    
    # ==============================================================================
    # BOTTOM RIGHT: Pixel Space Analysis - 2x2 nested grid
    # ==============================================================================
    inner_br = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1, 1], wspace=0.3, hspace=0.4)
    
    # [0,0] - Bernoulli Noise (pixel level)
    ax = fig.add_subplot(inner_br[0, 0])
    ax.scatter(bnoise_mean, bnoise_var, marker=".", alpha=0.006, s=1.001, color='tab:orange')
    if x_fit_b is not None:
        ax.plot(x_fit_b, y_fit_b, 'r-', linewidth=2, label=f'Fit: a={params_b_pixel[0]:.2e}, b={params_b_pixel[1]:.2f}')
        model = LinearRegression(fit_intercept=False).fit(bnoise_mean.reshape(-1,1), bnoise_var)
        y_pred_br = model.predict(bnoise_mean.reshape(-1,1))
        ax.plot(bnoise_mean, y_pred_br, color = "black", alpha=0.4)
    # if x_pooled_b is not None:
        #ax.plot(x_pooled_b, y_pooled_b, 'k--', linewidth=2, label=f'Pooled: a={a_glob_st_lin:.2e}, b={b_glob_st_lin:.2f}')
    # ax.text(0.01, 0.99, f"ff={model.coef_[0]:.1f}", transform=ax.transAxes, verticalalignment='top',
    #          horizontalalignment='left', fontsize=10)
    ax.set_ylabel("Variance in Pixel Value")
    ax.set_title("Bernoulli Noise")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
    # [1,0] - Bernoulli Noise ReLU (pixel level)
    ax = fig.add_subplot(inner_br[1, 0])
    ax.scatter(fb_mean_relu, fb_var_relu, marker=".", alpha=0.006, s=1.001, color='tab:orange')
    if x_fit_br is not None:
        ax.plot(x_fit_br, y_fit_br, 'r-', linewidth=2, label=f'Fit: a={params_br_pixel[0]:.2e}, b={params_br_pixel[1]:.2f}')
        model = LinearRegression(fit_intercept=False).fit(fb_mean_relu.reshape(-1,1), fb_var_relu)
        y_pred_br = model.predict(fb_mean_relu.reshape(-1,1))
        ax.plot(fb_mean_relu, y_pred_br, color = "black", alpha=0.4)
    # if x_pooled_br is not None:
        #ax.plot(x_pooled_br, y_pooled_br, 'k--', linewidth=2, label=f'Pooled: a={a_glob_st_lin:.2e}, b={b_glob_st_lin:.2f}')
    # ax.text(0.01, 0.99, f"ff={model.coef_[0]:.1f}", transform=ax.transAxes, verticalalignment='top',
    #          horizontalalignment='left', fontsize=10)
    ax.set_xlabel("Mean Pixel Value")
    ax.set_ylabel("Variance in Pixel Value")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_title("Bernoulli RELU")
    
    # [0,1] - Gaussian Noise (pixel level)
    ax = fig.add_subplot(inner_br[0, 1])
    ax.scatter(gnoise_mean, gnoise_var, marker=".", alpha=0.006, s=1.001, color='tab:blue')
    if x_fit_g is not None:
        ax.plot(x_fit_g, y_fit_g, 'r-', linewidth=2, label=f'Fit: a={params_g_pixel[0]:.2e}, b={params_g_pixel[1]:.2f}')
        model = LinearRegression(fit_intercept=False).fit(gnoise_mean.reshape(-1,1), gnoise_var)
        y_pred_br = model.predict(gnoise_mean.reshape(-1,1))
        ax.plot(gnoise_mean, y_pred_br, color = "black", alpha=0.4)
    # if x_pooled_g is not None:
        #ax.plot(x_pooled_g, y_pooled_g, 'k--', linewidth=2, label=f'Pooled: a={a_glob_sg_lin:.2e}, b={b_glob_sg_lin:.2f}')
    # ax.text(0.01, 0.99, f"ff={model.coef_[0]:.1f}", transform=ax.transAxes, verticalalignment='top',
    #          horizontalalignment='left', fontsize=10)
    ax.set_title("Gaussian Noise")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 6000)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    

    # [1,1] - Gaussian Noise ReLU (pixel level)
    ax = fig.add_subplot(inner_br[1, 1])
    ax.scatter(fg_mean_relu, fg_var_relu, marker=".", alpha=0.006, s=1.001, color='tab:blue')
    if x_fit_gr is not None:
        ax.plot(x_fit_gr, y_fit_gr, 'r-', linewidth=2, label=f'Fit: a={params_gr_pixel[0]:.2e}, b={params_gr_pixel[1]:.2f}')
        model = LinearRegression(fit_intercept=False).fit(fg_mean_relu.reshape(-1,1), fg_var_relu)
        y_pred_br = model.predict(fg_mean_relu.reshape(-1,1))
        ax.plot(fg_mean_relu, y_pred_br, color = "black", alpha=0.4)
    # if x_pooled_gr is not None:
        #ax.plot(x_pooled_gr, y_pooled_gr, 'k--', linewidth=2, label=f'Pooled: a={a_glob_sg_lin:.2e}, b={b_glob_sg_lin:.2f}')
    # ax.text(0.01, 0.99, f"ff={model.coef_[0]:.1f}", transform=ax.transAxes, verticalalignment='top',
    #          horizontalalignment='left', fontsize=10)
    ax.set_xlabel("Mean Pixel Value")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_title("Gaussian RELU")
    
    # Add title above bottom-right plot
    bbox = ax.get_position()
    fig.text(bbox.x1 - .17, bbox.y1+0.225, "Pixel Space Analysis",
             ha='center', va='bottom', fontsize=14, fontweight='bold', transform=fig.transFigure)
    
    if savefig: 
        plt.savefig(f"{figure_name}.png")
        plt.close()
    else: plt.show()
    return fig
