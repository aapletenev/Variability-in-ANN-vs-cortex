import numpy as np
from numpy import full
from prediction import make_predictions
from visualizations import line_plot_mean, spike_plot_4x2, sixteen_tuning_curves, mean_var_scatter_4x4,  mean_var_scatter_4x4_regression, violin_4row, violin_combined
import os
"""
File Description
----------------
The file makes example predictions with different parameters and then proceeds to pull other predictions from memory. The new batch of 
predictions is visualized.

Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""

# note that image input must have shape (x, 144, 256) where x is the number of frames 
frames = full(shape = (5,144,256), fill_value = 128) #this is the example of 5 frames

a, b, c, d = make_predictions('dynamic', frames, 3, [[4,7]], noise_seeds = 2, num_frames = 3)

ano, bno, cno, dno = make_predictions('no noise', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3)

print('dynamic to no noise: ', np.unique(a == ano, return_counts = True))
print(f'max of dynamic: {np.max(a):.3f}; max of no noise: {np.max(ano):.3f}\n')

a, b, c, d = make_predictions('dynamic', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3, stochastic_bin_param = True)

ano, bno, cno, dno = make_predictions('constant', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3, stochastic_bin_param = True)

print('two stochastic binarization: ', np.unique(a == ano, return_counts = True))
print(f'max of first: {np.max(a):.3f}; max of no second" {np.max(ano):.3f}')

# check august1.ipynb for examples for all plots if errors persist

#---loading arrays from data used for presentation---#
# in real code we know d3_labels from np.unique() on region label output from predictions
labels_lst = [1,2,3,4]

string_path = 'fnn//input_noise//saved_inputs_outputs//predictions_july21//'

for noise in ['c3', 'c15', 'c30', 'd3', 'd15', 'd30', 'dbin', 'cbin', 'no_noise']:
    for type_noise in ['_sum', '_mean', '_var', '_labels']:
        file_name = string_path + noise + type_noise + '.npy'
        arr_name = noise + type_noise
        globals()[arr_name] = np.load(file_name)

# load predictions by region
for r in labels_lst:
    for noise in ['c3', 'c15', 'c30', 'd3', 'd15', 'd30', 'dbin', 'cbin', 'no_noise']:
        for type_noise in ['_sum', '_mean', '_var']:
            file_name = string_path + noise + type_noise + '_r' + str(r) + '.npy'
            arr_name = noise + type_noise + '_r' + str(r)
            globals()[arr_name] = np.load(file_name)

#---defining useful arrays---#
dmeans1 = [[d3_mean_r1, d15_mean_r1, d30_mean_r1, dbin_mean_r1],
                 [d3_mean_r2, d15_mean_r2, d30_mean_r2, dbin_mean_r2],
                 [d3_mean_r3, d15_mean_r3, d30_mean_r3, dbin_mean_r3],
                 [d3_mean_r4, d15_mean_r4, d30_mean_r4, dbin_mean_r4]]
dvars1 = [[d3_var_r1, d15_var_r1, d30_var_r1, dbin_var_r1],
               [d3_var_r2, d15_var_r2, d30_var_r2, dbin_var_r2],
               [d3_var_r3, d15_var_r3, d30_var_r3, dbin_var_r3],
               [d3_var_r4, d15_var_r4, d30_var_r4, dbin_var_r4]]

dmeans2 = [[d15_mean_r1, d30_mean_r1, dbin_mean_r1],
                 [d15_mean_r2, d30_mean_r2, dbin_mean_r2],
                 [d15_mean_r3, d30_mean_r3, dbin_mean_r3],
                 [d15_mean_r4, d30_mean_r4, dbin_mean_r4]]
dvars2 = [[d15_var_r1, d30_var_r1, dbin_var_r1],
               [d15_var_r2, d30_var_r2, dbin_var_r2],
               [d15_var_r3, d30_var_r3, dbin_var_r3],
               [d15_var_r4, d30_var_r4, dbin_var_r4]]

cmeans1 = [[c3_mean_r1, c15_mean_r1, c30_mean_r1, cbin_mean_r1],
                 [c3_mean_r2, c15_mean_r2, c30_mean_r2, cbin_mean_r2],
                 [c3_mean_r3, c15_mean_r3, c30_mean_r3, cbin_mean_r3],
                 [c3_mean_r4, c15_mean_r4, c30_mean_r4, cbin_mean_r4]]
cvars1 = [[c3_var_r1, c15_var_r1, c30_var_r1, cbin_var_r1],
               [c3_var_r2, c15_var_r2, c30_var_r2, cbin_var_r2],
               [c3_var_r3, c15_var_r3, c30_var_r3, cbin_var_r3],
               [c3_var_r4, c15_var_r4, c30_var_r4, cbin_var_r4]]

#---plots---# 

# spike plot
dmeans_spike = [[d3_mean_r1, d15_mean_r1, d30_mean_r1],
                 [d3_mean_r2, d15_mean_r2, d30_mean_r2],
                 [d3_mean_r3, d15_mean_r3, d30_mean_r3],
                 [d3_mean_r4, d15_mean_r4, d30_mean_r4]]


cmeans_spike = [[c3_mean_r1, c15_mean_r1, c30_mean_r1],
                 [c3_mean_r2, c15_mean_r2, c30_mean_r2],
                 [c3_mean_r3, c15_mean_r3, c30_mean_r3],
                 [c3_mean_r4, c15_mean_r4, c30_mean_r4]]
spike_plot_4x2('Dynamic Noise', dmeans_spike, cmeans_spike, normalized = False)

# tuning curves
ctuning_arrs = [[c3_sum_r1, c15_sum_r1, c30_sum_r1, cbin_sum_r1],
 [c3_sum_r2, c15_sum_r2, c30_sum_r2, cbin_sum_r2],
 [c3_sum_r3, c15_sum_r3, c30_sum_r3, cbin_sum_r3],
 [c3_sum_r4, c15_sum_r4, c30_sum_r4, cbin_sum_r4]]

dtuning_arrs = [[d3_sum_r1, d15_sum_r1, d30_sum_r1, dbin_sum_r1],
 [d3_sum_r2, d15_sum_r2, d30_sum_r2, dbin_sum_r2],
 [d3_sum_r3, d15_sum_r3, d30_sum_r3, dbin_sum_r3],
 [d3_sum_r4, d15_sum_r4, d30_sum_r4, dbin_sum_r4]]
titles = ['Sigma=3', 'Sigma=15', 'Sigma=30', 'Stochastic Binarization']

sixteen_tuning_curves(arrays = dtuning_arrs, main_title =  'Dynamic Noise',titles = titles, savefig = False )
titles = ['Sigma=3', 'Sigma=15', 'Sigma=30', 'Stochastic Binarization']

sixteen_tuning_curves(arrays = ctuning_arrs, main_title =  'Constant Noise',titles = titles, savefig = False )

# mean var scatter
mean_var_scatter_4x4(means = cmeans1,vars = cvars1,main_title = 'Dynamic Noise (Top 1% of Responses Removed)', 
                     savefig = False)

# mean var reg. 
# random neurons by region from last prediction 

# region 1
r1_randoms = [2322, 3279, 4060, 1756,  146, 4441, 3978, 2034, 5025, 3515, 3534,
       2167, 2310, 1837, 3864, 2529, 4127, 2325, 4791,  314, 3371, 1857,
       1095, 1917, 3097, 1722, 4344, 3627, 3930, 3459]
# region 2
r2_randoms = [958, 907, 514, 432, 594, 385, 184, 368, 525, 351, 302, 551, 550,
       575, 375, 719, 242, 613, 789, 872, 340, 402, 397, 643, 255, 928,
       471, 894, 281, 523]
# region 3
r3_randoms = [279, 337, 208, 128,  46, 115, 248, 197,  52, 269,  37, 364, 340,
       203,  36, 153,  84, 278,  71, 115,  24,  78, 138,  38,  22,  17,
        89, 349, 178, 332]
# region 4
r4_randoms = [376, 295, 517,  25, 663, 429, 427, 651, 133, 500, 689, 266, 470,
       424,  90, 560, 290, 198, 149, 181, 749, 461, 658, 771, 367, 758,
       125, 195, 191,   8]
randoms = [r1_randoms, r2_randoms, r3_randoms, r4_randoms]
mean_var_scatter_4x4_regression(means = cmeans1,vars = cvars1, neurons = randoms, main_title = 'Dynamic Noise (Top 1% of Responses Removed)', 
                     savefig = False)

#### violin plot 1
violin_4row(means = dmeans2, vars = dvars2, main_title = 'Dynamic Noise', savefig = False)

#### violin plot 2
violin_combined(dmeans1, dvars1, 'Dynamic Noise', savefig = False)
violin_combined(dmeans1, dvars1, 'Dynamic Noise', savefig = True)