import numpy as np
from numpy import full
from prediction import make_predictions

"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""

# note that image input must have shape (x, 144, 256) where x is the number of frames 
frames = full(shape = (5,144,256), fill_value = 128)

a, b, c, d = make_predictions('dynamic', frames, 3, [[4,7]], noise_seeds = 2, num_frames = 3)

ano, bno, cno, dno = make_predictions('no noise', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3)

print('dynamic to no noise: ', np.unique(a == ano, return_counts = True))
print(f'max of dynamic: {np.max(a):.3f}; max of no noise: {np.max(ano):.3f}\n')

a, b, c, d = make_predictions('dynamic', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3, stochastic_bin_param = True)

ano, bno, cno, dno = make_predictions('constant', frames, 100, [[4,7]], noise_seeds = 2, num_frames = 3, stochastic_bin_param = True)

print('two stoch bin: ', np.unique(a == ano, return_counts = True))
print(f'max of first: {np.max(a):.3f}; max of no second" {np.max(ano):.3f}')