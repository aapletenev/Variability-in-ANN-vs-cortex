from numpy import full
from prediction import make_predictions

"""
Assumptions
1. scans.csv must be in same directory
2. microns_area_labels.csv is in same directory
"""

# note that image input must have shape (x, 144, 256) where x is the number of frames 
frames = full(shape = (5,144,256), fill_value = 128)

a, b, c, d = make_predictions('dynamic', frames, 15, [[4,7]], noise_seeds = 2, num_frames = 3)

print(f'shape of:\na: {a.shape}\nb: {b.shape}\nc: {c.shape}\nd(list): {len(d)}')