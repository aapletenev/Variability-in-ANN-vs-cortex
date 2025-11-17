from noise_seeds import generate_noise, stochastic_binarization
import numpy as np
import matplotlib.pyplot as plt

frames = np.full((50, 144, 256), fill_value = 128) # say we have 50 images

"""
use generate_noise to get sigma10 noise
"""

sigma10 = generate_noise("dynamic", frames.shape[0], 10)
sigma10_frames = frames + sigma10

"""
get bernoulli noise
"""
bernoulli_frames = stochastic_binarization(frames)


# mean var scatter plot and compute mean var
plt.figure(figsize = (10, 6))
# sigma10
x = np.mean(sigma10_frames, axis = 0).flatten()
y = np.var(sigma10_frames, axis = 0).flatten()
plt.subplot(121)
plt.scatter(x, y, marker = ".", alpha = 0.3)
plt.title("sigma=10")
plt.xlabel("Mean Pixel Value")
plt.ylabel("Variance in Pixel Value")

# bernoulli
x = np.mean(bernoulli_frames, axis = 0).flatten()
y = np.var(bernoulli_frames, axis = 0).flatten()
plt.subplot(122)
plt.scatter(x, y, marker = ".", alpha = 0.3)
plt.xlabel("Mean Pixel Value")
plt.title("bernoulli")
plt.tight_layout()
plt.show()