import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('MacOSX')

##load the data
wd = os.getcwd()
string_path = '/predictions/8-27-2025/'
#show all dirictories inside it
dirs = os.listdir(wd + '/' + string_path)

#Now load in the data
#create empty npy arrays to  append data
data = []
data_mean = []
var = []
labels = [] #this is the area labels
for dir in dirs:
    data.append(np.load(wd + '/' + string_path + dir + '/sum_arr_'+ dir + '.npy'))
    data_mean.append(np.load(wd + '/' + string_path + dir + '/mean_arr_'+ dir + '.npy'))
    labels.append(np.load(wd + '/' + string_path + dir + '/label_arr_'+ dir + '.npy'))
    #variance across noise levels
    var.append(np.var(np.load(wd + '/' + string_path + dir + '/sum_arr_'+ dir + '.npy'), axis = 1))

data = np.concatenate(data, axis = 0)
data_mean = np.concatenate(data_mean, axis = 0)
labels = np.concatenate(labels, axis = 0)
var = np.concatenate(var, axis = 0)
#data is now of shape (num_images, num_noise, num_neurons), labels is of shape (num_images, num_neurons), I need to select neurons with area == 1
data_V1 = data[:,:,labels[0,:] == 1]
data_mean_V1 = data_mean[:,labels[0,:] == 1]
var_V1 = var[:,labels[0,:] == 1]

#now I need the regression var vs mean, the function is a*mean^b, so i need to cllect a anad and R^2 for regression log(var) = log(a) + b*log(mean)
from scipy.stats import linregress
a_list = []
b_list = []
r2_list = []
for neuron in range(data_V1.shape[2]):
    mean_vals = data_mean_V1[:,neuron]
    var_vals = var_V1[:,neuron]
    #remove zeros
    mask = (mean_vals > 0) & (var_vals > 0)
    mean_vals = mean_vals[mask]
    var_vals = var_vals[mask]
    log_mean = np.log(mean_vals)
    log_var = np.log(var_vals)
    slope, intercept, r_value, p_value, std_err = linregress(log_mean, log_var)
    a = np.exp(intercept)
    b = slope
    r2 = r_value**2
    a_list.append(a)
    b_list.append(b)
    r2_list.append(r2)

#convert to numpy arrays
a_array = np.array(a_list)
b_array = np.array(b_list)
r2_array = np.array(r2_list)

#plot a vs b scatter plot, be aware that there are 5000 points so make them small and transparent
plt.figure(figsize=(10, 6))
plt.scatter(b_array, a_array, s=1, alpha=0.5)
plt.xlabel('b parameter', fontsize=16)
plt.ylabel('a parameter', fontsize=16)
plt.title('Bernoulli noise', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#add the regression to the plot as a black line with confidence interval
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(b_array, a_array)
x_vals = np.array([np.min(b_array), np.max(b_array)])
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='black', label='Regression line')


#save the figure in pdf
plt.savefig(wd + '/plots/Anton/a_vs_b_scatter_plot_bernulli.pdf')
#png
plt.savefig(wd + '/plots/Anton/a_vs_b_scatter_plot_bernulli.png', dpi=300)



