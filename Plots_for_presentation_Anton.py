%matplotlib qt
import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from scipy import linalg

from functions_Anton import *




############Start#################################
##load the data
wd = os.getcwd()
string_path = '/predictions/Anton/'