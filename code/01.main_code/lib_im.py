from ast import Continue, Pass
from re import I
from selectors import EpollSelector
from tkinter import ttk
from cf_units import decode_time
from matplotlib.font_manager import ttfFontProperty
#from matplotlib.lines import _LineStyle
import pandas as pd
import matplotlib
import numpy as np
import pandas as pd
import os
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas import read_csv
import csv
import xarray as xr
from matplotlib.image import imread
import numpy as np
import pandas as pd

params = {
    'axes.titlesize' :25,
	'axes.labelsize': 25,
	'font.size': 20,
    'font.family':'serif',
	'legend.fontsize': 20,
    'legend.loc': 'upper right',
    'legend.labelspacing':0.25,
	'xtick.labelsize': 20,
	'ytick.labelsize': 20,
	'lines.linewidth': 3,
	'text.usetex': False,
	# 'figure.autolayout': True,
	'ytick.right': True,
	'xtick.top': True,

	'figure.figsize': [12, 10], # instead of 4.5, 4.5
	'axes.linewidth': 1.5,

	'xtick.major.size': 15,
	'ytick.major.size': 15,
	'xtick.minor.size': 5,
	'ytick.minor.size': 5,

	'xtick.major.width': 5,
	'ytick.major.width': 5,
	'xtick.minor.width': 3,
	'ytick.minor.width': 3,

	'xtick.major.pad': 10,
	'ytick.major.pad': 12,
	#'xtick.minor.pad': 14,
	#'ytick.minor.pad': 14,

	'xtick.direction': 'inout',
	'ytick.direction': 'inout',
   }
plt.clf()
matplotlib.rcParams.update(params)
