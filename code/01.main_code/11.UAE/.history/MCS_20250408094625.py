# region: Import libraries
import os
import subprocess
import shlex
import concurrent.futures as con_fu
import time
import datetime as dt
import warnings
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable
import json
import h5py
import multiprocessing

# Data manipulation and analysis
import pandas as pd
import numpy as np
# import scipy.stats as sst
# from scipy.optimize import curve_fit, root_scalar
# from scipy.stats import genextreme as gev
# from scipy.signal import detrend
# from scipy.interpolate import CubicSpline
# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.utils import check_random_state
import xarray as xr
import numpy as np
# from sklearn.cluster import KMeans
from sklearn_som.som import SOM
from kneed import KneeLocator
# endregion

# region: Import libraries
import os
import subprocess
import shlex
import concurrent.futures as con_fu
import time
import datetime as dt
import warnings
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable
import json
import h5py
import multiprocessing

# Data manipulation and analysis
import pandas as pd
import numpy as np
import geocat.comp as gc
# import scipy.stats as sst
# from scipy.optimize import curve_fit, root_scalar
# from scipy.stats import genextreme as gev
# from scipy.signal import detrend
# from scipy.interpolate import CubicSpline
# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.utils import check_random_state
import xarray as xr
import numpy as np
# from sklearn.cluster import KMeans

from my_junk import gradient_ds, find_name
# endregion

# --------------- * Calculate the water vapour mass * -------------- #
def water_vapour_mass_relative(T, RH, P):
    """
    Calculate the water vapour mass in the air.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    RH : float
        Relative humidity in percentage.
    P : float
        Atmospheric pressure in Pa.

    Returns
    -------
    float
        Water vapour mass in the air in kg/m^3.

    """
    # Constants
    Mw = 18.01528e-3  # kg/mol
    R = 8.314462618  # J/(mol.K)

    # Calculate the saturation vapour pressure
    Tc = T - 273.15
        # This use like a linear approximation of the Clausius-Clapeyron equation
    Ps = 611.21 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))

    # Calculate the water vapour mass
        # Eq: RH = Pv / Ps
        # SH = 0.622 * Pv / (P - Pv)
        
    Pv = RH / 100 * Ps
    rho_v = Pv * Mw / (R * T)
    return rho_v

def water_vapour_mass_absolute(
    SH: Union[np.ndarray, xr.DataArray],
    U: Union[np.ndarray, xr.DataArray],
    V: Union[np.ndarray, xr.DataArray],
    sP: Union[np.ndarray, xr.DataArray]
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate vertically intergrated water vapour mass in the air.

    Parameters
    ----------
    SH : float
        Specific humidity in percentage.
    P : float
        Atmospheric pressure in Pa.

    Returns
    -------
    float
        Water vapour mass in the air in kg/m^3.
        
    """
    
    #? MFC = -u\frac{dq}{dx} - v\frac{dq}{dy} - q\left(\frac{du}{dx} + \frac{dv}{dy}\right)
    
    g = 9.80665 #m/s^2
    # Vapour mass from each layer
        # Delta pressure between layers
        #? Negative since pressure decreases with height
    
    P_name = find_name(SH, 'level')
    P = SH[P_name]
    # Calculate pressure gradient based on surface pressure and pressure levels
    dp = gc.meteorology.delta_pressure(P, sP)
    
    layer_mass = dp/g
    vapour_mass = SH * layer_mass
    int_vapour_mass = vapour_mass * (U**2 + V**2)**0.5
    integrated_vapour_mass = np.sum(vapour_mass, axis=0)
    
    
    
    
    
    
