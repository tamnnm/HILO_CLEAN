import numpy as np
import xarray as xr
import pandas as pd
import os
from datetime import datetime
from typing import Union, Optional, List, Tuple, Dict, Literal, Any
from shapely.geometry import Point
import geopandas as gpd
import subprocess
from my_junk import find_name

<<<<<<< HEAD
def gradient_ds(ds: xr.Dataset, dim: str ) -> xr.Dataset:
    # Find the name of the coordinate
    coord_name = find_name(ds, dim)
    
    # Gradient of the dataset
    ds_grad = ds.diff(dim=coord_name)
    return ds_grad
=======
def gradient_ds(ds: Union[xr.DataArray, np.ndarray], dim: str ) -> Union[xr.DataArray, np.ndarray]:
    
    if isinstance(ds, np.ndarray):
        if dim == 'lon':
            dim = 0
        elif dim == 'lat':
            dim = 1
        elif dim == 'lev':
            dim = 2
        elif dim == 'time':
            dim = 3
        else:
            raise ValueError('dim must be lon, lat, lev or time')
        return np.gradient(ds, axis=dim)
    else:
        # Find the name of the coordinate
        coord_name = find_name(ds, dim)
        
        # Gradient of the dataset
        ds_grad = ds.diff(dim=coord_name)
        return ds_grad
>>>>>>> c80f4457 (First commit)
