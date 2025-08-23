import os
import pandas as pd
import xarray as xr
import json
import pickle
from functools import lru_cache
import numpy as np

# ------------------------ check indices ----------------------- #

def check_non_exist(name, force=False):
    return False if os.path.exists(name) else True
def check_exist(name):
    return True if os.path.exists(name) else False

class DataCache:
    _cache_dir = ".data_cache"  # Default cache directory

    @classmethod
    def set_cache_dir(cls, directory):
        """Set the cache directory"""
        cls._cache_dir = directory
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_data(file_path, type_cache='xarray', chunks = None):
        """
        Load data from file and cache it.
        Type can be 'xarray', 'pandas', 'pickle', or 'json'
        """
        if type_cache == 'pandas' or file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif type_cache == 'xarray' or file_path.endswith('.nc'):
            return xr.decode_cf(xr.open_dataset(file_path), chunks = chunks) if chunks else xr.open_dataset(file_path)
        elif type_cache == 'pickle' or file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif type_cache == 'json' or file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported type_cache: {type_cache}")

    @classmethod
    def save_data(cls, data, file_path, type_cache=None):
        """
        Save data to file based on data type or specified format.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Determine data type if not specified
        if type_cache is None:
            if isinstance(data, pd.DataFrame):
                type_cache = 'pandas'
            elif isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
                type_cache = 'xarray'
            elif isinstance(data, (dict, list)):
                type_cache = 'json'
            else:
                type_cache = 'pickle'

        # Save based on type
        if type_cache == 'pandas' or file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif type_cache == 'xarray' or file_path.endswith('.nc'):
            data.to_netcdf(file_path)
        elif type_cache == 'pickle' or file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif type_cache == 'json' or file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:
            raise ValueError(f"Unsupported type_cache: {type_cache}")

        # Clear cache for this file to ensure fresh data on next load
        cls.get_data.cache_clear()

        return file_path

def testprint(*args, **kwargs):
    """
    A simple print function that can be used for debugging.
    It prints the arguments passed to it.
    """
    print("testprint:", *args, **kwargs)
