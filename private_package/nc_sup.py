import numpy as np
import xarray as xr
import pandas as pd
import os
from datetime import datetime
from typing import Union, Optional, List, Tuple, Dict, Literal, Any
from shapely.geometry import Point
import geopandas as gpd
import subprocess
<<<<<<< HEAD
=======
from functools import lru_cache
>>>>>>> c80f4457 (First commit)

coord_name = {"level":('level','bottom_top','isobaricInhPa','z'),
                "time":('time','Time','ISO'),
                "lat":('lat','latitude','XLAT','south_north','y'),
                "lon":('lon','longitude','XLONG','west_east','x')}
<<<<<<< HEAD
var_name = {'tmp':('tmp','air','temp','t2m','var167'),
            'tmx':('tmx','tmax'),
            'tmn':('tmin','tmn'),
            'precip':('precip','apcp','tp','prec','var228'),
=======
var_name = {'tmp':('tmp','air','temp','t2m','var167', 'T2m', 'Tm', 'Tn', 'Tx'),
            'tmx':('tmx','tmax'),
            'tmn':('tmin','tmn'),
            'precip':('precip','apcp','tp','prec','var228', 'R'),
>>>>>>> c80f4457 (First commit)
            'rain':('twcr',),
            'shum':('shum','q','sh'),
            'rhum':('rhum','rh'),
            'z':('z','hgt','gh'),
            'msl':('msl','prmsl'),
<<<<<<< HEAD
            'uwnd':('uwnd','u','u10','u850'),
=======
            'uwnd':('uwnd','u','u10','u850', 'Um'),
>>>>>>> c80f4457 (First commit)
            'vwnd':('vwnd','v','v10','v850'),
            'land':('land', 'lsm'),
            'tsoil':('tsoil','soilt'),
            'soilm':('soilm','msoil','soilw','wsoil'),
<<<<<<< HEAD
            'pres':('pres','psfc','pres.sfc'),
=======
            'pres':('pres','dp','psfc','pres.sfc'),
>>>>>>> c80f4457 (First commit)
            'sst':('sst','ts','skt'),}
wrf_name = ('avo','eth','cape_2d','cape_3d','ctt','cloudfrac','dbz','mdbz',
            'geopt','geopt_stag','helicity','lat','lon','omg','p/pres','pressure',
            'pvo','pw','rh','rh2','slp','T2','ter','td2','td','tc','th/theta','temp','tk','times',
            'xtimes','tv','twb','updraft_helicity','ua','va','wa','uvmet10','uvmet','wspd_wdir','wspd_wdir10','uvmet_wspd_wdir','uvmet10_wspd_wdir','z','height_agl','zstag','Variable Name','mcape','mcin','lcl','lfc','cape3d_only','cin3d_only','low_cloudfrac','mid_cloudfrac','high_cloudfrac','uvmet_wspd','uvmet_wdir','uvmet10_wspd','uvmet10_wdir','wspd','wdir','wspd10','wdir10'
)
formats = {"3": ('%Y-%m-%d',),
           "2": ('%Y-%m', '%m-%d'),
            "1": ('%Y', '%m')}


def print_list():
    for key, value in dict.items():
        print(f"{key}: {value}")
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)

def get_alias_path(alias_name):
    result = subprocess.run(['bash', '-i', '-c', 'alias'], capture_output=True, text=True)
    alias_output = result.stdout
    alias_line = None
    for line in alias_output.splitlines():
        if line.startswith(f'alias {alias_name}='):
            alias_line = line
            break
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
    if alias_line:
        alias_path = alias_line.split('=')[1].strip("'\"")
        return alias_path
    else:
        return None

def get_exp_path(exp_path):
    return os.environ.get(exp_path)

var_tuple = ()
for value in var_name.values():
    var_tuple += value
coord_tuple = ()
for value in coord_name.values():
    coord_tuple += value

#for key in coord_name:
#    coord_name[key] = Literal(coord_name[key])
#for key in var_name:
#    var_name[key] = Literal(var_name[key])


# THIS VERSION HAS NOT BEEN TESTED
def func_resample(ds,freq: str, method: Literal["sum","mean"] = "sum", closed="left", offset=None):
    # Define the resampling function based on the aggregation function
    resample_params = {'time': freq}
    if closed is not None:
        resample_params['closed'] = closed
    if offset is not None:
        resample_params['offset'] = offset

    method_funcs = {
    "sum": ds.sum,
    "mean": ds.mean
}

    func = method_funcs[method]
    # Resample the dataset
    try:
        ds.resample(**resample_params).reduce(func, dim="time")
    except:
        resample_params['Time'] = resample_params.pop("time")
        ds.resample(**resample_params).reduce(func, dim="Time")
    return ds

def shift_time(ds, time_period=None, hour_period=None, output=None):
    # To test using loffset
    try:
        ds=ds.resample(loffset=f'{time_period}{hour_period}')
        print("loofset does work")
        return ds

    # Convert the time coordinate to a pandas.DatetimeIndex
    except:
        if isinstance(ds, xr.Dataset):
            for name in coord_name['time']:
                if name in ds.coords:
                    time_name = name
                    break
            df_time=ds[time_name].to_pandas()
            df=ds.to_pandas()
        else:
            df=ds
            df_time=ds.index

        # Shift the pandas.DatetimeIndex
        if time_period and hour_period:
            df_time = df_time + pd.Timedelta(f'P{time_period}T{hour_period}')

        if output == "pd":
            try:
                df = df.set_index(df_time)
            except:
                #Time is already the index column
                df=df
            df_time = df_time.to_pydatetime()
            return(df, df_time)
        else:
            ## Convert the pandas.DatetimeIndex back to an xarray.DataArray
            if isinstance(ds, xr.Dataset):
            # Update the `time` coordinate in the xarray.Dataset
                ds = ds.assign_coords({time_name: df_time})
                # ds has time as a coordinate and dimension so no need for this
                #ds = ds.set_index({time_name: time_name})
                #ds = ds.sortby(time_name)
                return ds
def find_name(dts, fname = None, fname_group = None, opt = "coords"):
    """
        dts: xarray dataset
        fname: name of the variable
<<<<<<< HEAD
        fname_group: group of the variable
        opt: option to search for the name in the coordinates or dimensions]
        return: name of the variable
    """
    
=======
        fname_group: group of the variable (can be custom as list or tuple)
        ==> If not, use the automatic defined lookup variables
        opt: option to search for the name in the coordinates or dimensions]
        ==> opt can be "coords", "dims", "var"
        return: name of the variable
    """

>>>>>>> c80f4457 (First commit)
    list_name = coord_name
    list_tuple = coord_tuple

    if [fname,fname_group] == [None,None]:
        raise ValueError("Must provide at least fname or fname_group to search for its name")
    if opt == "coords":
        list_dts = list(dts.coords)
    elif opt == "dims":
        list_dts = list(dts.dims)
<<<<<<< HEAD
    elif opt == "var":
=======
    elif "var" in opt:
        if isinstance(dts, xr.DataArray):
            fname = dts.name
            return fname
>>>>>>> c80f4457 (First commit)
        list_dts = list(dts.variables)
        list_name = var_name
        list_tuple = var_tuple

    if fname in list(list_dts):
        return fname
    else:
<<<<<<< HEAD
        fname_group = fname_group if fname_group else fname
        # check other possibilities in the group name
        if fname_group in list_name:
=======
        # If fname_group is provided, use it, otherwise use fname
        fname_group = fname_group if fname_group else fname
        # check other possibilities in the group name
        if isinstance(fname_group, tuple) or isinstance(fname_group, list):
            list_name_group = fname_group
        # check in the dict of var or dict of coords
        elif fname_group in list_name:
>>>>>>> c80f4457 (First commit)
            list_name_group = list_name[fname_group]
        # check in the tuple of var or tuple of coords
        else:
            list_name_group = list_tuple
<<<<<<< HEAD
        return next((name for name in list_name_group if name in list_dts), None)
=======
        fname =  next((name for name in list_name_group if name in list_dts), None)

        if fname is None:
            raise ValueError(f"Check the name of this dimension {fname_group}.\n \
                            Please check the list of available variable are: {list_name_group}")
        return fname
>>>>>>> c80f4457 (First commit)
        """
        result = next((name for name in list_name_group if name in list(list_name)), None)
        if result == "south_north":
            result = "XLAT"
        elif result == "west_east":
            result = "XLONG"
        return result
        """
#WHERE IS SLOWWWW (10 TIMES) SO USE IT CAREFULLY
# wanted_data = wanted_data.where(wanted_data.values, drop=True)
#MODULE to cut a single variable
<<<<<<< HEAD
def cut_var(dts, var=None, val_only: bool = False):
=======
def cut_var(dts, var=None, val_only: bool = False, name_val = False):
>>>>>>> c80f4457 (First commit)
    """_summary_

    Args:
        dts (_type_): original datast
        var (_type_, optional): Name of the variable. Defaults to None.
        val_only (bool, optional): Return option. Defaults to False: return all dataset. True: return only values.

    """
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
    if var is None:
        dts_final = dts
    else:
        # List of variables from the var_name
        # Try to find the name in each group
        var_final = find_name(dts, fname=var, opt = "var")
        # If not found, find in other groups
        if var_final is None:
            if var in var_tuple:
                var_final = var
            else:
                raise ValueError(f"Check the name of this dimension {var}.\n \
                                Please check the list of available variable are: t2m, temp, tmax,\
                                tmin, precip, shum, rhum, z, msl")
        dts_var = dts[var_final]
<<<<<<< HEAD
        print(f"Opened {var} as name: {var_final}")

    if val_only:
        dts_final = dts_var.values
    else:
        dts_final = dts_var
    return dts_final, var_final
=======

    if val_only:
        dts_final = dts_var.to_numpy()
    else:
        dts_final = dts_var

    return (dts_final, var_final) if name_val else dts_final
>>>>>>> c80f4457 (First commit)

#MODULE to cut dataset according to grid and time

def cut_co_exist(dts, lat_dts, lon_dts, name = [None, None], full: bool = False, data: bool = True):
<<<<<<< HEAD
=======
    #TODO: Fix this following /work/users/tamnnm/code/01.main_code/03.indices_model_eva/ME_prob.py
>>>>>>> c80f4457 (First commit)
    result = {}
    if name[0] is None:
        lat_co_name = find_name(dts, fname_group='lat', opt = "coords")
    elif name[0] in dts.coords:
        lat_co_name = name[0]
    else:
        lat_co_name = find_name(dts, name[0], fname_group='lat', opt = "coords")
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
    if name[1] is None:
        lon_co_name = find_name(dts, fname_group='lon', opt = "coords")
    elif name[1] in dts.coords:
        lon_co_name = name[1]
    else:
        lon_co_name = find_name(dts, name[1], fname_group='lon', opt = "coords")
<<<<<<< HEAD
    
    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")
    
    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}
    
=======

    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")

    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}

>>>>>>> c80f4457 (First commit)
    if (data or full) is True:
        coords = True
        try:
            lat_co = dts[lat_co_name]
            lon_co = dts[lon_co_name]
        except KeyError:
            lat_co = dts.coords[lat_co_name]
            lon_co = dts.coords[lon_co_name]
<<<<<<< HEAD
        
        if lat_co.isnull().any() or lon_co.isnull().any():
            print("Check the missing values in the coordinate \n Please do not use cut_co")
            return
        
=======

        if lat_co.isnull().any() or lon_co.isnull().any():
            print("Check the missing values in the coordinate \n Please do not use cut_co")
            return

>>>>>>> c80f4457 (First commit)
        lat_check = False
        lat_cond= lat_co == lat_dts
        if (lat_cond).any():
            lat_check = True
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
        lon_check = False
        lon_cond= lon_co == lon_dts
        if (lon_cond).any():
            lon_check = True
<<<<<<< HEAD
            
=======

>>>>>>> c80f4457 (First commit)
        # Condition for lat and lon
        if lat_check and lon_check:
            geo_cond = (lat_cond & lon_cond)
        elif lat_check:
            geo_cond = lat_cond
        elif lon_check:
            geo_cond = lon_cond
        else:
            geo_cond = None
<<<<<<< HEAD
        
        if geo_cond is None:
            raise ValueError("Check the range of latitude and longitude")
        
=======

        if geo_cond is None:
            raise ValueError("Check the range of latitude and longitude")

>>>>>>> c80f4457 (First commit)
        # Apply for one-dimensional coordinate
        # It will always be quicker or more efficient (sel > where)
        if len(lat_co.shape) == 1 and len(lon_co.shape) == 1:
            lat_data = lat_dts if lat_check else lat_co
            lon_data = lon_dts if lon_check else lon_co
            dts_new=dts.sel({lat_co_name: lat_data, lon_co_name: lon_data})
        # Apply for multi-dimensional coordinate (wrf-output)
        else:
            if data is True:
                lat_data = lat_co.where(lat_cond, drop=True) if lat_check is not None else lat_co
                lon_data = lon_co.where(lon_cond, drop=True) if lon_check is not None else lon_co
            if full is True:
                dts_new=dts.where(geo_cond,drop=True) if geo_cond is not None else dts

    if data and full is False:
        return {'lat': lat_data, 'lon': lon_data}
    if full and data is False:
        return dts_new
    else:
        return {'full': dts_new, 'data': {'lat': lat_data, 'lon': lon_data}}
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)

# ! For TC track, don't use this one
# ! The lat,lon is not filled but has nan values depends on each TC
def cut_co(dts, ulat, dlat, ulon, dlon, name = [None, None], full: bool = True, data: bool = False):
    result = {}
    if name[0] is None:
        lat_co_name = find_name(dts, fname_group='lat', opt = "coords")
    elif name[0] in dts.coords:
        lat_co_name = name[0]
    else:
        lat_co_name = find_name(dts, name[0], fname_group='lat', opt = "coords")
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
    if name[1] is None:
        lon_co_name = find_name(dts, fname_group='lon', opt = "coords")
    elif name[1] in dts.coords:
        lon_co_name = name[1]
    else:
        lon_co_name = find_name(dts, name[1], fname_group='lon', opt = "coords")
<<<<<<< HEAD
    
    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")
    
    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}
    
=======

    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")

    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}

>>>>>>> c80f4457 (First commit)
    if (data or full) is True:
        coords = True
        try:
            lat_co = dts[lat_co_name]
            lon_co = dts[lon_co_name]
        except KeyError:
            lat_co = dts.coords[lat_co_name]
            lon_co = dts.coords[lon_co_name]
<<<<<<< HEAD
        
=======

>>>>>>> c80f4457 (First commit)
        if lat_co.isnull().any() or lon_co.isnull().any():
            print("Check the missing values in the coordinate \n Please do not use cut_co")
            return
        lat_check = False
        if [ulat,dlat].count(None) < 2:
            dlat = dlat if dlat is not None else min(lat_co.values)
            ulat = ulat if ulat is not None else max(lat_co.values)
            lat_cond=(lat_co >= dlat) & (lat_co <= ulat)
            if (lat_cond).any():
                lat_check = True
<<<<<<< HEAD
        
=======

>>>>>>> c80f4457 (First commit)
        lon_check = False
        if [ulon,dlon].count(None) < 2:
            dlon = dlon if dlon is not None else min(lon_co.values)
            ulon = ulon if ulon is not None else max(lon_co.values)
            lon_cond=(lon_co >= dlon) & (lon_co <= ulon)
            if (lon_cond).any():
                lon_check = True
<<<<<<< HEAD
                
=======

>>>>>>> c80f4457 (First commit)
        if [ulat, dlat, ulon, dlon].count(None) == 4:
            dts_new = dts
        else:
            # Condition for lat and lon
            if lat_check and lon_check:
                geo_cond = (lat_cond & lon_cond)
            elif lat_check:
                geo_cond = lat_cond
            elif lon_check:
                geo_cond = lon_cond
            else:
                geo_cond = None
<<<<<<< HEAD
            
            if geo_cond is None:
                raise ValueError("Check the range of latitude and longitude")
            
=======

            if geo_cond is None:
                raise ValueError("Check the range of latitude and longitude")

>>>>>>> c80f4457 (First commit)
            # Apply for one-dimensional coordinate
            # It will always be quicker or more efficient (sel > where)
            if len(lat_co.shape) == 1 and len(lon_co.shape) == 1:
                lat_data = lat_co[lat_cond] if lat_check else lat_co
                lon_data = lon_co[lon_cond] if lon_check else lon_co
                dts_new=dts.sel({lat_co_name: lat_data, lon_co_name: lon_data})
            # Apply for multi-dimensional coordinate (wrf-output)
            else:
                if data is True:
                    lat_data = lat_co.where(lat_cond, drop=True) if lat_check is not None else lat_co
                    lon_data = lon_co.where(lon_cond, drop=True) if lon_check is not None else lon_co
                if full is True:
                    dts_new=dts.where(geo_cond,drop=True) if geo_cond is not None else dts

    if data and full is False:
        return {'lat': lat_data, 'lon': lon_data}
    if full and data is False:
        return dts_new
    else:
        return {'full': dts_new, 'data': {'lat': lat_data, 'lon': lon_data}}
<<<<<<< HEAD
    
    # if lat does not change, name of lat will be None but the data is full coords. Same for lon
    # if it's WRF, lat and lon data will be lat_condition and lon_condition
    
# cut the datset with list of points
def cut_points(dts, points, name = [None, None], full: bool = True, data: bool = False):
    result = {}
    if name:
        lat_co_name = find_name(dts, name[0], fname_group='lat', opt = "coords")
        lon_co_name = find_name(dts, name[1], fname_group='lon', opt = "coords")
=======

    # if lat does not change, name of lat will be None but the data is full coords. Same for lon
    # if it's WRF, lat and lon data will be lat_condition and lon_condition

# cut the datset with list of points
def cut_points(dts, points,name = [None, None], full: bool = True, data: bool = False):
    result = {}

    lat_co_name = find_name(dts, name[0], fname_group='lat', opt = "coords")
    lon_co_name = find_name(dts, name[1], fname_group='lon', opt = "coords")

>>>>>>> c80f4457 (First commit)
    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")

    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}
    lat_co = dts.coords[lat_co_name]
    lon_co = dts.coords[lon_co_name]
    lat_data=lat_co[lat_co.isin([point[0] for point in points])]
    lon_data = lon_co[lon_co.isin([point[1] for point in points])]
    # lat_data = xr.concat(lat_list, dim='points')
    # lon_data = xr.concat(lon_list, dim='points')
    dts_new = dts.sel({lat_co_name: lat_data, lon_co_name: lon_data})

    if data and full is False:
        return {'lat': lat_data, 'lon': lon_data}
    if full and data is False:
        return dts_new
    else:
        return {'full': dts_new, 'data': {'lat': lat_data, 'lon': lon_data}}
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)

# cut the time coordinate according to certain range
def cut_time(dts, ss: Optional[Literal["DJF","MAM","JJA","SON"]] = None, name = None,
             start_date: str = None, end_date: str = None, run_end: bool= False, full: bool = False, data: bool = True) :
    """ opt 1: choose to only cut time co or cut for one DataArray
        opt 2: choose the define time range"""
    result = {}
    time_name_co = find_name(dts, name, fname_group='time', opt = "coords")
    if time_name_co:
        result['name'] = time_name_co
    else:
        raise KeyError(f"Check the name of this dimension {name}.\n")
    time_co = dts.coords[time_name_co]

    # return formatted date, length of date, and format
    def date_indentify(value):
        if value is None:
            return None, None
        date_split = value.split("-")
        format = formats[f'{len(date_split)}'][0] if int(date_split[0]) > 12 else formats[f'{len(date_split)}'][1]
            #date_time = date_split
        return len(date_split), format

    # define before hand
    non_start = start_date is None
    non_end = end_date is None
    len_end, format_end = date_indentify(end_date)
    len_start, format_start = date_indentify(start_date)
    print(format_end)
    # SYNTAX
    # time_co[cond] can only use if having .dt.month, dt.year, dt.day
    # time is the only coordinate/ variable that DON'T USE WHERE

    # region: date condition
    # if end_start doesn't have year, raise error
    if format_end != formats[f'{len_end}'][0]:
        raise ValueError("End date is invalid")
    # if both start_date and end_date are None, return the whole dataset
    elif non_start and non_end:
        pass
    # if start doesn't have year, specify the month + day through all years
    # we ignore the possibility to slice certain month and day through all years
    # we ignore end_date
    elif format_start != formats[f'{len_start}'][0]:
        date_start = [(ind) for ind in start_date.split("-")]
        cond = (time_co.dt.month == date_start[0]) if len_start == 1 else ((time_co.dt.month == date_start[0]) & (time_co.dt.day == date_start[1]))
        time_co = time_co[cond]
    # run_end is True, return time with one limit OR full slice of start and end date
    # if time is limit at one end, return the dataset with the other condition
    else:
        if non_start ^ non_end and run_end is not None:
            date = start_date if non_end else end_date
            start_date, end_date = date, date
            print("Only start date is selected")
        time_co = time_co.sel({time_name_co: slice(start_date, end_date)})
    #endregion

    # region: season condition
    # if we have season, it must range more than a month
    if ss is not None and format_start == formats[f'{len_start}'][0]:
        print(ss)
        fin_yr_start = time_co.values[0][:4]
        fin_yr_end = time_co.values[-1][:4]
        # Check the year range is more than one year
        if ss == "DJF" and fin_yr_start < fin_yr_end:
            # Take only from December of the last year and February of the last year
            cond_ss_1 = ((time_co.dt.month <= 2) & (time_co.dt.year > fin_yr_start))
            cond_ss_2 = ((time_co.dt.month == 12) & (time_co.dt.year < fin_yr_end))
            cond_ss = (cond_ss_1 | cond_ss_2)
        elif ss not in ['MAM','JJA','SON']:
            raise ValueError(
                "Invalid season, change option to MAM, JJA, SON")
        else:
            cond_ss =(time_co.dt.season == ss)
        time_co = time_co[cond_ss]
    # endregion

    if data and full is False:
        return time_co
    if full and data is False:
        return dts.sel({time_name_co: time_co})
    else:
        return {'full': dts.sel({time_name_co: time_co}), 'data': time_co}

# level doesn't need to extract, this to directly cut the dataset
def cut_level(dts, level = 0, opt = "e", reverse = False, data: bool = False, full: bool = True, name = None):
    # level: usual case
    # bottom_top: wrfout -> It's in dimension
    # isobaricInhPa: era5
    result = {}
    if level == 0 and reverse is False:
        raise ValueError("Please specify the level")
    level_name_co = find_name(dts, name, fname_group='level', opt = "dims")
    if level_name_co:
        result['name'] = level_name_co
    else:
        raise KeyError(f"Check the name of this dimension {name}.\n" )
    level_co= dts.coords[level_name_co]
    if data:
        if level == 0 :
            level_co = level_co
        else:
            if opt == "x":
                level_co = level_co[(level_co != level)] if level else level_co
            if opt == "e":
                level_co = level_co[(level_co == level)] if level else level_co
            if opt == "gt":
                level_co = level_co[(level_co >= level)] if level else level_co
            if opt == "lt":
                level_co = level_co[(level_co <= level)] if level else level_co
<<<<<<< HEAD
            
        result['data'] = level_co[::-1] if reverse else level_co
        
=======

        result['data'] = level_co[::-1] if reverse else level_co

>>>>>>> c80f4457 (First commit)
    if full:
        if level == 0:
            temp_result = dts
        else:
            if opt == "e":
                temp_result = dts.sel({level_name_co: level})
            else:
                dbound = slice(level, None)
                ubound = slice (None, level)
                if opt == "gt":
                    bound = ubound if level_co[0] > level_co[1] else dbound
                if opt == "lt":
                    bound = dbound if level_co[0] > level_co[1] else ubound
                temp_result = dts.sel({level_name_co: bound})
        result['full'] = temp_result.isel({level_name_co: slice(None,None,-1)}).squeeze() if reverse else temp_result.squeeze()

    if full and data: return result
    if data: return result['data']
    if full: return result['full']
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)

# calculate mean along certain dimension
# this is kinda trash, please use cut_mlp instead
def calculate_mean(final_dts, dim):
    result = {}
    if dim not in list(final_dts.dims):
        dim = next((dim_name for dim_name in coord_name[dim] if dim_name in list(final_dts.dims)), None)
    if dim is None: raise KeyError(f"Dimension {dim} is not in the dataset")
    result['name'] = dim
    result['full'] = final_dts.mean(dim=dim)
    return result

<<<<<<< HEAD
def cut_mlp(dts, time_dict=None, geo_dict=None, level_val: Optional[int] = None, dim_mean: list = None, dim_plot: list =  None, var_plot: str =  None):
=======
def cut_mlp(dts, time_dict=None, geo_dict=None, level_val: Optional[int] = None, dim_mean: list = None, dim_plot: list =  None, var: str =  None, val_only: bool = False):
>>>>>>> c80f4457 (First commit)
    #print(
    #"Be careful with the dimension of axis you are choosing\n\
    #you should check the order of the dimenions\n\
    #e.g. level,time,lat,lon then after cut lat, lon you should choose ax=0 (mean along time element)\n\
    #It's going to return the mean at a certain level")

    # collect name for each dimension to use in mean
    co_name = {}

    if time_dict:
        co_name['time'] = time_dict['name']
        if time_dict['full']:
            dts = time_dict['full']
        else:
            time_co_name = time_dict['name']
            time_data = time_dict['data']
            dts = dts.sel(**{time_co_name: time_data})
<<<<<<< HEAD
=======

>>>>>>> c80f4457 (First commit)
    if geo_dict:
        co_name['lat'] = geo_dict['name']['lat']
        co_name['lon'] = geo_dict['name']['lon']
        geo_co_dict = {}
        if geo_dict['full']:
            dts = geo_dict['full']
        else:
            lat_co_name = geo_dict['name']['lat']
            lon_co_name = geo_dict['name']['lon']
            if len(geo_dict['data']) == 1:
                geo_cond = geo_dict['data']
                dts = dts.where(geo_cond)
            else:
                if geo_dict['data']['lat']:
                    lat_data = geo_dict['data']['lat']
                    geo_co_dict[lat_co_name] = lat_data
                if geo_dict['data']['lon']:
                    lon_data = geo_dict['data']['lon']
                    geo_co_dict[lon_co_name] = lon_data
                dts = dts.sel(**geo_co_dict)
            # WRF or any case that Lat, Lon has more than one dimension
            #Lat, Lon here is boolen object - condition
                #print(lat_data,lon_data)
    if level_val:
        level_dict= cut_level(dts, level=level_val)
        dts = level_dict['full']
        co_name['level'] = level_dict['name']

    # choose ax=0 in this case to mean along the time and return the average on a certain level
    if dim_mean:
        #print("Dimension must be a list")
        list_dim_mean = []
        if "lat" not in co_name:
            co_name['lat'] = find_name(dts, fname = 'lat', opt = "dims")
        if "lon" not in co_name:
            co_name['lon'] = find_name(dts, fname = 'lon', opt = "dims")
        # if co_name['lat] = "south_north": co_name['lat']="XLAT"
        # if co_name['lon] = "west_east": co_name['lon']="XLONG"

        if "all" in dim_mean:
<<<<<<< HEAD
            dts_mean = dts.mean()
=======
            dts= dts.mean()
>>>>>>> c80f4457 (First commit)
        else:
            for dimension in dim_mean:
                list_dim_ind = None
                if dimension in list(dts.dims):
                    list_dim_ind = [dimension]
                if dimension == "geo":
                    list_dim_ind = [co_name['lat'], co_name['lon']]
                elif dimension in co_name:
                    list_dim_ind = [co_name[dimension]]
                else:
                    list_dim_ind = find_name(dts, name=dimension, opt = "dims")
                if list_dim_ind is None:
                    raise KeyError(f"Dimension {dimension} is not in the dataset")
                list_dim_mean += list_dim_ind
            dts = dts.mean(dim=list_dim_mean)
            ##print(final_dts_mean_anchor)

    # Please choose 2 coordinates and 1 variable to make plot
    # To plot, we need to mean all other coordinations
    if dim_plot:
<<<<<<< HEAD
        attrs = dts.attrs
        vars_dts_name= list(dts.variables)
        # if the input dts is already the Dataarray or Dataset with one variables
        if var_plot is None:
            if len(vars_dts_name >= 2):
                raise("Please choose the variable you want to plot")
            else:
                var_plot = vars_dts_name[0]

        # Actual name of the var_plot
        dts, var_plot_name = cut_var(dts, var=var_plot, val_only=True)

=======
>>>>>>> c80f4457 (First commit)
        # Call out the dim_plot
        list_dim_mean = list(dts.dims)
        if "geo" in dim_plot:
            dim_plot=['lat','lon']
        if len(dim_plot) != 2:
            raise("Please choose only 2 dimensions to plot")
        for dim in dim_plot:
            co_plot = find_name(dts, fname = dim, opt = "dims")
            if dim in list_dim_mean:
                list_dim_mean.remove(co_plot)
        dts = dts.mean(dim=list_dim_mean)
<<<<<<< HEAD
    return dts
=======

    if var:
        vars_dts_name= list(dts.variables)
        # if the input dts is already the Dataarray or Dataset with one variables

        if var not in vars_dts_name:
            var = find_name(dts, fname = var, opt = "vars")

        # Actual name of the var
        dts = cut_var(dts, var=var, val_only = val_only)

    return dts

>>>>>>> c80f4457 (First commit)
    # Don't be greedy.One variable at a time
    # Modified: 17/02/2024
    # Message: I haven't test for the case of WRF output.
        #     What will happens if its avg along XLAT, XLONG (XLAT, XLONG).\
        #     Will XLONG maintain the same value without XLAT and vice versa?

    """old_code

        for dimension in dim_plot:
            if dimension=='time':
                array_coords.append(time_data)
            elif dimension=="geo":
                #print(geo_data)
                array_coords = {}
                for geo_coord in geo_data:
                    array_coords[geo_coord.name] = geo_coord
                #Geo_data is tuple -> Must turn into array-like
                print("You already chosen 2 dimensions.")
                break
            elif dimension=="lon":
                array_coords.append(lon_data)
            elif dimension=="lat":
                array_coords.append(lat_data)
            elif dimension=="level":
                array_coords.append(level_data)

        if "DataArray" in str(type(final_dts_mean)):
                var=final_dts_mean.variable
                final_dts_plot = xr.DataArray(data=var, coords=array_coords, attrs=attrs).squeeze(drop=True)
        #This to filter out the variable only acting as index not real values
        else:
            list_var_name=list(final_dts_mean.variables)
            list_var=[]
            for i,var in enumerate(list_var_name):
                ##print(f"This dataset has this {var}")
                var = final_dts_mean.variables[f'{var}']
                    # wrap into a new dataarray to avoid multiple dimensions
                if len(var.dims)<=1:
                    continue
                else:
                    final_dts_plot = xr.DataArray(data=var, coords=array_coords, attrs=attrs)
                    list_var.append(var)
            if len(list_var) > 1:
                print(
    "Please check the number of variables in this case.\n\
    This cannot be used for contour plot since only 2-dimension dataarray is allowed.\n\
    After this step, be careful to seperate the variables for your use.")
                final_dts_plot = xr.Dataset(data_vars=list_var, coords=array_coords, attrs=attrs)
        return final_dts_plot
        """
def list_var(ds):
    coords_name = set(ds.coords)
    vars_name = set(ds.variables)

    # Check variable that is variable and not coords
    ds_var_list = []
    var_name_uniform_list = []
    for var_name in list(vars_name - coords_name):
        try:
<<<<<<< HEAD
            [ds_var, var_name_uniform] = cut_var(ds, var=var_name)
=======
            [ds_var, var_name_uniform] = cut_var(ds, var=var_name, name_val= True)
>>>>>>> c80f4457 (First commit)
            ds_var_list.append(ds_var)
            var_name_uniform_list.append(var_name_uniform)
        except Exception as e:
            print(e)
            continue
    if len(var_name_uniform_list) == 1:
        print("Only one variable is returned from grib file")
        return ds_var, var_name_uniform
    else:
        print('Return the list of available variables')
        return ds_var_list, var_name_uniform_list

"""
if __name__ == "__main__":
    os.chdir(cs.twcr_dir)
    dts_name="shum.mon.nc"
    var_name=dts_name.split(".")[0]
    dts_org=xr.open_dataset(dts_name)[var_name]
    #call the class
    dts_co = cut_co(dts_org,ulat=40,dlat=-15,ulon=180,dlon=80,opt=1)
    dts_co_t = cut_time(dts_co,opt1=1)
    if var_name in no_lv_dts: #this has no level
        dset_final=dts_co_t
    else:
        dset_final=np.mean(dts_co_t,axis=0)
 """
<<<<<<< HEAD
 
=======

>>>>>>> c80f4457 (First commit)
def save_nc(dts, ofile, dname=None):
    if isinstance(dts, xr.DataArray):
        dts.to_dataset(name=dname) \
            .to_netcdf(ofile,
                    encoding={dname: {'dtype': 'float32',
                                        'zlib': True, 'complevel': 1}},
                    unlimited_dims='time')
    else:
        if dname is None:
            dname = dts.name
            if dname is None:
                raise ValueError("dname must be provided or the DataArray must have a name")
        dts.to_netcdf(ofile,
                    encoding={dname: {'dtype': 'float32',
                                        'zlib': True, 'complevel': 1}},
                    unlimited_dims='time')
<<<<<<< HEAD
        
=======

>>>>>>> c80f4457 (First commit)

def ctime_short(data, start_date="1881-09-25", end_date="1881-10-15", R_time_name="time"):
    return data.sel({R_time_name: slice(start_date, end_date)})

# * Function for land mask
def land_filter(data, land_mask, opt):
    if opt == "land":
        return data.where(land_mask == 1)
    else:
        return data.where(land_mask == 0)
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)

def open_nc_file(name, sel_year=None):  # , land_mask=land_mask):
    # ! Squeeze unnecessary dimension (here is the level)
    # Load the data
    if sel_year is None:
        data = xr.open_dataset(f'{name}.nc')
    else:
        data = xr.open_dataset(f'{name}.{sel_year}.nc')
    var_name = find_name(data, fname_group=name.split('_')[0], opt='var')
    # Check if the data is a Dataset or DataArray
    if isinstance(data, xr.Dataset):
        # If it's a Dataset, select a specific variable
        # Replace 'variable_name' with the actual variable name
        DArray1 = data[var_name]
    elif isinstance(data, xr.DataArray):
        DArray1 = data
    else:
        raise TypeError(
            "The loaded data is neither a Dataset nor a DataArray.")
    return DArray1

def R_names(DArray):
    Dndim = DArray.ndim
    R_lat = find_name(DArray, fname="lat", opt='coords')
    R_lon = find_name(DArray, fname="lon", opt='coords')
    if Dndim > 2:
        R_time = find_name(DArray, fname="time", opt='coords')
        if Dndim == 3:
<<<<<<< HEAD
            return R_time, R_lat, R_lon
        elif Dndim == 4:
            R_level = find_name(DArray, fname="level", opt='coords')
            return R_time, R_lat, R_lon, R_level
    else:
        return R_lat, R_lon
=======
            return R_lat, R_lon,R_time
        elif Dndim == 4:
            R_level = find_name(DArray, fname="level", opt='coords')
            return R_lat, R_lon,R_time, R_level
    else:
        return R_lat, R_lon
>>>>>>> c80f4457 (First commit)
