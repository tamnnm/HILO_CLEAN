import numpy as np
import xarray as xr
import pandas as pd
import os
from datetime import datetime
from typing import Union, Optional, List, Tuple, Dict, Literal

coord_name = {"level":('level','bottom_top','isobaricInhPa'),
                "time":('time','Time'),
                "lat":('lat','latitude','XLAT','south_north'),
                "lon":('lon','longitude','XLONG','west_east')}
var_name = {'tmp':('tmp','air','temp','t2m','var167'),
            'tmx':('tmx','tmax'),
            'tmn':('tmin','tmn'),
            'precip':('precip','apcp','tp','prec','var228'),
            'rain':('twcr',),
            'shum':('shum','q','sh'),
            'rhum':('rhum','rh'),
            'z':('z','hgt'),
            'msl':('msl','prmsl')}
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
def find_name(dts, fname, fname_group = None, opt = "coords"):
    list_name = coord_name
    list_tuple = coord_tuple
    if opt == "coords":
        list_opt = dts.coords
    elif opt == "dims":
        list_opt = dts.dims
    elif opt == "var":
        list_opt = dts.variables
        list_name = var_name
        list_tuple = var_tuple

    if fname in list(list_opt):
        return fname
    else:
        fname_group = fname_group if fname_group else fname
        # check other possibilities in the group name
        if fname_group in list_name:
            list_name_group = list_name[fname_group]
        # check in the tuple of var or tuple of coords
        else:
            list_name_group = list_tuple
        return next((name for name in list_name_group if name in list(list_name)), None)
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
def cut_var(dts, var=None, val_only: bool = False):
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
        print(f"Opened {var} as name: {var_final}")

    if val_only:
        dts_final = dts_var.values
    else:
        dts_final = dts_var
    return dts_final, var_final

#MODULE to cut dataset according to grid and time
def cut_co(dts, ulat, dlat, ulon, dlon, name = [None, None], full: bool = False, data: bool = True):
    result = {}
    if name:
        # this does not check if name in co_names
        lat_co_name = find_name(dts, name[0], fname_group='lat', opt = "coords")
        lon_co_name = find_name(dts, name[1], fname_group='lon', opt = "coords")
    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")

    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}
    if data:
        lat_co = dts.coords[lat_co_name]
        lon_co = dts.coords[lon_co_name]

        #
        dlat = dlat if dlat else min(lat_co.values)
        ulat = ulat if ulat else max(lat_co.values)
        lat_cond=(lat_co >= dlat) & (lat_co <= ulat)


        dlon = dlon if dlon else min(lon_co.values)
        ulon = ulon if ulon else max(lon_co.values)
        lon_cond=(lon_co >= dlon) & (lon_co <= ulon)

        if [ulat, dlat, ulon, dlon].count(None) == 4:
            dts_new = dts
        else:
            cut_co_dict = {}
            if len(lat_co.shape) == 1 or len(lon_co.shape) == 1:
                # Only apply for one-dimensional coordinate
                if lat_cond.any():
                    lat_data = lat_co[lat_cond]
                    cut_co_dict[lat_co_name] = lat_data
                else:
                    lat_data = None
                if lon_cond.any():
                    lon_data = lon_co[lon_cond]
                    cut_co_dict[lon_co_name] = lon_data
                else:
                    lon_data = None
                # sel is much faster & efficient for memory than where
                dts_new=dts.sel(**cut_co_dict)
                result['data'] = {'lat': lat_data, 'lon': lon_data}
            else:
                # Apply for multi-dimensional coordinate (wrf-output)
                # cut must based on value, but mean must based on dimension
                #lat_data = lat_co.where(lat_cond, drop=True) if lat_cond else lat_co
                #lon_data = lon_co.where(lon_cond, drop=True) if lon_cond else lon_co
                if lat_cond and lon_cond:
                    geo_cond = (lat_cond & lon_cond)
                elif lat_cond:
                    geo_cond = lat_cond
                elif lon_cond:
                    geo_cond = lon_cond
                else:
                    geo_cond = None
                dts_new=dts.where(lat_cond & lon_cond, drop=True) if geo_cond else dts
                result['data'] = geo_cond
    if full:
        result['full_dts'] = dts_new

    # if lat does not change, name of lat will be None but the data is full coords. Same for lon
    # if it's WRF, lat and lon data will be lat_condition and lon_condition
    return result

# cut the datset with list of points
def cut_points(dts, points: Tuple[Tuple[float,float]], name = [None, None], full: bool = False, data: bool = True):
    """_summary_

    This function cuts a dataset based on given points and returns the result.

    Args:
        dts(xarray.Dataset): The input dataset to be cut.
        points(Tuple[Tuple[float, float]]): A tuple of tuples, where each inner tuple represents a point(latitude, longitude) in float.
        name(list, optional): A list containing the names of the latitude and longitude dimensions in the dataset. Defaults to[None, None].
        full(bool, optional): If True, the function will return the full dataset after cutting. Defaults to False.
        data(bool, optional): If True, the function will return the data of the cut points. Defaults to True.

    Raises:
        KeyError: If the names of the latitude or longitude dimensions are not found in the dataset.

    Returns:
        {'data': {'lat': lat_data, 'lon': lon_data},
         'full_dts' (optional): dts_new}

    """

    result = {}
    if name:
        lat_co_name = find_name(dts, name[0], fname_group='lat', opt = "coords")
        lon_co_name = find_name(dts, name[1], fname_group='lon', opt = "coords")
    if None in [lat_co_name, lon_co_name]:
        raise KeyError(f"Check the name of this dimension {name}.\n")

    points= ((float(point[0][0]), float(point[0][1])) for point in points)
    result['name'] = {'lat': lat_co_name,'lon': lon_co_name}

    if data:
        lat_co = dts.coords[lat_co_name]
        lon_co = dts.coords[lon_co_name]
        lat_data = lat_co[points[0][0]]
        lon_data = lon_co[points[0][1]]
        for point in points[1:]:
            lat_data = xr.concat([lat_data, lat_co[point[0]]], dim='points')
            lon_data = xr.concat([lon_data, lon_co[point[1]]], dim='points')
        dts_new = dts.sel({lat_co_name: lat_data, lon_co_name: lon_data})
        result['data'] = {'lat': lat_data, 'lon': lon_data}
    if full:
        result['full_dts'] = dts_new
    return result


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

    if data:
        result['data'] = time_co
    if full:
        result['full_dts'] = dts.sel({time_name_co: time_co})
    return result

# level doesn't need to extract, this to directly cut the dataset
def cut_level(dts, level: Optional[int], data: bool = False, full: bool = True, name = None):
    # level: usual case
    # bottom_top: wrfout -> It's in dimension
    # isobaricInhPa: era5
    result = {}
    level_name_co = find_name(dts, name, fname_group='level', opt = "dims")
    if level_name_co:
        result['name'] = level_name_co
    else:
        raise KeyError(f"Check the name of this dimension {name}.\n" )
    if data:
        level_co= dts.coords[level_name_co]
        level_co = level_co[(level_co == level)] if level else level_co
        result['data'] = level_co
    if full:
        result['full_dts'] = dts.sel({level_name_co: level})
    return result

# calculate mean along certain dimension
# this is kinda trash, please use cut_mlp instead
def calculate_mean(final_dts, dim):
    result = {}
    if dim not in list(final_dts.dims):
        dim = next((dim_name for dim_name in coord_name[dim] if dim_name in list(final_dts.dims)), None)
    if dim is None: raise KeyError(f"Dimension {dim} is not in the dataset")
    result['name'] = dim
    result['full_dts'] = final_dts.mean(dim=dim)
    return result

def cut_mlp(dts, time_dict=None, geo_dict=None, level_val: Optional[int] = None, dim_mean: list = None, dim_plot: list =  None, var_plot: str =  None):
    #print(
    #"Be careful with the dimension of axis you are choosing\n\
    #you should check the order of the dimenions\n\
    #e.g. level,time,lat,lon then after cut lat, lon you should choose ax=0 (mean along time element)\n\
    #It's going to return the mean at a certain level")

    # collect name for each dimension to use in mean
    co_name = {}

    if time_dict:
        co_name['time'] = time_dict['name']
        if time_dict['full_dts']:
            dts = time_dict['full_dts']
        else:
            time_co_name = time_dict['name']
            time_data = time_dict['data']
            dts = dts.sel(**{time_co_name: time_data})
    if geo_dict:
        co_name['lat'] = geo_dict['name']['lat']
        co_name['lon'] = geo_dict['name']['lon']
        geo_co_dict = {}
        if geo_dict['full_dts']:
            dts = geo_dict['full_dts']
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
        dts = level_dict['full_dts']
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
            dts_mean = dts.mean()
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
    return dts
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
            [ds_var, var_name_uniform] = cut_var(ds, var=var_name)
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
