# Standard library imports
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
from scipy.optimize import curve_fit, root_scalar
from scipy.stats import genextreme as gev
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

# Plotting and visualization
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
# import matplotlib.dates as mdates
# import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, LogNorm
# from matplotlib.tri import Triangulation
# from matplotlib.animation import FuncAnimation
# from matplotlib.image import imread
# from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Geospatial libraries
# import geopandas as gpd
import shapefile as shp
from shapely.geometry import MultiPolygon, Polygon, Point
import cartopy.io.img_tiles as cimgt
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.feature as cfeature
import cartopy.crs as ccrs
import geopandas as gpd

from my_junk import plot_shp
# import re

warnings.filterwarnings("ignore")


params = {
    'axes.titlesize': 40,
    'axes.labelsize': 40,
    'axes.labelpad': 15,
    'font.size': 50,
    'font.family': 'cmss10',
    'mathtext.fontset': 'stixsans',
    'legend.fontsize': 30,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    # 'text.usetex': True,
    # 'figure.autolayout': True,
    'ytick.right': False,
    'xtick.top': False,

    'figure.figsize': [12, 10],  # instead of 4.5, 4.5
    'axes.linewidth': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,

    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.minor.width': 3,

    'xtick.major.pad': 10,
    'ytick.major.pad': 12,
    # 'xtick.minor.pad': 14,
    # 'ytick.minor.pad': 14,

    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'axes.unicode_minus': False,
}
plt.clf()
matplotlib.rcParams.update(params)

color_list = ['#bd004e','#142354']
gradient_list =['#3e5466','#3a7c90','#a9d3d6','#8c9bae','#137c7a']
cluster_cmap = ListedColormap(color_list)
def myLogFormat(x,pos):
    # Return the formatted tick label
    times = np.log10(x)
    formatstring = r'$10^{%.0f}$' % times
    return formatstring

resample_mode = 3
# axis and

# GEV equation
#   Block maxima index
#? I have monthly data (12 values/year) and I want return period of 50 years => ri = 1/12

def RI(resample_mode = resample_mode):
    return 1/resample_mode if resample_mode > 0 else 1/365

def F(x, loc, scale, shape):
    return gev.cdf(x, c=shape, loc=loc, scale=scale)

def EV(x, loc, scale, shape):
    """_summary_
    Args:
        x (number): shape of GEV fit
        loc (float): location of GEV fit
        scale (float): scale/width of GEV fit
        shape (float): shape/fat tail/small tail of GEV fit

    Returns:
        float: value of exceedance probability function
        e.g.: Temperature rise is 1.5 degrees, how many percentage of value is above this point?
        
    """
    return gev.sf(x, c=shape, loc=loc, scale=scale)

def E_inc(x, loc, scale, shape, rise):
    return EV(x+rise, loc, scale, shape)/EV(x, loc, scale, shape)

def x_inv(E, loc, scale, shape):
    """
    What value of x that has E exceedance probability
    e.g. E = 0.99 => 99% of the dat
    a is below x or 1% is larger than x => Find x
    
    Args:
        E (float): exceedance probability
        loc (float): location of GEV fit
        scale (float): scale/width of GEV fit
        shape (float): shape/fat tail/small tail of GEV fit !! Reverse of the shape in the function
    Returns:
        float: value of x that has E exceedance probability
    """
    return gev.isf(E, c=shape, loc=loc, scale=scale)

def Tr(x, loc, scale, shape, ri = RI()):
    return ri / EV(x, loc, scale, shape)

def x_Tr(yr, loc, scale, shape, ri = RI()):
    #? Given a return period, what is the value of x
    return gev.isf((ri/yr), c=shape, loc=loc, scale=scale)

def Tr_pct_reduce(TR, loc, scale, shape, rise):
    return TR - Tr(x_Tr(TR, loc, scale, shape), loc+rise, scale, shape) / TR

def DT(yr, loc, scale, shape, ri= RI()):
    return x_Tr(yr, loc, scale, shape, ri= ri) - x_Tr(1, loc, scale, shape, ri=ri)
    # ----------------------------- return ----------------------------- #

def f_interp(x_out, x_in, y_in):
    sort_index = np.argsort(x_in)
    x_in_sorted = x_in[sort_index]
    y_in_sorted = y_in[sort_index]
    return np.interp(x_out, x_in_sorted, y_in_sorted)

def last_index(data):
    [_,index,_] = np.unique(data[::-1], return_index=True, return_inverse=True)
    return len(data) - 1 - index

# region - INPUT from json
with open('constant.json') as f:
    data_dict = json.load(f)
    
rain_csv = pd.read_csv(data_dict['rain_csv'],names = data_dict['rain_columns'])
temp_csv = pd.read_csv(data_dict['temp_csv'], names = data_dict['temp_columns'])
temp_tuple = data_dict['temp_tuple']
rain_tuple = data_dict['rain_tuple']
no_T_station = data_dict['no_T2m_city']
no_R_station = data_dict['no_R_city']
name_T_station = data_dict['name_T2m_city']
name_R_station = data_dict['name_R_city']
dataset_list = data_dict['dataset']
#endregion

map_pro = ccrs.PlateCarree()
shp_path = os.getenv("vnm_sp")
vnmap = gpd.read_file(shp_path)
# vnmap = shp.Reader(shp_path)

obs_path = os.path.join(os.getenv("data"), "wrf_data/netcdf/para/")
Data_ind = os.path.join(obs_path, "indices/")
MST_ds = xr.open_dataset(os.path.join(obs_path, "indices/MST_obs_detrend.nc"))['Tx']
og_dict = {}

# region - MAIN FUNCTIONS

def open_og(var):
    """_summary_

    Args:
        var (str): ['Tx', 'Tn', 'R', 'T2m']

    Returns:
        xarray.Dataset: Dataset of the original data in the UPDATE_METEO
    """
    if var == "Tx":
        path = "para_167/max_obs_detrend_167.nc"
    elif var == "Tn":
        path = "para_167/min_obs_detrend_167.nc"
    elif var == "R":
        path = "para_228/obs_detrend_228.nc"
    else:
        path = f"indices/{var}_obs_detrend.nc"
    
    if 'R' in var: var_name = 'R'
    elif 'Tx' in var: var_name = 'Tx'
    elif 'Tn' in var: var_name = 'Tm'
    og_dict[var] = xr.open_dataset(os.path.join(obs_path, path))[var_name]
    return og_dict[var]

def fit_gev(ind,og=False,resample_mode = resample_mode):
    
    def fit_som(gev_df, og = True, ind = "Tx"):
        inertia = []
        gev_df.rename(columns={'Unnamed: 0': 'ind','Unnamed: 1': 'name_station'}, inplace=True)
        data = gev_df if og else gev_df[gev_df['ind'] == ind]
        columns = ['shape','loc','scale']
        data_numpy = data[columns].to_numpy()
        for i in range(30):
            som = SOM(m=1,n=i+1,dim = len(columns))
            som.fit_predict(data_numpy)
            # Kneelocator
            inertia.append(som.inertia_)
        Kneeloc = KneeLocator(range(1,31),inertia,curve='convex',direction='decreasing')
        
        # plt.plot(range(1,31),inertia)
        # plt.title('elbow method for SOM')
        # plt.xlabel('number of clusters')
        # plt.ylabel('WCSS')
        # # #print it by using the vlines
        # plt.vlines(Kneeloc.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        # plt.savefig(os.getenv('img')+"/SOM_inertia.png")
        no_cluster= Kneeloc.knee
        som = SOM(m=1,n=no_cluster,dim = len(columns))
        data['cluster'] = som.fit_predict(data_numpy)+1
        # #print(data[data['cluster']==1])
        # #print(data[data['cluster']==2])
        # #print(data[data['cluster']==3])
        # #print(data[data['cluster']==4])
        return data
    """_summary_

    Args:
        ind (str): Can be 'Tx', 'Tn', 'R', 'T2m', 'Tm' or ETCDDI index
        og (bool, optional): original data: True, ETCDDI = False. Defaults to False.
        resample_mode (int, optional): number of extremes/year. Defaults to resample_mode.

    Returns:
        _type_: _description_
    """
    if "T" in ind:
        no_list = no_T_station
        name_list = name_T_station
        loc_df = temp_csv
    else:
        no_list = no_R_station
        name_list = name_R_station
        loc_df = rain_csv
        
        
    gev_dict = {}
    clean_dict={}
    og_suffix = "OG" if og else "IND"
    
    # # indices - wise
    # if os.path.exists(f"gev_{ind}{og_suffix}.csv"):
    #     return pd.read_csv(f"gev_{ind}{og_suffix}.csv")
    
    #? We put it inside the function so they comply with the change of resample_mode
    def sort_top(data, axis=None, resample_mode = resample_mode, **kwargs):
        # Sort the data along the specified axis and select the top values
        sorted_data = np.sort(data.flatten())[::-1]
        final = sorted_data[:resample_mode]
        return final
    
    print(f"Resample mode: {resample_mode}")
    
    if og:
        """_summary_
        #? THE ORIGINAL DATA
        #! You should not use Txx or Tnn for 3 max or min values
           i.e. multiple high peak in the same months -> hidden
        """
        if ind not in og_dict.keys():
            og_ds= open_og(ind)
        else:
            og_ds = og_dict[ind]
    
        for i, no_city in enumerate(no_list):
            name_city = name_list[i]
            main_ds = og_ds.sel(no_station=no_city).drop_vars('no_station')
            lon, lat, elev = loc_df.loc[loc_df['name_station'] == name_city, ['lon', 'lat', 'elev']].values[0]
            if elev < 0: elev = 0
                    # Drop coordinates that have only one value
            for coord in list(main_ds.coords):
                if main_ds[coord].size == 1:
                    main_ds = main_ds.drop_vars(coord)
            # Clean and detrend data
            #? The output of apply has to be a DataArray => convert to numpy array
            
            if resample_mode == 0:
                main_yr_ds = main_ds.values.flatten()
            else:
                main_yr_ds = main_ds.groupby('time.year').apply(lambda x: xr.DataArray(sort_top(x.values), dims=['top_values'])).values.flatten()
            clean_data = main_yr_ds[np.isfinite(main_yr_ds)]
            mst_data = MST_ds.sel(no_station=no_city).values[0]
            
            if len(clean_data) > 0:
                clean_dict[name_city] = clean_data.tolist() # Turn to list to save as hd5
                params = gev.fit(clean_data)
                if name_list[i] not in gev_dict.keys():
                    gev_dict[name_city] = {}
                gev_dict[name_city] = {"shape": params[0],
                                        "loc": params[1], "scale": params[2],
                                        "lon": lon, "lat": lat, "elev": elev, 'MST': mst_data}
        # mu = loc, sigma = scale, shape = k
        gev_df = pd.DataFrame.from_dict({i: gev_dict[i]
                                        for i in gev_dict.keys()
                                        },
                                        orient='index')
        gev_df.index.name = 'name_station' # Index to read later
    else:
        #? THE INDICES DATA (TXX, TNN, .....)
        obs_nc = Data_ind+ind+f"_obs.nc"
        ds = xr.open_dataset(obs_nc)
        var_name = next((var for var in list(
                                ds.data_vars.keys()) if ind in ['R', 'T2m', 'Tm', 'Tx']), None)
        for i, no_city in enumerate(no_list):
            main_ds = ds.sel(no_station=no_city)
            lon, lat, elev = loc_df.loc[loc_df['name_station'] == name_list[i], ['lon', 'lat', 'elev']].values[0]
            if elev < 0: elev = 0
            clean_data = main_ds[var_name].values
            clean_data = clean_data[np.isfinite(clean_data)]
            mst_data = MST_ds.sel(no_station=no_city).values[0]
            if len(clean_data) > 0:
                clean_dict[name_city] = clean_data.tolist() # Turn to list to save as hd5
                params = gev.fit(clean_data)
                if ind not in gev_dict.keys():
                    gev_dict[ind] = {}
                gev_dict[ind][name_list[i]] = {"shape": params[0],
                                                "loc": params[1], "scale": params[2],
                                                "lon": lon, "lat": lat, "elev": elev, 'MST': mst_data}
        # mu = loc, sigma = scale, shape = k
        gev_df = pd.DataFrame.from_dict({(i, j): gev_dict[i][j]
                                        for i in gev_dict.keys()
                                        for j in gev_dict[i].keys()},
                                        orient='index')
        gev_df.index.names = ['ind', 'name_station'] # MultiIndex to read later
        
    with h5py.File(f'clean_{ind}/{og_suffix}_{resample_mode}.h5', 'w') as f:
        # Save the dictionary to an HDF5 file
        for key, value in clean_dict.items():
            f.create_dataset(str(key), data=value)
    
    if not os.path.exists(f"gev_{ind}"):
        os.makedirs(f"gev_{ind}")
    
    # SOM clustering
    gev_df = fit_som(gev_df, og=og, ind=ind)
    gev_df.to_csv(f"gev_{ind}/{og_suffix}_{resample_mode}_test.csv")
    return gev_df
    

# fit_gev('Tx', og=True, resample_mode=3)
# fit_gev('Tx', og=True, resample_mode=0)


# region - GEV FIT

# # input
# futures = []
# var_tuple = ['Tx']
# with multiprocessing.Pool() as pool:
#     for ind in var_tuple:
#         futures += [pool.apply_async(fit_gev, args=(ind, True, i+1)) for i in range(12)]
#     pool.close()
#     pool.join()

#endregion

# region - DELTA GEV
def delta_gev():

    markers = ['o','s','^','v']#,'<','>']
    # edgecolors = ['black','red','blue','green','yellow']#,'purple']
    indice_list =['DT_20','DT_50','DT_MST']
    indice_plot_list = [
        r'$\Delta T_{20\mathrm{yr} \rightarrow 1\mathrm{yr}}$',
        r'$\Delta T_{50\mathrm{yr} \rightarrow 1\mathrm{yr}}$',
        r'$\Delta T_{1\mathrm{yr} \rightarrow \mathrm{MST}}$'
    ]
    fig,axes = plt.subplots(1,3,figsize=(30,15))
    fig.tight_layout(pad = 0.1)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    sns.despine()
    cmap=sns.color_palette("RdYlBu_r", as_cmap=True)

    min_20, max_20, min_50, max_50, min_shape, max_shape = 10, 0, 10, 0, 10, 0
    for i in range(12):
        i_ri = i+1
        # Run fit check
        csv_name = f"gev_Tx/OG_DT_{i}_test.csv"
        if os.path.exists(csv_name):
        #     data = pd.read_csv(csv_name)
        # else:
            # data = fit_gev("Tx", og=True, resample_mode=i)
            org_data = pd.read_csv(f"gev_Tx/OG_{i}.csv")
            ri = 1/(i_ri)
            
            org_data['DT_20'] = DT(20, org_data['loc'], org_data['scale'], org_data['shape'], ri=ri)
            org_data['DT_50'] = DT(50, org_data['loc'], org_data['scale'], org_data['shape'], ri=ri)
            org_data['DT_MST'] = x_Tr(1, org_data['loc'], org_data['scale'], org_data['shape'], ri=ri) - org_data['MST']
            data = pd.concat([org_data['name_station'], org_data['loc'], org_data['scale'], org_data['shape'], org_data['DT_20'], org_data['DT_50'], org_data['DT_MST']], axis=1)
        
        #print(ri, x_Tr(20,org_data['loc'], org_data['scale'], org_data['shape'], ri= ri))
        #print(x_Tr(1,org_data['loc'], org_data['scale'], org_data['shape'], ri= ri))
            # #print(org_data['DT_20'], org_data['DT_50'], org_data['DT_MST'])
            # #print(ri, data['DT_20'],data['DT_50'],data['DT_MST'])
        # raise KeyboardInterrupt
        data.to_csv(csv_name)
        if i > 1:
            min_20 = data['DT_20'].min() if data['DT_20'].min() < min_20 else min_20
            max_20 = data['DT_20'].max() if data['DT_20'].max() > max_20 else max_20
            min_50 = data['DT_50'].min() if data['DT_50'].min() < min_50 else min_50
            max_50 = data['DT_50'].max() if data['DT_50'].max() > max_50 else max_50
            min_shape = data['shape'].min() if data['shape'].min() < min_shape else min_shape
            max_shape = data['shape'].max() if data['shape'].max() > max_shape else max_shape

        for j,ax in enumerate(axes.flatten()):
            if i_ri%3 == 0: # or i_ri == 1:#(i_ri == 1 and j%2 != 0):
                order = int(i_ri/3) - 1
                valid_data = data.dropna(subset=[indice_list[j]])
                if 'MST' in indice_list[j]:
                    sc = ax.scatter(-valid_data['shape'], valid_data[indice_list[j]], c=valid_data['scale'], marker=markers[order], cmap=cmap, s=250, edgecolors='k')
                else:
                    max_value = max_20 if '20' in indice_list[j] else max_50
                    #!! Original is MATLAB. Shape in MATLAB = - shape in Python
                    #!! Must do: -valid_data['shape']
                    sc = ax.scatter(-valid_data['shape'],valid_data[indice_list[j]], c=valid_data['scale'], marker=markers[order], cmap=cmap, s=250, edgecolors='k')
                # Remember to check MST as a single value only
        # ax.set_ylim(0, max_value)
        # ax.set_xlim(min_shape, max_shape)
        # Set the title inside the graphing area
            ax.text(0.05, 1.1, indice_plot_list[j], transform=ax.transAxes,
                    fontsize=50, va='top', ha='left')
            ax.set_xlabel(r'$\mathrm{k}$')
            ax.set_ylabel(r'$\Delta T(^\circ C)$')
    #! The outlier dude is VIETTRI

    #print(min_20,max_20,min_50,max_50,min_shape,max_shape)

    mu= 10 #????????????????
    k = np.linspace(min_shape,max_shape,100)
    DTi_20 = np.linspace(min_20,max_20, 100)
    DTi_50 = np.linspace(min_50,max_50, 100)

    # k = np.linspace(min_shape,max_shape,1000)
    # DTi_20 = np.linspace(min_20,max_20, 1000)
    # DTi_50 = np.linspace(min_50,max_50, 1000)
    [K_20,DTi_20] = np.meshgrid(k,DTi_20)
    [K_50,DTi_50] = np.meshgrid(k,DTi_50)
    SIG_20 = np.zeros(K_20.shape)
    SIG_50 = np.zeros(K_50.shape)

    def compute_SIG(args):
        i, j = args
        result = root_scalar(lambda sig: DTi_20[i, j] - DT(20, mu, sig * (sig > 0) + 1e-6, K_20[i, j]), bracket=[1e-5, 1000]).root
        result_50 = root_scalar(lambda sig: DTi_50[i, j] - DT(50, mu, sig * (sig > 0) + 1e-6, K_50[i, j]), bracket=[1e-5, 1000]).root
        return (i, j, result, result_50)

    indices = list(np.ndindex(K_20.shape))
    with con() as pool:
        results = pool.map(compute_SIG, indices)
        pool.close()
        pool.join()

    for i, j, sig_20, sig_50 in results:
        SIG_20[i, j] = sig_20
        SIG_50[i, j] = sig_50


    # for i,j in  np.ndindex(K_20.shape):
    #     SIG_20[i, j] = root_scalar(lambda sig: DTi_20[i, j] - DT(20, mu, sig * (sig > 0) + 1e-6, K_20[i, j]), bracket=[1e-5, 1000]).root
    #     SIG_50[i, j] = root_scalar(lambda sig: DTi_50[i, j] - DT(50, mu, sig * (sig > 0) + 1e-6, K_50[i, j]), bracket=[1e-5, 1000]).root
    #     print(SIG_20[i, j], SIG_50[i, j])

    # Custom format function
    def custom_fmt(value):
        return f'$\sigma = {value:.2f}$'

    for j, ax in enumerate(axes.flatten()):
        if indice_list[j] == 'DT_20':
            CS = ax.contour(-K_20,DTi_20,SIG_20,level = 10, colors='black',linestyles='dashed')
            ax.clabel(CS, inline=1, fontsize=35, fmt=custom_fmt)
        if indice_list[j] == 'DT_50':
            CS = ax.contour(-K_50,DTi_50,SIG_50,level = 10,colors='black',linestyles='dashed')
            ax.clabel(CS, inline=1, fontsize=35, fmt=custom_fmt)
    # Create custom legend for markers
    legend_elements = [mlines.Line2D([0], [0], marker=markers[i], color='w', markeredgecolor='k', label=f'ri={(i+1)*3}', markerfacecolor='k', markersize=30) for i in range(len(markers))]
    fig.legend(handles=legend_elements,
            loc='center',
            ncol= len(markers),
            #? len(markers) if horizontal, 1 if vertical
            bbox_to_anchor=(0.5, 0.22),
            handlelength=1,  # Reduce the length of the legend handles
            handletextpad=0.5,  # Reduce the padding between the handles and the text
            frameon=False,
            fontsize = 35)

    # Add horizontal colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', pad=0.18)
    cbar.ax.set_aspect(0.01, adjustable='box')  # Adjust the aspect ratio to change the height
    cbar.set_label(r'$\sigma (^\circ C)$', rotation=0, fontsize=40, labelpad=3)

    fig.savefig(os.getenv('img')+"/Delta_GEV_test.png", bbox_inches='tight', transparent=True)
    # axes = axes.flatten()
    plt.close(fig)
    return

#endregion

# region - CLUSTER
#"""""
def cluster_plot(resample_mode = 3):
    params ={
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'axes.labelsize': 50,
        'xtick.minor.size': 10,
        'ytick.minor.size': 10,
    }
    matplotlib.rcParams.update(params)
    data = pd.read_csv(f"gev_Tx/OG_{resample_mode}.csv")
    fig = plt.figure(figsize=(40, 30))
    gs = fig.add_gridspec(1, 2, width_ratios=[4,1], wspace = -0.5)
    # Merge the first column into a single subplot
    ax_1 = fig.add_subplot(gs[0],projection=map_pro)
    gs_2 = gs[1].subgridspec(3, 1, hspace=0.3)
    # Create the other subplots
    ax2 = fig.add_subplot(gs_2[0])
    ax3 = fig.add_subplot(gs_2[1])
    ax4 = fig.add_subplot(gs_2[2])
    geo_limit = [101, 110, 7, 25]
    ax_1.axis('off')
    ax_1.set_extent(geo_limit)
    ax_1.xlabels_top = False
    ax_1.ylabels_right = False
    
    # ax_1.coastlines(color='black', linewidth=1, resolution='10m', alpha=1)
    # ax_1.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='-')
    # ax_1.set_xticks(np.linspace(101, 110, 5), crs=map_pro)
    # ax_1.set_yticks(np.linspace(10, 25, 5), crs=map_pro)
    # lon_formatter = LongitudeFormatter(degree_symbol=r'$^\circ$')
    # lat_formatter = LatitudeFormatter(degree_symbol=r'$^\circ$')
    # ax_1.xaxis.set_major_formatter(lon_formatter)
    # ax_1.yaxis.set_major_formatter(lat_formatter)

    # Ensure tick labels are visible
    ax_1.tick_params(axis='both', which='major',
                     direction='in', length=20)
    # ax_1.xaxis.set_tick_params(labelbottom=True)
    # ax_1.yaxis.set_tick_params(labelleft=True)
    ax_1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax_1.yaxis.set_minor_locator(MultipleLocator(0.5))
    # ax_1.set_xlabel('Longitude')
    # ax_1.set_ylabel('Latitude')
    # ax_1.set_title(ind, fontsize=20, pad=20, loc='left')
    # ax_1.set_xticklabels([])
    # ax_1.set_yticklabels([])

    # ax_1.set_xlabel('Longitude')
    # ax_1.set_ylabel('Latitude')

    # ----------------------- Plot the shapefile ----------------------- #
    
    # Plot border + DEM + mask
    plot_shp(ax_1)
    
    # Fill the inside with a certainc color
    # ax_1.fill(x, y, facecolor='red', edgecolor="k", linewidth=0.4)
    # ax_1.text(112, 10, "Spratly Islands", fontsize=30, rotation=45)
    # ax_1.text(111, 15, "Paracel Islands", fontsize=30, rotation=45)
    # ax_1.text(112, 13, "East Sea", fontsize=50,alpha=0.6)
    # ax_1.grid(linewidth=1, color='gray', alpha=0.5,
    #         linestyle='--')

    # fig.patch.set_facecolor('none')
    # fig.patch.set_alpha(0)
    # ax_1.patch.set_facecolor('none')
    
    ax_1.spines['top'].set_visible(True)
    ax_1.spines['right'].set_visible(True)
    ax_1.spines['bottom'].set_visible(True)
    ax_1.spines['left'].set_visible(True)

    # Set the title inside the graphing area
    ax_1.scatter(data['lon'], data['lat'], c=data['cluster'], cmap=cluster_cmap, s=500)

    ax2.scatter(data['shape'], data['loc'], c=data['cluster'], cmap=cluster_cmap, s=800, alpha = 0.5)
    ax2.set_xlabel(r'$\mathrm{k}$')
    ax2.set_ylabel(r'$\mu$')

    ax3.scatter(data['loc'], data['scale'], c=data['cluster'], cmap=cluster_cmap, s=800, alpha = 0.5)
    ax3.set_xlabel(r'$\mu$')
    ax3.set_ylabel(r'$\sigma$')
    ax4.scatter(data['scale'], data['shape'], c=data['cluster'], cmap=cluster_cmap, s=800, alpha=0.5)
    ax4.set_xlabel(r'$\sigma$')
    ax4.set_ylabel(r'$\mathrm{k}$')
    fig.tight_layout()
    fig.savefig(os.getenv('img')+f"/SOM_cluster_map_{resample_mode}.png", bbox_inches='tight', transparent=True)
#"""
#endregion

# cluster_plot(3)
# cluster_plot(0)

#region - ODD OF EXCEEDING
def doubling(resample_mode, gev_fit=False, doubling: Optional[Literal['linear', 'log', 'both']] = 'linear'):
    """_summary_

    Args:
        resample_mode (int): 0 for daily, other for maximum events/year
        gev_fit (bool, optional): Use empirical data or just GEV . Defaults to False.
    """
    
    params ={
        'xtick.labelsize': 90,
        'ytick.labelsize': 90,
        'axes.labelsize': 120,
        'axes.labelpad': 10,
        'xtick.minor.size': 10,
        'ytick.minor.size': 10,
        
    }
    matplotlib.rcParams.update(params)
    
    
    # fig, axes = plt.subplots(2, 2, figsize=(30, 40))
    fig, axes = plt.subplots(1, 2, figsize=(70, 40))
    fig.subplots_adjust(wspace = 0.2)
    ax = axes[0]
    ax_1 = axes[1]
    # ax_2 = axes[1,0]
    # ax_3 = axes[1,1]
    
        # gen_props
    line_props = {
    'color': 'black',
    'linestyle': '-',
    'linewidth': 13,
    'path_effects': [pe.withStroke(linewidth=7, foreground='w')]
    }
    
    text_props = {
    'fontsize': 100,
    'color': 'k',
    'fontweight': 'heavy',
    'transform': ax_1.transAxes,
    'path_effects': [pe.withStroke(linewidth=3, foreground='w')]
    }
    text_props_2 = {
    'fontsize': 150,
    'color': 'k',
    'fontweight': 'heavy',
    'transform': ax.transAxes,
    'path_effects': [pe.withStroke(linewidth=3, foreground='w')]
    }

        
    # ??: Recommended by the author
    # TODO: Need to try out this threshold
    #print("Importing data")
    raw_data_dict = {}
    with h5py.File(f'clean_Tx/OG_{resample_mode}.h5', 'r') as f:
        for key,item in f.items():
            raw_data_dict[key] = item[:]
    if os.path.exists(f"gev_Tx/OG_{resample_mode}.csv"):
        data_fitted_df = pd.read_csv(f"gev_Tx/OG_{resample_mode}.csv")
        if resample_mode == 0:
            data_fitted_df.rename(columns = {'Unnamed: 0': 'name_station'}, inplace=True)
        else:
            data_fitted_df.rename(columns = {'station': 'name_station'}, inplace=True)
    else:
        data_fitted_df = fit_gev('Tx', og=True, resample_mode=0)
    
    num_station = len(data_fitted_df.index)
    ri = RI(resample_mode = resample_mode)
    TR = 50
    mu_T_threshold = 3 # Threshold for fit of sigma 2x
    #! Use the daily data to calculate the exceedence probability
    for i,name_station in enumerate(raw_data_dict.keys()):
        # Initialize the variables
        mu_T = np.linspace(0,20,100) #local temperature rise
        mu_all = np.linspace(0,100,1000) #local temperature rise
        TR_rise = np.zeros(shape=(mu_T.size, num_station)) # return period
        E_rise = np.zeros(shape=(mu_T.size, num_station)) # exceedence probability
        ODD_rise = np.zeros(shape=(mu_T.size, num_station)) # odd of exceeding
        sigma_2x = np.zeros(mu_T.size) # slope of the line
        sigma_2x_fit = np.zeros(mu_T.size) # slope of the line
        
        # Get input values
        station_df = data_fitted_df[data_fitted_df['name_station'] == name_station]
        raw_data = raw_data_dict[name_station]
        mu_org = station_df['loc'].values[0]
        sigma_org = station_df['scale'].values[0]
        k_org = station_df['shape'].values[0]
        clus = station_df['cluster'].values[0]
        if not gev_fit:
            # ALL DATA
            if resample_mode == 0:
            # Value of x that has return period: 50 years (x_50)
                x0 = x_Tr(TR, mu_org, sigma_org, k_org, ri) # Reverse of cdf at TR
                
            # Calculate the exceedence probability emperically
                
                # Find the exceedance value of all values
                E_list = EV(raw_data, mu_org, sigma_org, k_org) # TODO: Think about the exceedance of ri = 1/3 or just all data
                
                # Find the last index of each exceedance value
                # Find the corresponding real value of the exceedance value
                last_index_E = last_index(E_list)
                x_tmp, E_tmp = raw_data[last_index_E], E_list[last_index_E]
                
                # Develop the relationship between x and E
                # Use x_50 to find E_50 (exceedence probability of x_50)(percent)
                E_base = f_interp(x0,x_tmp,E_tmp) # or E0 if you want original use linear but it's kinda outdated
                
                # Rising odd
                # E(x_50 - mu_T) - right shift compared to original
                for j, mu in enumerate(mu_T):
                    x_rise = x_tmp + mu
                    E_rise[j,i] = f_interp(x0, x_rise, E_tmp) # Eceedance of rise
            # EXTREME DATA (n events/year)
            else:
                E_base = ri/TR
                #?? Why doesn't they use the same way?
                #! They define Exceedence first here??? Why???
                x_tmp = raw_data[last_index(raw_data)][::-1] # Descending
                len_x = x_tmp.size
                E_tmp = np.linspace(1/len_x,len_x/(len_x+1),len_x)
                
                #? x_temp is descending<=we try to match the CDF => larger temp = smaller exceedance
                
                x0 = f_interp(E_base, E_tmp, x_tmp)
                
                for j, mu in enumerate(mu_T):
                    x_rise = x_tmp + mu
                    E_rise[j,i] = f_interp(x0, x_rise, E_tmp)
                    TR_rise[j,i] = ri/E_rise[j,i]
        # Use GEV distribution to calculate the exceedence probability
        else:
            #? Use GEV distribution to calculate the exceedence probability
            for j, mu in enumerate(mu_T):
                x_rise = x0 + mu
                E_rise[j,i] = EV(x0, mu_org+mu_T, sigma_org, k_org)
                TR_rise[j,i] = ri/E_rise[j,i]
                
        ODD_rise[:,i] = E_rise[:,i]/(1-E_rise[:,i])
        ODD_base = E_base/(1-E_base)
        ODD_ratio = ODD_rise[:,i]/ODD_base
        ODD_ratio_log = np.log2(ODD_ratio)
        if ODD_base in (np.nan, 0):
            sigma_2x[i] = np.nan
            sigma_2x_fit[i] = np.nan
        else:
            # Index for mu_T < mu_T_threshold
            index_sel = np.where(mu_T<mu_T_threshold)
            x_sel = mu_T[index_sel]
            
            # Choose ODD_ratio_log where mu_t < mu_T_threshold
            ODD_ratio_log_sel = ODD_ratio_log[index_sel]
            log_index = ~np.isnan(ODD_ratio_log_sel) # index of non-NaN values
            #? Slope of the line = doubling rate of growth
            # We find the slope of this line (m): log_2(ODD_ratio) = mu_T/sigma_2x = m.mu_T
            # => sigma_2x = 1/m
            
            # Method 1: Linear
            if doubling in ('linear', 'both'):
                sigma_2x[i] = 1/np.nanmean(np.diff(ODD_ratio_log_sel)/np.diff(x_sel))
            # Method 2: Polynomial
            if doubling in ('log', 'both'):
                p = np.polyfit(x_sel[log_index], ODD_ratio_log_sel[log_index], 1)
                sigma_2x[i] = 1/p[0] # Use polynomial relationship
        #print(sigma_2x[i])
        # PLOT
        ax.plot(mu_T/sigma_2x[i], ODD_ratio_log, color=color_list[clus-1], alpha=0.5, linewidth=7)
        ax_1.plot(mu_T, ODD_ratio, color=color_list[clus-1], alpha=0.4, linewidth=10)
        # ax_2.plot(mu_T/sigma_2x[i], ODD_ratio_log, color=color_list[clus-1], alpha=0.5)
        # ax_3.plot(mu_T, ODD_ratio, color=color_list[clus-1], alpha=0.5)
    
 
    # ax configuration
    ax.set_yticks(np.arange(0, 18, 2))
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 8)
    # if resample_mode==0: ax.plot(np.linspace(0,40,1000), 2.5*(np.log2(np.linspace(0,40,1000)+1)), color='gray', linestyle='-.', linewidth=10)
    ax.plot(mu_T,mu_T , **line_props)
    
    # Draw lines at x=0 and y=0
    ax.axhline(0, color='black', linewidth=5)
    ax.axvline(0, color='black', linewidth=5)
    ax.set_xlabel(r'$\frac{\mu_T}{\sigma_{2x}}$', fontsize = 140)
    ax.set_ylabel(r'$\log_2\left(\frac{\mathrm{Odds}}{\mathrm{Odds_0}}\right)$')
    
    # The equation of odds of exceedance
    ax.text(0.7, 0.2, r'$\frac{\mathrm{Odds}}{\mathrm{Odds_0}}$', **text_props_2)
    ax.annotate("",xy=(0.75, 0.28), xycoords = 'axes fraction', xytext = (0.45, 0.4), textcoords = 'axes fraction', arrowprops = {'arrowstyle':'simple, head_width=1, head_length=1', 'facecolor': 'black','linewidth': 6, 'connectionstyle':"arc3,rad=-0.2"})
    
    # # ax configuration
    # ax_2.plot(np.linspace(0,40,1000), 2.5*(np.log2(np.linspace(0,40,1000)+1)), color='black', linestyle='-.', linewidth=6)
    # ax_2.plot(np.linspace(0,40,1000),np.linspace(0,40,1000) , color='grey', linestyle='-.', linewidth=6)
    
    # # Draw lines at x=0 and y=0
    # ax_2.axhline(0, color='black', linewidth=0.5)
    # ax_2.axvline(0, color='black', linewidth=0.5)
    # ax_2.set_xlabel(r'$\frac{\mu_T}{\sigma_{2x}}$')
    # ax_2.set_ylabel(r'$\log_2\left(\frac{\mathrm{Odds}}{\mathrm{Odds_0}}\right)$')
    
    # ax_1 configuration
    ax_1.set_yscale('log',base=2)
    ax_1.set_ylim(1, 2**14)
    ax_1.set_xlim(0, 10)
    
    ax_1.plot(mu_all, 2**(mu_all/0.25), color='k', linestyle='-', linewidth=10)
    ax_1.plot(mu_all, 2**(mu_all/0.5), color='k', linestyle='-', linewidth=10)
    ax_1.plot(mu_all, 2**(mu_all/0.75), color='k', linestyle='-', linewidth=10)
    ax_1.plot(mu_all, 2**(mu_all), color='k', linestyle='-', linewidth=10)
    
    ax_1.text(0.23, 0.85, r'$0.25 ^\circ C$', rotation=73, **text_props)
    ax_1.text(0.5, 0.83, r'$0.5 ^\circ C$', rotation=60, **text_props)
    ax_1.text(0.73, 0.75, r'$0.75 ^\circ C$', rotation=55, **text_props)
    ax_1.text(0.9, 0.68, r'$1 ^\circ C$', rotation=35, **text_props)

    # Draw lines at x=0 and y=0
    ax_1.axhline(1, color='black', linewidth=5)
    ax_1.axvline(0, color='black', linewidth=5)
    ax_1.set_xlabel(r'$\mu_T$')
    ax_1.set_ylabel(r'$\frac{\mathrm{Odds}}{\mathrm{Odds_0}}$', fontsize = 140)
    
    # # ax_4 configuration
    # ax_3.set_yscale('log',base=2)
    # ax_3.s@import url(https://fonts.googleapis.com/css?family=Noto+Sans+Adlam+Unjoined:regular,500,600,700);et_ylim(1, 2**40)
    # ax_3.set_xlim(0, 20)
    
    # ax_3.plot(mu_all, 2**(mu_all/0.25), color=gradient_list[0], linestyle='-.', linewidth=6)
    # ax_3.plot(mu_all, 2**(mu_all/0.5), color=gradient_list[1], linestyle='-.', linewidth=6)
    # ax_3.plot(mu_all, 2**(mu_all/0.75), color=gradient_list[2], linestyle='-.', linewidth=6)
    # ax_3.plot(mu_all, 2**(mu_all), color=gradient_list[3], linestyle='-.', linewidth=6)
    # ax_3.plot(mu_all, 2**(mu_all**2/2.5), color='black', linestyle='-.', linewidth=6)
    
    # # Draw lines at x=0 and y=0
    # ax_3.axhline(1, color='black', linewidth=0.5)
    # ax_3.axvline(0, color='black', linewidth=0.5)
    # ax_3.set_xlabel(r'$\mu_T$')
    # ax_3.set_ylabel(r'$\frac{\mathrm{Odds}}{\mathrm{Odds_0}}$')
     
    fig.tight_layout(pad=2.0)
    fig.savefig(os.getenv('img')+f"/Doubling_rate_{resample_mode}.png", transparent=True)


# if __name__ == '__main__':
delta_gev()
    # fit_gev('Tx', og=True, resample_mode=0)
    # cluster_plot(resample_mode=0)
    # doubling(resample_mode=0, gev_fit=False, doubling='log')
    # doubling(resample_mode=3, gev_fit=False, doubling='log')


        

#endregion