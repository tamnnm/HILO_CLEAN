#region MODULES
from scipy.optimize import curve_fit
import os
import numpy as np
import xarray as xr
import json
import scipy.stats as sst
import multiprocessing

# Plotting and visualization
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation
from matplotlib.image import imread
from matplotlib.patches import Wedge, PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Geospatial libraries
# import geopandas as gpd
import shapefile as shp
# from shapely.geometry import MultiPolygon
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.feature as cfeature
import cartopy.crs as ccrs

#endregion

params = {
    'axes.titlesize': 40,
    'axes.labelsize': 60,
    'axes.labelpad': 15,
    'font.size': 50,
    'font.family': 'cmss10',
    'mathtext.fontset': 'stixsans',
    'legend.fontsize': 30,
    'legend.loc': 'lower left',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    # 'lines.linewidth': 3,
    # 'text.usetex': True,
    # 'figure.autolayout': True,
    'ytick.right': False,
    'xtick.top': False,

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

Data_wd = os.getenv("data")
Code_wd = os.path.join(os.getenv("mcode"), "01.city_list_obs/")
Ind_wd = os.path.join(Data_wd, "wrf_data/netcdf/para/indices/")
img_wd = os.getenv("img")

pcp_ds = xr.decode_cf(xr.open_dataset(os.path.join(Ind_wd, "seas_PRCPTOT_obs.nc")))['R']
no_wet_ds = xr.decode_cf(xr.open_dataset(os.path.join(Ind_wd, "seas_no_wet_obs.nc")))['R']
sdii_ds = xr.decode_cf(xr.open_dataset(os.path.join(Ind_wd, "seas_SDII_obs.nc")))['R']
obs_ds = xr.decode_cf(xr.open_dataset('/data/projects/REMOSAT/tamnnm/obs/UPDATE_METEO/OG_R_daily_1960_2019.nc'))

# pcp_ds = pcp_ds.where(pcp_ds.time.dt.year <= 1990,drop=True)
# no_wet_ds = no_wet_ds.where(no_wet_ds.time.dt.year <= 1990,drop=True)
# sdii_ds = sdii_ds.where(sdii_ds.time.dt.year <= 1990,drop=True)

def open_percentile(filename):
    main = xr.decode_cf(xr.open_dataset(os.path.join(Ind_wd, filename)))['R']
    base = main.where(main.time.dt.year <=1990,drop=True)
    now = main.where(main.time.dt.year >1990,drop=True)
 
    return main, base, now

r99p_tot = xr.decode_cf(xr.open_dataset(os.path.join(Ind_wd, "R99p_obs.nc")))['R']
r99p_day, r99p_day_base, r99p_day_now = open_percentile("R99p_day_obs.nc")
r99p_sing, r99p_sing_base, r99p_sing_now = open_percentile("R99p_sing_obs.nc")

r95p_tot = xr.decode_cf(xr.open_dataset(os.path.join(Ind_wd, "R95p_obs.nc")))['R']
r95p_day, r95p_day_base, r95p_day_now = open_percentile("R95p_day_obs.nc")
r95p_sing, r95p_sing_base, r95p_sing_now = open_percentile("R95p_sing_obs.nc")

script_path = script_path = os.path.dirname(os.path.abspath(__file__))
json_path = script_path+"/constant.json"
map_pro = ccrs.PlateCarree()
shp_path = os.getenv("vnm_sp")
vnmap = shp.Reader(shp_path)


seas = ['MAM', 'JJA', 'SON','DJF']

with open(json_path, 'r') as json_file:
    data_dict = json.load(json_file)

rain_csv = data_dict['rain_csv']
rain_tuple = data_dict['rain_tuple']
no_R_station = data_dict['no_R_city']
name_R_station = data_dict['name_R_city']

def pcp_func(xdata,a,b,c):
    no_wet, sdii = xdata
    return a*no_wet+b*sdii+c

def update_proj(ax, projection = map_pro):
    rows, cols, start, stop = ax.get_subplotspec().get_geometry()
    ax.remove()
    ax = fig.add_subplot(rows, cols, start+1, projection = projection)
    return ax
dom_dict = {}

fig, axes = plt.subplots(4,2, figsize = (20,45))

#region - PLOT (2 plot : scatter - map)
# for i in range(4):
#     dom_seas = []
#     s_n_list = []
#     s_i_list = []
    # markers_list = []
    
    # ax_left = axes[i,0]
    # ax_right = update_proj(axes[i,1])
    
    # for no_station in no_R_station:
    #     pcp_ind = pcp_ds.sel(no_station = no_station).values[1::][i::4]
    #     no_wet_ind = no_wet_ds.sel(no_station = no_station).values[1::][i::4]
    #     sdii_ind = sdii_ds.sel(no_station = no_station).values[1::][i::4]
    
    #     nonnan_index = np.where(~np.isnan(no_wet_ind) & ~np.isnan(sdii_ind) & ~np.isnan(pcp_ind))
    #     no_wet_ind_clean= no_wet_ind[nonnan_index]
    #     sdii_ind_clean = sdii_ind[nonnan_index]
    #     ydata = pcp_ind[nonnan_index]
        
    #     no_wet_std = np.std(no_wet_ind_clean)
    #     sdii_std = np.std(sdii_ind_clean)
         
    #     params = curve_fit(pcp_func, (no_wet_ind_clean, sdii_ind_clean), ydata, maxfev = 10000)[0]
    #     s_n = abs(params[0])*no_wet_std
    #     s_i = abs(params[1])*sdii_std
    #     print(s_n>s_i, no_wet_std>sdii_std, params[0]> params[1])
    #     # Save the values
    #     s_n_list.append(s_n)
    #     s_i_list.append(s_i)
    
    #     # 0: frequency, 1: intensity
    #     dom_seas.append(0) if s_n > s_i else dom_seas.append(1)
        
        # Plot each station
        # lat = obs_ds.sel(no_station = no_station).lat.values
        # lon = obs_ds.sel(no_station = no_station).lon.values
        
        # if s_n > s_i:
        #     ax_left.scatter(s_n, s_i, marker = '8', s = 350, facecolors='#bd004e', alpha = 0.75)
        #     ax_right.scatter(lon, lat, marker = '8', s = 100, c = '#bd004e')
        # else:
        #     ax_left.scatter(s_n, s_i, marker = 'X', s = 350, facecolors='#142354',alpha = 0.75)
        #     ax_right.scatter(lon,lat, marker = 'X', s = 100, c='#142354')
    
    # dom_dict[seas[i]] = dom_seas
    
    # Plot the line
    # top_corner = max(s_n_list + s_i_list)+10
    # ax_left.plot([0,top_corner],[0,top_corner], color = 'k', linestyle = '--')
    # ax_left.axhline(0, color='black', linewidth=0.5)
    # ax_left.axvline(0, color='black', linewidth=0.5)
    # ax_left.set_xlim(0,top_corner)
    # ax_left.set_ylim(0,top_corner)
    
    # ax_left.text(0.05, 0.45,"{:.2f}%".format(dom_seas.count(1)/len(dom_seas)*100), fontsize = 45, transform = ax_left.transAxes, rotation = 270, color = '#142354', weight = 'bold')
    # ax_left.text(0.9, 0.45,"{:.2f}%".format(dom_seas.count(0)/len(dom_seas)*100), fontsize = 45, transform = ax_left.transAxes, rotation = 90, color = '#bd004e', weight = 'bold')
    # ax_left.text(0.05, 0.9, seas[i], fontsize = 45, transform = ax_left.transAxes, weight = 'bold')
    # ax_left.set_xlabel(r'$S_n$')
    # ax_left.set_ylabel(r'$S_i$', rotation = 90)
    
        
    # ax_right.set_extent([101, 110, 8, 25])
    # ax_right.xlabels_top = False
    # ax_right.ylabels_right = False
    # # ax_right.coastlines(color='black', linewidth=1, resolution='10m', alpha=1)
    # # ax_right.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='-')
    # ax_right.set_xticks(np.linspace(101, 110, 5), crs=map_pro)
    # ax_right.set_yticks(np.linspace(8, 25, 5), crs=map_pro)
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # lat_formatter = LatitudeFormatter()
    # ax_right.xaxis.set_major_formatter(lon_formatter)
    # ax_right.yaxis.set_major_formatter(lat_formatter)

    # # Ensure tick labels are visible
    # ax_right.tick_params(axis='both', which='major',
    #                 labelsize=10, direction='in', length=10)
    # ax_right.tick_params(axis='both', which='minor',
    #                 labelsize=8, direction='in', length=4)
    # ax_right.xaxis.set_tick_params(labelbottom=False)
    # ax_right.yaxis.set_tick_params(labelleft=False)
    # ax_right.xaxis.set_minor_locator(MultipleLocator(0.5))
    # ax_right.yaxis.set_minor_locator(MultipleLocator(0.5))


    # # ax_right.set_title(ind, fontsize=20, pad=20, loc='left')
    # # ax_right.set_xticklabels([])
    # # ax_right.set_yticklabels([])

    # # ax_right.set_xlabel('Longitude')
    # # ax_right.set_ylabel('Latitude')


    # # plot vietnam and   the station point
    # txt_shapes = []
    # for vnmapshape in vnmap.shapeRecords():
    #     listx = []
    #     listy = []
    #     # parts contains end index of each shape part
    #     parts_endidx = vnmapshape.shape.parts.tolist()
    #     parts_endidx.append(len(vnmapshape.shape.points) - 1)
    #     for i in range(len(vnmapshape.shape.points)):
    #         x, y = vnmapshape.shape.points[i]
    #         if i in parts_endidx:
    #             # we reached end of part/start new part
    #             txt_shapes.append([listx, listy])
    #             listx = [x]
    #             listy = [y]
    #         else:
    #             # not end of part
    #             listx.append(x)
    #             listy.append(y)

    # for zone in txt_shapes:
    #     x, y = zone
    #     # Plot only the border
    #     ax_right.plot(x, y, color="k", markersize=10e-6,
    #             linewidth=0.4, alpha=0.5)
    # # Fill the inside with a certainc color
    # # ax_right.fill(x, y, facecolor='red', edgecolor="k", linewidth=0.4)
    # # ax_right.text(112, 10, "Spratly Islands", fontsize=5, rotation=45)
    # # ax_right.text(111, 14.5, "Paracel Islands", fontsize=5, rotation=45)
    # # ax_right.text(110, 13, "EAST SEA", fontsize=5, rotation=90, alpha=0.6)
    # # ax_right.text(102,20 , "Laos", color="w", fontsize=16)
    # # ax_right.text(108,13 , "Vietnam",color="w", fontsize=16,rotation=90)
    # # ax_right.text(104, 13, "Cambodia",color="w", fontsize=16)
    # # ax_right.grid(linewidth=1, color='gray', alpha=0.5,
    # #         linestyle='--')

    # # fig.patch.set_facecolor('none')
    # # fig.patch.set_alpha(0)
    # # ax_right.patch.set_facecolor('none')
    # # ax_right.patch.set(lw=2, ec='k', alpha=0.5)

    # ax_right.spines['top'].set_visible(True)
    # ax_right.spines['right'].set_visible(True)
     
    # ax_right.spines['bottom'].set_visible(True)
    # ax_right.spines['left'].set_visible(True)

def process_station(option,no_station, pcp_subset, no_wet_subset, sdii_subset):
    start_index = 0 if option == "now" else 1
    pcp_subset = pcp_subset.sel(no_station = no_station).values[start_index::]
    no_wet_subset = no_wet_subset.sel(no_station = no_station).values[start_index::]
    sdii_subset = sdii_subset.sel(no_station = no_station).values[start_index::]
    
    
    dom_seas = []
    
    for i in range(4):
        if option == "now":
            i+=1 if i+1 < 4 else 0
        
        no_wet_ind = no_wet_subset[i::4]
        sdii_ind = sdii_subset[i::4]
        pcp_ind = pcp_subset[i::4]
        
        nonnan_index = np.where(~np.isnan(no_wet_ind) & ~np.isnan(sdii_ind) & ~np.isnan(pcp_ind))
        no_wet_ind_clean= no_wet_ind[nonnan_index]
        sdii_ind_clean = sdii_ind[nonnan_index]
        ydata = pcp_ind[nonnan_index]
        
        no_wet_std = np.std(no_wet_ind_clean)
        sdii_std = np.std(sdii_ind_clean)
        
        params = curve_fit(pcp_func, (no_wet_ind_clean, sdii_ind_clean), ydata, maxfev = 10000)[0]
        s_n = abs(params[0])*no_wet_std
        s_i = abs(params[1])*sdii_std
        dom_seas.append([0 if s_n > s_i else 1, s_n, s_i, no_station])
    return dom_seas

def s_n_i(option):
    if option == 'all':
        pcp_subset = pcp_ds
        no_wet_subset = no_wet_ds
        sdii_subset = sdii_ds
    elif option == 'base':
        pcp_subset = pcp_ds.where(pcp_ds.time.dt.year <= 1990,drop=True)
        no_wet_subset = no_wet_ds.where(no_wet_ds.time.dt.year <= 1990,drop=True)
        sdii_subset = sdii_ds.where(sdii_ds.time.dt.year <= 1990,drop=True)
    elif option == 'now':
        pcp_subset = pcp_ds.where(pcp_ds.time.dt.year > 1990,drop=True)
        no_wet_subset = no_wet_ds.where(no_wet_ds.time.dt.year > 1990,drop=True)
        sdii_subset = sdii_ds.where(sdii_ds.time.dt.year > 1990,drop=True)
    
    dom_seas_all = []
    with multiprocessing.Pool() as pool:
        for no_station in no_R_station:
            dom_seas_all.append(pool.apply(process_station, args = (option,no_station, pcp_subset, no_wet_subset, sdii_subset)))
    return np.array(dom_seas_all)

# Structure
#? (number of station, number of season, [0: dominance (0 for freq, 1 for intensity), 1: frequency component, 2: intensity component])
dom_seas_all = s_n_i('all')
dom_seas_base = s_n_i('base')
dom_seas_now = s_n_i('now')

#Example: :,0,0 -> all station, MAM, dominance component

def plot_dom():
    params = {
        'patch.linewidth': 10,
        'patch.edgecolor': 'w',
        }
    matplotlib.rcParams.update(params)
    
    fig, axes = plt.subplots(2,2, figsize = (35,35))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    for i,ax in enumerate(axes.flatten()):
        ax.axis('off')
        # Plot donut chart
        
        #!!! CHECK THIS. ALL THIS MUST RETURN 49 STATIONS
        all = dom_seas_all[:,i,0].tolist()
        base = dom_seas_base[:,i,0].tolist()
        now = dom_seas_now[:,i,0].tolist()
        angle = 60
        
        textprops = {'color': 'w', 'fontsize': 45, 'rotation': 0 if i!=1 else 60, 'fontweight': 'bold'}
        # print(all.shape)
        # raise KeyboardInterrupt
        w,l,p=ax.pie([all.count(0), all.count(1)],# labels = ['Frequency', 'Intensity'],
            autopct='%1.1f%%', colors = ['fbe499', 'c7eae9'], startangle=angle,
            radius=1.5,pctdistance = 0.85, textprops = textprops,
            wedgeprops=dict(width=0.9, edgecolor = 'w', linewidth = 10))
        
        for autotext in p:
            pct = float(autotext.get_text().strip('%'))
            if pct > 50:
                autotext.set_position((0,-1.3))
                autotext.set_rotation(0)
            elif i!=1:
                autotext.set_rotation(10) if pct < 30 and pct > 15 else autotext.set_rotation(-7)
                
        w,l,p=ax.pie([now.count(0), now.count(1)],# labels = ['Frequency', 'Intensity'],
            autopct='%1.1f%%',pctdistance = 0.78, colors = ['#f9d55c', '#98dad9'], startangle=angle,
            radius=1.1,textprops = textprops,
            wedgeprops=dict(width=0.85,edgecolor = 'w', linewidth = 10))
        
        for autotext in p:
            pct = float(autotext.get_text().strip('%'))
            if pct > 50:
                autotext.set_position((0,-0.9))
                autotext.set_rotation(0)
            elif i != 1:
                autotext.set_rotation(5) if (pct < 30 and pct > 15) else autotext.set_rotation(-5)
        
        w,l,p=ax.pie([base.count(0), base.count(1)],# labels = ['Frequency', 'Intensity'],
               colors = ['#bd004e', '#142354'], startangle=angle,
               radius=0.65,autopct='%1.1f%%', pctdistance = 0.65,textprops = textprops,
               wedgeprops=dict(width=0.4,edgecolor = 'w', linewidth = 10))
        
        for autotext in p:
            pct = float(autotext.get_text().strip('%'))
            if pct > 50:
                autotext.set_position((0,-.45))
                autotext.set_rotation(0)
            else:
                if i in (1,2):
                    autotext.set_fontsize(35)
                elif i ==0:
                    autotext.set_rotation(0)
                elif i==3:
                    autotext.set_rotation(10)
                
        
        ax.text(0, 1.1, seas[i], fontsize = 70, fontweight = 'bold', ha = 'center', va = 'center', transform = ax.transAxes)
        
    fig.savefig(os.path.join(img_wd, "ME_dom_pie.png"), bbox_inches = 'tight', dpi = 300, transparent=True)
#endregion

plot_dom()



# freq_95p, freq_99p, int_95p, int_99p = [],[],[],[]

# for no_station in no_R_station:
#     def clean_df(ds):
#         raw_ds =  ds.sel(no_station = no_station).values
#         return raw_ds[~np.isnan(raw_ds)]
        
#     # 95p
#     freq_95p_base = np.sum(clean_df(r95p_day_base))
#     freq_95p_now = np.sum(clean_df(r95p_day_now))
#     min_freq = int(np.min([freq_95p_base, freq_95p_now]))
        
#     int_95p_base = np.sort(clean_df(r95p_sing_base))[::-1]
#     int_95p_now = np.sort(clean_df(r95p_sing_now))[::-1]
    

#     freq_95p_per = (freq_95p_now/freq_95p_base -1) * 100
#     int_95p_per = (np.mean(int_95p_now[::min_freq])/np.mean(int_95p_base[::min_freq]) -1) * 100
    
#     # 99p
#     freq_99p_base = np.sum(clean_df(r99p_day_base))
#     freq_99p_now = np.sum(clean_df(r99p_day_now))
#     min_freq = int(np.min([freq_99p_base, freq_99p_now]))
    
#     int_99p_base = np.sort(clean_df(r99p_sing_base))[::-1]
#     int_99p_now = np.sort(clean_df(r99p_sing_now))[::-1]
    
#     freq_99p_per = (freq_99p_now/freq_99p_base -1) * 100
#     int_99p_per = (np.mean(int_99p_now[:min_freq])/np.mean(int_99p_base[:min_freq]) -1) * 100
    
#     freq_95p.append(freq_95p_per)
#     freq_99p.append(freq_99p_per)
#     int_95p.append(int_95p_per)
#     int_99p.append(int_99p_per)
    
#     print(freq_95p_per, int_95p_per, freq_99p_per, int_99p_per)

# axes[0].scatter(freq_95p, int_95p, marker = '8', s = 350, facecolors='#bd004e', alpha = 0.75)
# axes[1].scatter(freq_99p, int_99p, marker = 'X', s = 350, facecolors='red', alpha = 0.75)

# freq_95p = np.sort(freq_95p)
# int_95p = np.sort(int_95p)
# freq_99p = np.sort(freq_99p)
# int_99p = np.sort(int_99p)

# ax3 = axes[0].twinx()
# ax3.plot(freq_95p,sst.norm.pdf(freq_95p, np.mean(freq_95p), np.std(freq_95p)), color = '#bd004e')

# ax4 = axes[0].twiny()
# ax4.plot(sst.norm.pdf(int_95p, np.mean(int_95p), np.std(int_95p)), int_95p, color = '#bd004e')
# fig.savefig(os.path.join(img_wd, "ME_dom_pctl.png"), bbox_inches = 'tight', dpi = 300)


# # for i in range(4):
# #     for no_station in no_R_station:
        
