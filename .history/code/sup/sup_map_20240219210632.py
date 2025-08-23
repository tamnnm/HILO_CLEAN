#%%
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#import metpy.calc as mpcalc
#from metpy.units import units
import numpy as np
import pandas as pd
import shapefile as shp
import os
from HersheyFonts import HersheyFonts
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%%
params = {
	'axes.labelsize': 5,
	'font.size': 5,
    'font.family':'monospace',
	'legend.fontsize': 5,
    'legend.loc': 'upper right',
    'legend.labelspacing':0.25,
	'xtick.labelsize': 7,
	'ytick.labelsize': 7,
	'lines.linewidth': 4,
	'text.usetex': False,
	# 'figure.autolayout': True,
	'ytick.right': True,
	'xtick.top': True,

	'figure.figsize': [10, 10], # instead of 4.5, 4.5
	'axes.linewidth': 1.5,

	'xtick.major.size': 2,
	'ytick.major.size': 2,
	'xtick.minor.size': 2,
	'ytick.minor.size': 2,

	'xtick.major.width': 2,
	'ytick.major.width': 2,
	'xtick.minor.width': 2,
	'ytick.minor.width': 2,

	'xtick.major.pad': 5,
	'ytick.major.pad': 5,
	#'xtick.minor.pad': 14,
	#'ytick.minor.pad': 14,

	'xtick.direction': 'inout',
	'ytick.direction': 'inout',
   }
plt.clf()
#%%
matplotlib.rcParams.update(params)
map_pro = ccrs.PlateCarree()
Data_path = "/work/users/tamnnm/code/main_code/city_list_obs/city/"
os.chdir(Data_path)
city_name = []
city_start_year = []
city_end_year = []
city_lat = []
city_lon = []

#%%

shp_path = "/work/users/tamnnm/geo_info/vnm/full_shp/vnm_admbnda_adm0_gov_20200103.shp"
vnmap = shp.Reader(shp_path)
var_long_name = {"rain": "Precipitation",
                 "vwnd": "V wind",
                 "uwnd": "U wind",
                 "temp": "Temp"}
#number_file=len(os.listdir(Data_path))
number_file = 4

for i in range(4,2,-1):
    if number_file%i == 0:
        if number_file <= 4:
            i_new = 2
        else:
            i_new = i
        no_ver = int(i_new)
        no_hor = int(number_file/i_new)
    else:
        continue
fig,axs = plt.subplots(no_hor,no_ver,subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0})
#sharex, sharey so that the suplot use the same x-bar or y-bar. The gridspec_kw set to 0 so that it delete the space between the suplots
for city_path,ax in zip(os.listdir(Data_path),axs.flatten()):
    """
    #Create the outer frame
    ax.set_extent([100,120,5,25])
    ax.xlabels_top = False
    ax.ylabels_right = False
    #ax.coastlines(color='black', linewidth=1, resolution='10m',alpha=0.5)
    #ax.add_feature(cfeature.BORDERS.with_scale('10m'),linestyle='-')
    ax.set_xticks(np.linspace(100, 120,5), crs=map_pro)
    ax.set_yticks(np.linspace(5,25,5), crs=map_pro)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both',which="minor",labelsize=5)
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    """
    #number of station acc
    station_1=0
    station_2=0
    station_3=0
    var_name=city_path.split('.')[0]
    #plot vietnam and the station point
    txt_shapes = []
    for vnmapshape in vnmap.shapeRecords():
        listx=[]
        listy=[]
        # parts contains end index of each shape part
        parts_endidx = vnmapshape.shape.parts.tolist()
        parts_endidx.append(len(vnmapshape.shape.points) - 1)
        for i in range(len(vnmapshape.shape.points)):
            x, y = vnmapshape.shape.points[i]
            if i in parts_endidx:
                # we reached end of part/start new part
                txt_shapes.append([listx,listy])
                listx = [x]
                listy = [y]
            else:
                # not end of part
                listx.append(x)
                listy.append(y)

#"""
    for zone in txt_shapes:
        x,y = zone
        #Plot only the border
        ax.plot(x,y,color="k",markersize=10e-6,linewidth=0.4)
        #Fill the inside with a certainc color
        #ax.fill(x,y,facecolor='red',edgecolor="k",linewidth=0.4)
    ax.text(112, 10, "Spratly Islands", fontsize=7,rotation=45)
    ax.text(111, 17.5, "Paracel Islands", fontsize=7)
    ax.text(110, 13, "EAST SEA", fontsize=7,rotation=90,alpha=0.6)
    #ax.text(102,20 , "Laos", color="w", fontsize=16)
    #ax.text(108,13 , "Vietnam",color="w", fontsize=16,rotation=90)
    #ax.text(104, 13, "Cambodia",color="w", fontsize=16)
    ax.grid(linewidth=1, color='gray', alpha=0.5,
                    linestyle='--')
    ax.set_title(var_long_name[var_name.split("_")[0]],color='black',fontsize=17)
    ax.text(111,23,"Stating year",fontsize=7)
#"""

    with open(f'{city_path}','r') as file:
        for i,line in enumerate(file):
            city_info=line.split(',')
            print(city_info)
            city_lon=float(city_info[3])
            city_lat=float(city_info[4])
            city_start_year=float(city_info[1])
            #print(city_lon[0])
            if city_start_year<=1961:
                station_1+=1
                ax.plot(city_lon, city_lat,  markersize=2, marker='o', color='red')
                #ax.text(115, city_lat,f'{city_info[0]}_{city_info[1]}_{city_info[2]}',color='red',fontsize=5)
                #ax.plot([city_lon,city_lon+5],[city_lat, city_lat], linewidth=1, color='r')
            elif 1961<=city_start_year<=1986:
                ax.plot(city_lon, city_lat,  markersize=2, marker='o', color='orange')
                station_2+=1
            else:
                ax.plot(city_lon, city_lat,  markersize=2, marker='o', color='black')
                station_3+=1
                #continue
    weight = station_1,station_2,station_3
    """
    weight_count = {
        "1961": int(station_1),
        "1986": int(station_2),
        "2019": int(station_3)
    }
    """
    name = ['-61','61-86','86-19']
    color = ['r','orange','black']
    ins = ax.inset_axes([0.65,0.7,0.3,0.2])
    p = ins.bar([1,2,3], weight, color = color, tick_label = name, width=0.5, bottom=1)

fig.suptitle("Station distribution in Vietnam",fontsize=20, y=0.995)
#fig.tight_layout()
fig.savefig('/work/users/tamnnm/code/sup/image/map_ob.jpg',format="jpg",dpi=1000)
#"""

#%%
#print("parts",vnmapshape.shape.points)
#print("listx",listx,'\n',"listy",listy)

