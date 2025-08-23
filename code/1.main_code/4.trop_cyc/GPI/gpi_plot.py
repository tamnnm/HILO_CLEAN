from my_junk import ini_plot, make_cbar
import xarray as xr
# ----------------------- Import plot module ----------------------- #
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
# import metpy.calc as mpcalc
# from metpy.units import units
import shapefile as shp
import matplotlib
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
import matplotlib
import numpy as np
import os
from my_junk import *

# gpi_path = "/work/users/tamnnm/code/1.main_code/4.trop_cyc/GPI/"
gpi_path = "/data/projects/REMOSAT/tamnnm/iwtrc/ASEAN/grid_1.0/"
img_path = "/work/users/tamnnm/code/2.sup/image/"
gpi_file = "vort.1881.nc"
gpi_2010 = "GPI2010.1881.nc"

gpi_DA = xr.open_dataarray(gpi_path+gpi_file)
gpi_DA = cut_time(gpi_DA, start_date="1881-09-29",
                  end_date="1881-10-04", full=True, data=False)
time_dim = gpi_DA.time.values
for time_step in time_dim:
    plt.clf()
    plt.close()
    data = gpi_DA.sel(time=time_step)*1e6
    fig, ax = ini_plot(dlat=-5, ulat=40, dlon=100,
                       ulon=150, pad=5, title="test")
    min_value = np.nanmin(data.values)
    max_value = np.nanmax(data.values)
    print(min_value, max_value)
    print(data)
    # , cmap='RdYlBu', vmin=min_value, vmax=max_value)
    contourf = ax.contourf(data, cmap = )
    
    ax.set_title("")
    # ax.set_xlabel("")
    # ax.set_ylabel("GPI")

    # make_cbar(ax, cmap='winter', sm=c, custom_cbar=False)
    fig.tight_layout()
    fig.savefig(f"img/re_vort_{time_step}.png", dpi=300)
    break
