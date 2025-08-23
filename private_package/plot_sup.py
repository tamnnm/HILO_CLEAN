# ----------------------- Import plot module ----------------------- #
import time
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, from_levels_and_colors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.axes as maxes
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from typing import Optional, List
from my_junk import *
from geopy.distance import distance as geodistance
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep
# from shapely.vectorized import contains
import xarray as xr
import concurrent.futures as cf

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
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
import matplotlib
import numpy as np
import os
import multiprocessing

import wradlib as wrl

# -------------------- set up background border -------------------- #

params = {
    'axes.titlesize': 40,
    'axes.labelsize': 23,
    'axes.labelpad': 15,
    'font.size': 50,
    'font.family': 'cmss10',
    'mathtext.fontset': 'stixsans',
    'legend.fontsize': 30,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'lines.linewidth': 3,
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

<<<<<<< HEAD
    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'axes.unicode_minus': False,
}

def ini_plot(dlat, ulat, dlon, ulon, pad, figsize = [12, 8], label = False, label_size = 10, ax_num=111,title="", params = params, coastline = True, border = True, grid = False, land = False, ocean = False):
    
    
    plt.clf()
    matplotlib.rcParams.update(params)
    
    map_pro = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.subplot(ax_num, projection=map_pro)
    
=======
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',

    'savefig.transparent': False,
    'savefig.bbox': 'tight',
    'savefig.dpi': 500,


    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'axes.unicode_minus': False,

}

def ini_plot(dlat, ulat, dlon, ulon, pad, figsize = [12, 8], label = False, label_size = 10, ax_num=111,title="", params = params, coastline = True, border = True, grid = False, land = False, ocean = False):


    plt.clf()
    matplotlib.rcParams.update(params)

    map_pro = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.subplot(ax_num, projection=map_pro)

>>>>>>> c80f4457 (First commit)
    # Set the extent (bounding box) for the map
    ax.set_extent([dlon, ulon, dlat, ulat], crs=map_pro)
    ax.set_xticks(np.arange(dlon, ulon + pad, pad), crs=map_pro)
    ax.set_yticks(np.arange(dlat, ulat + pad, pad), crs=map_pro)
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
<<<<<<< HEAD
    
        
    # Tick for axis
    ax.tick_params(axis='both', bottom=False, left=False, top=False, right=False, labelsize=label_size)
    # change fontsize of tick when have major
    
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    
    if label:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    # Add grid lines
    if grid: ax.grid(True, which='both', linestyle='--', linewidth=1, color = 'black', zorder=2)
    
    # Add title
    ax.set_title(title, pad = 30)
    
=======


    # Tick for axis
    ax.tick_params(axis='both', bottom=False, left=False, top=False, right=False, labelsize=label_size)
    # change fontsize of tick when have major

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())

    if label:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # Add grid lines
    if grid: ax.grid(True, which='both', linestyle='--', linewidth=1, color = 'black', zorder=2)

    # Add title
    ax.set_title(title, pad = 30)

>>>>>>> c80f4457 (First commit)
    # Draw border, coasral line,...
    if coastline: ax.coastlines(resolution='10m',zorder=3)
    if border: ax.add_feature(cfeature.BORDERS.with_scale('10m'),zorder=3)
    if land: ax.add_feature(cfeature.LAND,)#, facecolor=cfeature.COLORS["land_alt1"])
    if ocean: ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])

    return fig, ax

def make_cbar(ax, cmap, sm=None, norm=None, ticks=[], labels=[], custom_cbar=True):

    divider = make_axes_locatable(ax)
    ax.set_aspect('equal')
    # Extract the projection from the existing axis
    # projection = ax.projection if hasattr(ax, 'projection') else None
    if sm is None:
        if norm is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])

    if custom_cbar == True:
        cax = divider.append_axes(
            "bottom", size='5%', pad=0.6, axes_class=maxes.Axes)
        cbar = plt.colorbar(
            sm, ax=ax, cax=cax, orientation='horizontal', ticks=ticks, drawedges=True)

        cbar.ax.set_xticks([tick + 0.5 for tick in ticks])
        cbar.ax.set_xticklabels(labels)

        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(2.5)

        cbar.dividers.set_color('white')
        cbar.dividers.set_linewidth(2.5)
        cbar.ax.tick_params(size=0,  labelsize=8, pad=-10)
        cbar.ax.xaxis.set_ticks_position('bottom')
    else:
        cax = divider.append_axes(
            "bottom", size='5%', pad=0.6, axes_class=maxes.Axes)

        if len(ticks) == 0:
            cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='horizontal')
        else:
            cbar = plt.colorbar(
                sm, ax=ax, cax=cax, orientation='horizontal', format='%.3f', ticks=ticks)

        if len(labels) != 0:
            cbar.set_ticklabels(labels)

        cbar.ax.tick_params(size=5,  labelsize=8)
        cbar.ax.xaxis.set_ticks_position('bottom')
<<<<<<< HEAD
=======
    return cbar
>>>>>>> c80f4457 (First commit)


vietnam_shapefile = gpd.read_file("/work/users/tamnnm/geo_info/vnm/full_shp/vnm_admbnda_adm0_gov_20200103.shp")
vietnam_province_shapefile = gpd.read_file("/work/users/tamnnm/geo_info/vnm/full_shp/vnm_admbnda_adm1_gov_20201027.shp")

# Function to check if a point is in Vietnam
def in_vietnam_point(lat, lon):
    point = Point(lon, lat)
    return vietnam_shapefile.contains(point).any()

def in_shapefile_point(lat, lon, shapefile):
    point = Point(lon, lat)
    return shapefile.contains(point).any()

<<<<<<< HEAD
def border_plot(ax, geometry):
=======
def border_plot(ax, geometry = vietnam_shapefile.geometry):
>>>>>>> c80f4457 (First commit)
    """Plot borders for either single polygons, multipolygons, or GeoSeries"""
    if isinstance(geometry, gpd.GeoSeries):
        # Handle GeoSeries
        for geom in geometry:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                ax.plot(x, y, color='black', linewidth=1, alpha=0.5)
            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:
                    x, y = polygon.exterior.xy
                    ax.plot(x, y, color='black', linewidth=1, alpha=0.5)
    elif isinstance(geometry, (Polygon, MultiPolygon)):
        # Handle single Polygon or MultiPolygon
        if isinstance(geometry, Polygon):
            x, y = geometry.exterior.xy
            ax.plot(x, y, color='black', linewidth=1, alpha=0.5)
        else:  # MultiPolygon
            for polygon in geometry.geoms:
                x, y = polygon.exterior.xy
                ax.plot(x, y, color='black', linewidth=1, alpha=0.5)
    return

<<<<<<< HEAD
# ----------------------- Create custom mask ----------------------- #

# Create mask for points in a mask
def mask_create(geometry, lon_array, lat_array):
    
    # Default values for lon and lat arrays
    shape = lon_array.shape
    
=======
def read_ll_file(filename):
    """Read .ll file and return coordinates as a polygon"""
    coordinates = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                lon, lat = map(float, line.strip().split())
                coordinates.append((lon, lat))
    return Polygon(coordinates)

# ----------------------- Create custom mask ----------------------- #

# Create mask for points in a mask
def mask_create(lon_array, lat_array, output_df = False, geometry = vietnam_shapefile.geometry,  reversed = False, **kwargs):
    """_summary_

    Args:
        lon_array: Array of longitude values.
        lat_array: Array of latitude values.
        geometry: Geometry to use for masking. Defaults to vietnam_shapefile.geometry.
        reversed (bool): True = outside is True, False = inside is True.
                         Defaults to False (True is preferred when plot DEM).
        output_df (bool): Whether to save the mask to a file. Defaults to False.
        **kwargs: Additional arguments. If save=True, save_path must be provided.

    Returns:
        If output_df is False:
            numpy.ndarray: Boolean mask array with the same shape as input coordinates.
        If output_df is True:
            xarray.Dataset: Dataset containing the mask array and coordinates.
    """


    # Default values for lon and lat arrays

>>>>>>> c80f4457 (First commit)
    import timeit
    # ------------- Transform 1D coordinates to 2D meshgrid ------------ #
    if len(lon_array.shape) == 1 and len(lat_array.shape) == 1:
        shape = (len(lon_array), len(lat_array))
<<<<<<< HEAD
        lon_array,lat_array= np.meshgrid(lon_array,lat_array)
    
    print('Test 3')
    
    # Create points from lon and lat arrays
    # Size: 3200 x 1600
    
    # Method 1: 142s
    def create_point(x):
        return Point(x)
    
    coords = list(zip(lon_array.flatten(), lat_array.flatten()))
    start = timeit.default_timer()
    points = list(map(create_point, coords))
    # points = gpd.GeoSeries(list(map(create_point, coords)))
    end = timeit.default_timer()
    print('Time to create points:', end - start)
    
=======
        lon_array,lat_array= np.meshgrid(lon_array,lat_array) # Shape: (lat, lon)

    shape = lon_array.shape
    # Create points from lon and lat arrays
    # Size: 3200 x 1600

    # Method 1: 142s
    def create_point(x):
        return Point(x)

>>>>>>> c80f4457 (First commit)
    # Method 2: 130s
    # start = timeit.default_timer()
    # points = gpd.points_from_xy(lon_array.flatten()[:10], lat_array.flatten()[:10])
    # end = timeit.default_timer()
    # print('Time to create points:', end - start)
<<<<<<< HEAD
    
=======

>>>>>>> c80f4457 (First commit)
    # ------------------------ Create mask array ----------------------- #
    # for polygon in polygons:
    #     mask_array = points.within(polygon)
    #     print('test 4.1')
    #     if len(polygons) > 1:
    #         mask_all_array.append(mask_array)
<<<<<<< HEAD
    
    
=======


>>>>>>> c80f4457 (First commit)
    if isinstance(geometry, gpd.GeoSeries):
        multi_poly = geometry.unary_union
    else:
        multi_poly = geometry
<<<<<<< HEAD
    
    pre_polygons = prep(multi_poly)
    
    
    # METHOD 3: Super large image => Need chunking
    print( 'test 4')
    
    # Create chunks for parallel processing
    # chunk_size = 10000
    # n_chunks = len(lon_flat) // chunk_size + (1 if len(lon_flat) % chunk_size else 0)
    # chunks = [(lon_flat[i:i + chunk_size], lat_flat[i:i + chunk_size])
    #           for i in range(0, len(lon_flat), chunk_size)]
    
    # def process_chunk(chunk):
    #     lon_chunk, lat_chunk = chunk
    #     points = [Point(x, y) for x, y in zip(lon_chunk, lat_chunk)]
    #     return [point.within(prep_polygons) for point in points]

    # start = timeit.default_timer()
    
    # # Use multiprocessing to process chunks in parallel
    # with multiprocessing.Pool() as pool:
    #     results = pool.map(process_chunk, chunks)
    
    # # Combine results
    # mask_array = np.concatenate(results)
    
    # end = timeit.default_timer()
    # print(f'Time to process {len(lon_flat)} points: {end - start:.2f} seconds')
    
    start = timeit.default_timer()
    mask_array = [pre_polygons.contains(point) for point in points]
    end = timeit.default_timer()
    print('Time to process points:', end - start)
    
    return np.logical_not(mask_array).reshape(shape)

def dem_plot(ax, mask, geometry):
=======

    pre_polygons = prep(multi_poly)

    minx, miny, maxx, maxy = multi_poly.bounds
    lon_flat = lon_array.ravel()
    lat_flat = lat_array.ravel()

    start = timeit.default_timer()

    in_bbox = (lon_flat >= minx) & (lon_flat <= maxx) & (lat_flat >= miny) & (lat_flat <= maxy)

    # Only check points within bbox
    mask = np.zeros(len(lon_flat), dtype=bool)
    points_to_check = np.where(in_bbox)[0]

    # Process in chunks to avoid memory issues
    chunk_size = 100000
    for i in range(0, len(points_to_check), chunk_size):
        chunk = points_to_check[i:i+chunk_size]
        # Vectorized point creation and checking
        points = [Point(lon_flat[j], lat_flat[j]) for j in chunk]
        mask[chunk] = [pre_polygons.contains(p) for p in points]

    mask = mask.reshape(shape)
    if reversed:
        mask = ~mask

    # coords = list(zip(lon_array.flatten(), lat_array.flatten()))

    # start = timeit.default_timer()
    # points = list(map(create_point, coords))
    # # points = gpd.GeoSeries(list(map(create_point, coords)))
    # end = timeit.default_timer()
    # print('Time to create points:', end - start)

    # mask_array = []
    # for point in points:
    #     if pre_polygons.contains(point): print(point)
    #     mask_array += [pre_polygons.contains(point)]
    # end = timeit.default_timer()
    # print('Time to process points:', end - start)


    if 'save_path' not in kwargs:
        raise ValueError("If save=True, you must provide save_path in kwargs")

    save_path = kwargs['save_path']  # Extract save_path from kwargs
    # Save the mask as a netCDF file
    mask_df = xr.Dataset(data_vars={
        'mask': (['lat', 'lon'], mask)
    }, coords={
        'lon': lon_array[0, :],
        'lat': lat_array[:, 0],
    })
    print("Saving mask to:", save_path)
    mask_df.to_netcdf(save_path)

    return mask_df if output_df else mask

def dem_plot(ax, mask, geometry = vietnam_shapefile.geometry):
>>>>>>> c80f4457 (First commit)
    # Set this in bashrc or load-env.sh please
    filename = wrl.util.get_wradlib_data_file("geo/vietnam.tif")
    ds = wrl.io.open_raster(filename)
    # pixel_spacing is in output units (lonlat)
    ds = wrl.georef.reproject_raster_dataset(ds, spacing=0.005)
    rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds)
    if mask == True:
        if os.path.exists('/work/users/tamnnm/geo_info/vnm/vnm_dem.nc'):
            dem_df = xr.open_mfdataset('/work/users/tamnnm/geo_info/vnm/vnm_dem.nc')
            mask = dem_df['mask'].values
        else:
<<<<<<< HEAD
            mask = mask_create(geometry, lon_array = rastercoords[..., 0], lat_array = rastercoords[..., 1])
            dem_df = xr.Dataset(data_vars={'dem': (('x', 'y'), rastervalues), 'mask': (('x','y'), mask)}, coords={'x': rastercoords[..., 0][:, 0], 'y': rastercoords[..., 1][0, :]})
            dem_df.to_netcdf('/work/users/tamnnm/geo_info/vnm/vnm_dem.nc')
=======
            mask = mask_create(lon_array = rastercoords[..., 0], lat_array = rastercoords[..., 1], save_path='/work/users/tamnnm/geo_info/vnm/vnm_dem.nc', reversed=True)
>>>>>>> c80f4457 (First commit)
        rastervalues = np.ma.masked_array(rastervalues, mask)

    # specify kwargs for plotting, using terrain colormap and LogNorm
    dem = ax.pcolormesh(
        rastercoords[..., 0],
        rastercoords[..., 1],
        rastervalues,
<<<<<<< HEAD
        cmap=plt.cm.terrain,
        norm=LogNorm(vmin=1, vmax=3000),
    )
    
=======
        cmap='gist_earth',
        norm=LogNorm(vmin=1, vmax=3000),
    )

>>>>>>> c80f4457 (First commit)
    #Create an exact control of the colorbar and vertical
    plt.gcf().subplots_adjust(left = 0.1) #? Adjust the position of the colorbar
    cax1 = plt.gcf().add_axes([0.37,
                              ax.get_position().y0+0.05,
                              0.02,
                              ax.get_position().height*0.8])
    # add colorbar and title
    # we use LogLocator for colorbar
    cb = plt.gcf().colorbar(dem, cax=cax1, ticks=ticker.LogLocator(subs=range(10)))
    cb.set_label("terrain height [m]")
    # Move ticks and ticklabels to left side
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    # cbar = plt.colorbar(dem, ax=ax, orientation='horizontal', pad=0.02, aspect=50)
    # cbar.set_label("terrain height [m]")
    # cbar.locator = ticker.LogLocator(subs=range(10))
    # cbar.update_ticks()
<<<<<<< HEAD
    
    return dem

# -------------------- Plot shp ---------------------------- #
def plot_shp(ax, border=True, dem = True, mask=True,  shp_map = vietnam_shapefile):
    geometry = shp_map.geometry
    if border: border_plot(ax, geometry)
    if dem: dem_plot(ax, mask, geometry)
    return
=======

    return dem

def KV_plot(ax, KV_fill_color = True, KV_label = True, font_scale = 1):
    # Define region files and colors
    KV_path ="/work/users/tamnnm/geo_info/shp_7KV"
    region_files = {
        'TBB': 'B1.ll',
        'DBB': 'B2.ll',
        'DBBB': 'B3.ll',
        'BTB': 'B4.ll',
        'TN': 'N1.ll',
        'NTB': 'N2.ll',
        'NB': 'N3.ll'
    }

    custom_name = {
        'TBB': 'R1',
        'DBB': 'R2',
        'DBBB': 'R3',
        'BTB': 'R4',
        'NTB': 'R5',
        'TN': 'R6',
        'NB': 'R7'
    }

    colors = {
            'TBB': 'lightblue',
            'DBB': 'lightgreen',
            'DBBB': 'lightpink',
            'BTB': 'lightyellow',
            'TN': 'lightgray',
            'NTB': 'lightcoral',
            'NB': 'lightsalmon'
        }
    all_regions = None
    region_gdfs = {}
    for region, ll_file in region_files.items():
        # Read polygon
        polygon = read_ll_file(os.path.join(KV_path, ll_file))

        # GeoDataFrame
        region_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        # Plot region with fill color and black border

        region_gdfs[region] = region_gdf
        if all_regions is None:
                all_regions = region_gdf
        else:
            all_regions = pd.concat([all_regions, region_gdf])

    union = all_regions.unary_union
    union_bound = union.boundary
    for region, region_gdf in region_gdfs.items():
        if KV_fill_color:
            region_gdf.plot(ax=ax, facecolor=colors[region],
                    edgecolor='black', alpha=0.3, linewidth=2)
        else:
            for geom in region_gdf.geometry:
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                elif geom.geom_type == 'MultiPolygon':
                    coords = []
                    for part in geom:
                        coords.extend(list(part.exterior.coords))

                # Extract x and y coordinates
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]

                # Plot boundary with white outline first
                ax.plot(x_coords, y_coords, color='white', linewidth=2.5, zorder=1)
                # Plot thinner black line on top
                ax.plot(x_coords, y_coords, color='black', linewidth=1, zorder=2)
        if KV_label:
        # Optionally add region labels
            # Get centroid in original CRS
            region_gdf_utm = region_gdf.to_crs("EPSG:32648")
            centroid_utm = region_gdf_utm.geometry.centroid
            # Convert to CRS
            centroid_wgs84 = gpd.GeoDataFrame(geometry=centroid_utm, crs="EPSG:32648").to_crs("EPSG:4326")
            centroid = centroid_wgs84.geometry.iloc[0]
            ax.text(centroid.x if region != "NTB" else centroid.x + 0.1, centroid.y,  custom_name[region],
                    ha='center', va='center', fontsize=15,
                    color='red',
                    fontweight='bold',
                    #path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                    zorder = 1000)
    return ax

# -------------------- Plot shp ---------------------------- #
def plot_shp(ax, border=True, dem = True, mask=True,  shp_map = vietnam_shapefile,
             island_only = False, island_full = False, island_subplot = False,
             extend = [101, 110, 7, 25], axis = 'off', tick = False,
             KV = False, KV_fill_color = True, KV_label = True, font_scale = 1):
    geometry = shp_map.geometry

    if border: border_plot(ax, geometry)
    if dem:
        dem_plot(ax, mask, geometry)
        if KV and KV_fill_color:
            KV_fill_color = False


    # Extend
    if extend != [101, 110, 7, 25]:
        extend = extend
    else:
        if island_subplot: extend = [110.5, 116, 7, 25]
        if island_only or island_full: extend = [101, 116, 7, 25]
    ax.set_extent(extend)

    if island_full or island_subplot:
        ax.text(112, 7.3, "Spratly Islands", fontsize=5*font_scale, rotation=0, alpha = 0.8, color = 'k')
        ax.text(111, 15, "Paracel Islands", fontsize=5*font_scale, rotation=0, alpha = 0.8, color = 'k')
        ax.text(110, 10, "E a s t  V N  S e a", fontsize=5*font_scale, rotation = 90, fontstretch = 'expanded', alpha = 0.8, color = 'k')

    # Frame
    spline_style(ax)
    ax.spines['geo'].set_visible(False)

    if axis != 'off':
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

    elif island_subplot:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    elif island_only:
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

    if tick == False:
        ax.tick_params(bottom=False, top=False, left=False, right=False,
              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    if KV:
        KV_plot(ax, KV_fill_color, KV_label, font_scale)
    return

def spline_style(ax):
    """Apply custom styling to an axes."""
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(2)
        spine.set_linestyle((0, (16, 16)))
    return ax
>>>>>>> c80f4457 (First commit)
