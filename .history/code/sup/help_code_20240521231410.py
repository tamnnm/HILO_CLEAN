# ------------------------------------------------------------------ #
#                             panda frame                            #
# ------------------------------------------------------------------ #

# Merge multiple pandaframe consecutively

import multiprocessing as mp
import numpy as np
from functools import reduce
# Assuming dfs is a list of your dataframes
merged_df = reduce(lambda left, right: pd.merge(
    left, right, on='timestamp', how='outer'), dfs)


# ------------------------------------------------------------------ #
#                               xarray                               #
# ------------------------------------------------------------------ #
# How to create a time coordinate from a reference date
reference_date = '1940-01-01 07:00:00.0'
ds['time'] = pd.date_range(
    start=reference_date, periods=ds.sizes['time'], freq="2H")

or this

dt_xr = dt_xr.reindex(time=pd.date_range(
    "1943-01-01", "1953-12-01", freq="MS"))

# ------------------------------------------------------------------ #
# rename variable
ds_swap = ds.swap_dims({"time_step": "valid_time"}).drop(
    ['time', 'time_step', 'step', 'number', 'surface'])
# ------------------------------------------------------------------ #
# save to netcdf
ds_var.to_netcdf(f'{Data_final}wrf_{var_uniform}.nc', 'w', engine="h5netcdf",
                 format='NETCDF4')

# ------------------------------- wrf ------------------------------ #
Error: AttributeError: 'CRS' object has no attribute 'dtype' when subsetting xarray dataset
# xwrf creates a pyproj CRS object, which is not supported by xarray.
# This helps convert the wrf projection to lat/lon but not serializable (cannot be subsetted)
Step 1: Drop the 'wrf_projection' variable
Step 2: Keep the projection object in a separate variable(maybe used for latter)
(!) Refer to this file: / home/tamnnm/.conda/envs/tamnnm/lib/python3.9/site-packages/my_junk

# ------------------------------------------------------------------ #
#                                numpy                               #
# ------------------------------------------------------------------ #
# vectorize mapping
# Generate random values
value = np.arange(0, 256)
# Example image
image = np.random.randint(0, 255, size=(100, 100))
# Vectorized approach
new_image = value[image]
print(new_image)
# ------------------------------------------------------------------ #
#                                 cdo                                #
# ------------------------------------------------------------------ #
cdo option = 'L' to prevent potential
cdo does not process variable as char
e.g.obs
-> should subset the file with just the variable before you are going to do something
e.g. delete the variable or just let it be, it will skip it automatically

# Set calendar to standard
cdo setcalendar, standard

# Reset time
cdo settaxis, [start_date], [start time e.g. 00:00:00], [freq e.g. 1day]

# ------------------------------------------------------------------ #
#                                index                               #
# ------------------------------------------------------------------ #


def std(x):
    std = np.std(x)
    return std


def res(x, y):
    res = sst.linregress(x, y)
    return res.rvalue


def RMSE(x, y):
    mse = mean_squared_error(x, y)
    rmse = math.sqrt(mse)
    return rmse


def CRMSD(x, y):
    crmsd = sm.centered_rms_dev(x, y)
    return crmsd


def ccoef(x, y):
    ccoef_full = np.corrcoef(x, y)
    ccoef = ccoef_full[0, 1]
    return ccoef


def norm(df):
    result = df.copy()
    max_value = df.iloc[:, -1].max()
    min_value = df.iloc[:, -1].min()
    for feature_name in df.columns:
        result[feature_name] = (
            df[feature_name] - min_value) / (max_value - min_value)
    return result

# ------------------------------------------------------------------ #
#                           Taylor diagram                           #
# ------------------------------------------------------------------ #


def taylor_plot(pra, dts):
    if pra != "pre":
        nP = 1
    else:
        nP = 2
    fig = plt.figure()
    if dts == "grid":
        subfold = "GRID"
        col = ["m", "#e6d800", "#00bfa0"]
        if pra == "t2m":
            no = 2
            label = ['CRU', 'UDel']
        else:
            no = 3
            label = ['CRU', 'UDel', "GHCN"]
    elif dts == "rean":
        subfold = "REAN"
        col = ["#4daf4a", "#ff7f00", "#e41a1c"]
        no = 3
        label = ['ERA', 'CERA', "20CR"]

    # append data into dts1,2,3 to load up sved,cef,rms for each dataset
    ds1 = [[1, 0, 0]]
    ds2 = [[1, 0, 0]]
    ds3 = [[1, 0, 0]]
    plot_legend = []

    for ct in city_real:
        sved, ccoef, crmsd = plot_ss(ct, pra, dts, "tl")
        for i in range(no):
            locals()[f'ds{i+1}'].append([sved[i], crmsd[i], ccoef[i]])
    # concat them, after this, each dataset will have 3 columns sved,cef,rms
    for i in range(no):
        dts_full = pd.DataFrame(locals()[f'ds{i+1}'])
        sdev = np.asarray(dts_full.iloc[:, 0])
        crmsd = np.asarray(dts_full.iloc[:, 1])
        ccoef = np.asarray(dts_full.iloc[:, 2])
        # print(sdev,crmsd,ccoef)
        if i == 0:
            sm.taylor_diagram(sdev, crmsd, ccoef,
                              markerSize=20,
                              numberPanels=nP,
                              markercolor=col[i],
                              # tickRMS = range(0,1,5), #rincRMS=0.5,
                              tickRMSangle=125.0,
                              colRMS='k',
                              styleRMS=':', widthRMS=2.0,
                              titleRMS='off',
                              # alpha = 0.0,
                              styleOBS='-', colOBS='r', markerobs='o',
                              widthOBS=1.5, titleOBS="OBS",
                              # tickSTD = range(0, 2, 10), axismax = 2.0,
                              colSTD='k', styleSTD='-.', widthSTD=1.0,  # rincSTD=1.0,
                              colCOR='k', styleCOR='--', widthCOR=1.0
                              )
        else:
            plot = sm.taylor_diagram(sdev, crmsd, ccoef,
                                     markerSize=20,
                                     overlay='on',
                                     # alpha = 0.0,
                                     markerColor=col[i]
                                     )
        plot = mlines.Line2D([], [], color=col[i], marker='o',
                             markersize=10, label=label[i], linestyle='None')
        # linestyle has to have to be 'None' not just None or it won't work
        plot_legend.append(plot)
    plt.legend(handles=plot_legend, loc='best')
    plt.tick_params(axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    top=False)   # ticks along the top edge are off
    plt.tick_params(axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    right=False      # ticks along the right edge are off
                    )
    plt.savefig(os.path.join(Data_path, "Data", "TL",
                subfold, f'{pra}.jpg'), format="jpg")


taylor_plot("t2m", "grid")
taylor_plot("t2m", "rean")
taylor_plot("pre", "grid")
taylor_plot("pre", "rean")
# ------------------------------------------------------------------ #
#                                 NCL                                #
# ------------------------------------------------------------------ #
# convert from julian to standard calendar
ncatted - a calendar, time, m, c, standard CERA20C_tp_1901_2010.nc
# ------------------------------------------------------------------ #
#                                panda                               #
# ------------------------------------------------------------------ #
# Why rename/set_index doesn't have any effect on pandaframe
-> Remember to put inplace = True like ds.set_index('test')
Or it will create a copy if you assign like this ds_new = ds.set_index('test')
# Apply
If you can, avoid at all cost. If you must iterate through each line, try to use lambda like this
mask = (ds['day'] < 1) | (ds['day'] > ds.apply(
    lambda x: monthrange(x['year'], x['month'])[1], axis=1))
# Create an multi-dimensional array/dataset from dataframe
->Make it multi-index, extract
test_ds = ds.set_index(['time', 'no_station']).loc[[var]]


# ------------------------------------------------------------------ #
#                              Function                              #
# ------------------------------------------------------------------ #

def func(arg: List, kwargs: Dict)


*args, **kwargs


# How to use map
xs = [1, 3, 5]


def calc_y(an_x):
    return an_x + 1


ys = list(map(calc_y, xs))
# or
ys = list(map(lambda x: x + 1, xs))

# ------------------------------------------------------------------ #
#                                Tool                                #
# ------------------------------------------------------------------ #

#  List/ merge multiple sequential elements
"""
The functools.reduce function is a powerful tool in Python that applies a binary function(a function that takes two arguments) to all items in an input list in a cumulative way.
This means it takes the first two items and applies the function to them, then feeds the result and the third item back into the function, and so on.
It reduces the list to a single output."""


numbers = [1, 2, 3, 4, 5]
sum = reduce(lambda x, y: x + y, numbers)

print(sum)  # Output: 15

# ------------------------------------------------------------------ #
#                          Multi-processing                          #
# ------------------------------------------------------------------ #


def ind_fu():
    with con_fu.ProcessPoolExecutor() as executor:
        futures = []
        # Reanalysis
        for ind in rain_tuple:  # + temp_tuple:
            for rean_name in list(rean_year_dict.keys()):
                futures += [executor.submit(gen_class,
                                            ind=ind, dataset=rean_name)]
        # Observation
            # futures += [executor.submit(gen_class, ind=ind, dataset='obs')]
    con_fu.wait(futures)
    print("Finish creating indices file")

# Another way to do it


def ind_fu(args):
    # # Reanalysis
    # for ind in rain_tuple:  # + temp_tuple:
    #     for rean_name in list(rean_year_dict.keys()):
    #         gen_class(ind=ind, dataset=rean_name)
    # # Observation
    #     gen_class(ind=ind, dataset='obs')
    # print("Finish creating indices file")
    gen = gen_ind()
    gen(*args)


pool = mp.Pool(mp.cpu_count())
args = [(ind, dataset)
        for ind in rain_tuple for dataset in list(rean_year_dict.keys())]
pool.map(ind_fu, args)
pool.close()
pool.join()
