# Referenced from https://github.com/wy2136/TCI
import xarray as xr
import numpy as np
import os
from my_junk import *

from xtci import entropy_deficit, potential_intensity as pi, wind_shear as ws, absolute_vorticity_vort as av, absolute_vorticity_wind as avw


def gpi_cal(sst, slp, prs, T, rhum, shum, rhum2m, t2m, uwnd=None, vwnd=None, vort=None, lat=None, land_mask=None, ocean_flag=True, odir=None, year=0):
    '''calculate TC indices (e.g. GPI, VI) and related variables given FLOR/HiRAM atmos_month output'''

    if odir is None:
        odir = './'

    if land_mask is None:
        ocean_flag = False

    if rhum is None and shum is None:
        raise ValueError('rhum OR shum must be provided')
    # ds = xr.open_dataset(ifile)
    # is_ocean = ds.land_mask.load() < 0.1

    # ! Use only for VI (not related to GPI)
    # region - entropy deficit: (s_m_star - s_m)/(s_sst_star - s_b)
#     print('entropy deficit')
#     dname = 'chi'
#     ofile = os.path.join(odir,  f'{dname}.{year}.nc')
#     if os.path.exists(ofile):
#         chi = xr.open_dataset(ofile)[dname]
#         print('[opened]:', ofile)
#     else:
#         p_m = 600*100  # Pa
#         chi = entropy_deficit(
#             sst=sst,
#             slp=slp,
#             Tb=t2m,
#             RHb=RH2m/100,
#             p_m=p_m,
#             Tm=Ta.sel(level=p_m/100).drop('level'),
#             RHm=RH.sel(level=p_m/100).drop('level')/100
#         ).where(is_ocean)
#         chi.to_dataset(name=dname) \
#             .to_netcdf(ofile, t
#                        encoding={dname: {'dtype': 'float32',
#                                          'zlib': True, 'complevel': 1}},
#                        unlimited_dims='time')
#         print('[saved]:', ofile)
# endregion
    def save_nc(data, ofile, dname):
        if type(data) is xr.DataArray:
            data = xr.Dataset({dname: data})
        encoding = {dname: {'dtype': 'float32', 'zlib': True, 'complevel': 5}}
        print(data)
        data.to_netcdf(ofile, encoding=encoding, unlimited_dims=['time'])
        return

     # entropy deficit for GPI2010: (s_b - s_m)/(s_sst_star - s_b)
    print('entropy deficit for GPI2010')
    dname = 'chi_sb'
    ofile = os.path.join(odir, f'{dname}.{year}.nc')
    if os.path.exists(ofile):
        chi_sb = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        p_m = 600*100  # Pa
        chi_sb = entropy_deficit(
            sst=sst,
            slp=slp,
            Tb=t2m,
            RHb=rhum2m/100,
            p_m=p_m,
            Tm=cut_level(T, p_m/100),
            RHm=cut_level(rhum, p_m/100)/100,
            forGPI2010=True
        )
        if ocean_flag:
            chi_sb = chi_sb.where(land_mask)

        save_nc(chi_sb, ofile, dname)
        print('[saved]:', ofile)

    # region -  ventilation index: Vshear * chi_m /V_PI
    # print('ventilation index')
    # dname = 'VI'
    # ofile = os.path.join(odir,  f'{dname}.{year}.nc' )
    # if os.path.exists(ofile):
    #     VI = xr.open_dataset(ofile)[dname]
    #     print('[opened]:', ofile)
    # else:
    #     VI = Vshear*chi/PI.vmax.pipe(lambda x: x.where(x>0))
    #     VI.to_dataset(name=dname) \
    #         .to_netcdf(ofile,
    #             encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
    #             unlimited_dims='time')
    #     print('[saved]:', ofile)
    # endregion

    # potential intensity
    print('potential intensity')
    ofile = os.path.join(odir, 'PI.nc')
    if os.path.exists(ofile):
        PI = xr.open_dataset(ofile)
        print('[opened]:', ofile)
    else:
        if ocean_flag:
            PI = pi(
                sst=sst.where(land_mask),
                slp=slp.where(land_mask)*100,
                prs=prs,
                T=T.where(land_mask),
                q=shum.where(land_mask),
                dim_z='level',
                ptop=100  # ? Special for this case
            )
        else:
            PI = pi(
                sst=sst,
                slp=slp*100,
                prs=prs,
                T=T,
                q=shum,
                dim_z='level',
                ptop=100
            )
            raise KeyboardInterrupt
        encoding = {dname: {'dtype': 'float32', 'zlib': True, 'complevel': 5}
                    for dname in ('pmin', 'vmax')}
        encoding['iflag'] = {'dtype': 'int32'}
        PI.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
        print('[saved]:', ofile)

    # wind shear: ( (u200-u850)**2 + (v200-v850)**2 )**0.5
    print('wind shear')
    dname = 'Vshear'
    ofile = os.path.join(odir, f'{dname}.{year}.nc')
    if os.path.exists(ofile):
        Vshear = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        Vshear = ws(
            u850=cut_level(uwnd, 850),
            v850=cut_level(vwnd, 850),
            u200=cut_level(uwnd, 200),
            v200=cut_level(vwnd, 200)
        )
        if ocean_flag:
            Vshear = Vshear.where(land_mask)

        save_nc(Vshear, ofile, dname)
        print('[saved]:', ofile)

    # absolute vorticity at 850hPa
    print('absolute vorticity')
    dname = 'abs_vort'
    vname = 'vort'
    ofile_850 = os.path.join(odir, f'{dname}_850.{year}.nc')
    ofile = os.path.join(odir, f'{dname}.{year}.nc')
    if os.path.exists(ofile_850):
        eta = xr.open_dataset(ofile_850)[vname]
        print('[opened]:', ofile)
    elif os.path.exists(ofile):
        eta_full = xr.open_dataset(ofile)[vname]
        eta = cut_level(eta_full, 850)
        print('[opened]:', ofile)
    else:
        if vort is not None:
            eta = av(
                vort850=vort,
                lat=lat
            )
        else:
            eta_full, eta = avw(
                uw=uwnd,
                vw=vwnd,
            )
            save_nc(eta_full, ofile, dname)
        if ocean_flag:
            eta = eta.where(land_mask)
        save_nc(eta, ofile_850, dname)
    print('[saved]:', ofile)

    # relative humidity at 600hPa in %
    print('relative humidity in %')
    dname = 'rhum'
    ofile = os.path.join(odir, f'{dname}.{year}.nc')
    if os.path.exists(ofile):
        H = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        H = rhum.sel(level=600).drop('pfull')
        H.attrs['long_name'] = '600hPa relative humidity'
        H.attrs['units'] = '%'
        save_nc(H, ofile, dname)
        print('[saved]:', ofile)

    # GPI (Emanuel and Nolan 2004): |10**5\eta|**(3/2) * (H/50)**3 * (Vpot/70)**3 * (1+0.1*Vshear)**(-2)
    print('GPI')
    dname = 'GPI'
    ofile = os.path.join(odir, f'{dname}.{year}.nc')
    if os.path.exists(ofile):
        GPI = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
        # print(eta.values)
        # print(H.values)
        print(PI.vmax.values)
        # print(Vshear.values)
    else:
        GPI = (1e5 * np.absolute(eta))**(3/2) \
            * (H/50)**3 \
            * (PI.vmax/70)**3 \
            * (1+0.1*Vshear)**(-2)
        GPI.attrs['long_name'] = 'Genesis Potential Index'
        GPI.attrs['history'] = '|10**5\eta|**(3/2) * (H/50)**3 * (Vpot/70)**3 * (1+0.1*Vshear)**(-2)'
        save_nc(GPI, ofile, dname)
        print('[saved]:', ofile)

    # GPI2010 (Emanuel 2010): |\eta|**3 * chi**(-4/3) * max((Vpot-35),0)**2 * (25+Vshear)**(-4)
    print('GPI2010')
    dname = 'GPI2010'
    ofile = os.path.join(odir, f'{dname}.{year}.nc')
    if os.path.exists(ofile):
        GPI2010 = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        GPI2010 = np.absolute(eta)**3 \
            * chi_sb.where(chi_sb > 0)**(-4/3) \
            * (PI.vmax - 35).clip(min=0)**2 \
            * (25 + Vshear)**(-4)
        GPI2010.attrs['long_name'] = 'Genesis Potential Index of Emanuel2010'
        GPI2010.attrs['history'] = '|\eta|**3 * chi**(-4/3) * max((Vpot-35),0)**2 * (25+Vshear)**(-4)'
        GPI2010.to_dataset(name=dname) \
            .to_netcdf(ofile,
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)


if __name__ == '__main__':

    path = "/data/projects/REMOSAT/tamnnm/iwtrc/full_E/"
    # fn_ocean = xr.open_dataset(f"{path}land.nc", decode_times=True)
    # fn_ocean = fn_ocean.where(fn_ocean.land != 1)

    year = 1881

    def open_dts(name):
        try:
            dArray = xr.open_dataarray(
                f'{path}{name}.{year}.nc', decode_times=True)
        except:
            dts = xr.open_dataset(f'{path}{name}.{year}.nc', decode_times=True)
            for var in dts:
                if len(dts[var].shape) >= 2:
                    break
            dArray = dts[var]
        return cut_co(ctime_short(dArray, start_date="1881-09-01",
                                  end_date="1881-10-31"), 40, -5, 180, 100)

    print('Opening datasets')
    sst = open_dts("skt")
    rhum2m = open_dts("rhum.2m")
    t2m = open_dts("air.2m")
    prmsl = open_dts("prmsl")

    uwnd = open_dts("uwnd")
    vwnd = open_dts("vwnd")
    T = open_dts("air")
    rhum = open_dts("rhum")
    shum = open_dts("shum")

    if prmsl.attrs['units'] == 'hPa':
        prmsl = prmsl * 100

    t2m = cut_co(ctime_short(t2m, start_date="1881-09-01",
                 end_date="1881-10-31"), 40, -5, 180, 100)
    print("Reverse the data if needed")
    level_data = T[find_name(T, 'level')].values
    lat = T[find_name(T, 'lat')].values

    if level_data[0] > level_data[1]:
        # ? Level must be in ascending order and larger than 100 hPa
        level_data = level_data[level_data >= 100][::-1]
        uwnd = cut_level(uwnd, 100, opt="gt", reverse=True)
        vwnd = cut_level(vwnd, 100, opt="gt", reverse=True)
        T = cut_level(T, 100, opt="gt", reverse=True)
        rhum = cut_level(rhum, 100, opt="gt", reverse=True)
        shum = cut_level(shum, 100, opt="gt", reverse=True)
    # raise KeyboardInterrupt
    print('Calculating GPI')
    gpi_cal(sst=sst, slp=prmsl, prs=level_data, T=T, rhum=rhum, shum=shum,
            uwnd=uwnd, vwnd=vwnd, rhum2m=rhum2m, t2m=t2m, year=year)
