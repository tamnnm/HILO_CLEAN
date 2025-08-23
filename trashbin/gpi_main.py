# Referenced from https://github.com/wy2136/TCI
import xarray as xr
import numpy as np
import os

from import pi, wind_shear as ws, abs_vor as av, cut_one as co

os.chdir(cs.twcr_dir)


def do_tci(sst, prmsl, prs, T, rhum, uwnd, vwnd, lon_ocean, lat_ocean):
    '''calculate TC indices (e.g. GPI, VI) and related variables given FLOR/HiRAM atmos_month output'''

    if odir is None:
        odir = '.'
    ibasename = os.getcwd()
    # ds = xr.open_dataset(ifile)
    # is_ocean = ds.land_mask.load() < 0.1

# entropy deficit: (s_m_star - s_m)/(s_sst_star - s_b)
    print('entropy deficit')
    dname = 'chi'
    ofile = os.path.join(odir, ibasename.replace('.nc', f'.{dname}.nc'))
    if os.path.exists(ofile):
        chi = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        p_m = 600*100  # Pa
        chi = entropy_deficit(
            sst=sst,
            slp=slp,
            Tb=t2m,
            RHb=RH2m/100,
            p_m=p_m,
            Tm=Ta.sel(level=p_m/100).drop('level'),
            RHm=RH.sel(level=p_m/100).drop('level')/100
        ).where(is_ocean)
        chi.to_dataset(name=dname) \
            .to_netcdf(ofile, t
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)

    # entropy deficit for GPI2010: (s_b - s_m)/(s_sst_star - s_b)
    print('entropy deficit for GPI2010')
    dname = 'chi_sb'
    ofile = os.path.join(odir, ibasename.replace('.nc', f'.{dname}.nc'))
    if os.path.exists(ofile):
        chi_sb = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        p_m = 600*100  # Pa
        chi_sb = entropy_deficit(
            sst=sst,
            slp=slp,
            Tb=t2m,
            RHb=RH2m/100,
            p_m=p_m,
            Tm=Ta.sel(level=p_m/100).drop('level'),
            RHm=RH.sel(level=p_m/100).drop('level')/100,
            forGPI2010=True
        ).where(is_ocean)
        chi_sb.to_dataset(name=dname) \
            .to_netcdf(ofile,
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)

    # potential intensity
    print('potential intensity')
    ofile = os.path.join(ibasename, '/PI.nc')
    if os.path.exists(ofile):
        PI = xr.open_dataset(ofile)
        print('[opened]:', ofile)
    else:
        PI = pi.potential_intensity(
            sst=co.cut_co_mlp(sst, lat_ocean, lon_ocean),
            slp=co.cut_co_mlp(prmsl, lat_ocean, lon_ocean)*100,
            prs=prs,
            T=co.cut_co_mlp(T, lat_ocean, lon_ocean),
            rhum=co.cut_co_mlp(rhum, lat_ocean, lon_ocean),
            dim_x='lat', dim_y='lon', dim_z='level'
        )
        encoding = {dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}
                    for dname in ('pmin', 'vmax')}
        encoding['iflag'] = {'dtype': 'int32'}
        PI.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
        print('[saved]:', ofile)

    # wind shear: ( (u200-u850)**2 + (v200-v850)**2 )**0.5
    print('wind shear')
    dname = 'Vshear'
    ofile = os.path.join(ibasename, f'/{dname}.nc')
    if os.path.exists(ofile):
        Vshear = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        Vshear = ws.wind_shear(
            u850=uwnd.interp(pfull=850),
            v850=vwnd.interp(pfull=850),
            u200=uwnd.interp(pfull=200),
            v200=vwnd.interp(pfull=200)
        )
        Vshear.to_dataset(name=dname) \
            .to_netcdf(ofile,
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)

    """

    # ventilation index: Vshear * chi_m /V_PI
    print('ventilation index')
    dname = 'VI'
    ofile = os.path.join(odir, ibasename.replace('.nc', f'.{dname}.nc') )
    if os.path.exists(ofile):
        VI = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        VI = Vshear*chi/PI.vmax.pipe(lambda x: x.where(x>0))
        VI.to_dataset(name=dname) \
            .to_netcdf(ofile,
                encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
                unlimited_dims='time')
        print('[saved]:', ofile)
    """
    # absolute vorticity at 850hPa
    print('absolute vorticity')
    dname = 'eta'
    ofile = os.path.join(ibasename, f'/{dname}.nc')
    if os.path.exists(ofile):
        eta = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        eta = av.absolute_vorticity(
            file_uw=uwnd,
            file_vw=vwnd,
            level_data=850,
        )
        eta.to_dataset(name=dname) \
            .to_netcdf(ofile,
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)

    # relative humidity at 600hPa in %
    print('relative humidity in %')
    dname = 'H'
    ofile = os.path.join(ibasename, f'/{dname}.nc')
    if os.path.exists(ofile):
        H = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        H = rhum.sel(level=600).drop('pfull')
        H.attrs['long_name'] = '600hPa relative humidity'
        H.attrs['units'] = '%'
        H.to_dataset(name=dname) \
            .to_netcdf(ofile,
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)

    # GPI (Emanuel and Nolan 2004): |10**5\eta|**(3/2) * (H/50)**3 * (Vpot/70)**3 * (1+0.1*Vshear)**(-2)
    print('GPI')
    dname = 'GPI'
    ofile = os.path.join(ibasename, f'/{dname}.nc')
    if os.path.exists(ofile):
        GPI = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        GPI = (1e5 * np.absolute(eta))**(3/2) \
            * (H/50)**3 \
            * (PI.vmax/70)**3 \
            * (1+0.1*Vshear)**(-2)
        GPI.attrs['long_name'] = 'Genesis Potential Index'
        GPI.attrs['history'] = '|10**5\eta|**(3/2) * (H/50)**3 * (Vpot/70)**3 * (1+0.1*Vshear)**(-2)'
        GPI.to_dataset(name=dname) \
            .to_netcdf(ofile,
                       encoding={dname: {'dtype': 'float32',
                                         'zlib': True, 'complevel': 1}},
                       unlimited_dims='time')
        print('[saved]:', ofile)

    """
    # GPI2010 (Emanuel 2010): |\eta|**3 * chi**(-4/3) * max((Vpot-35),0)**2 * (25+Vshear)**(-4)
    print('GPI2010')
    dname = 'GPI2010'
    ofile = os.path.join(odir, ibasename.replace('.nc', f'.{dname}.nc') )
    if os.path.exists(ofile):
        GPI2010 = xr.open_dataset(ofile)[dname]
        print('[opened]:', ofile)
    else:
        GPI2010 = absolute(eta)**3 \
            * chi.where(chi>0)**(-4/3) \
            * (PI.vmax - 35).clip(min=0)**2 \
            * (25 + Vshear)**(-4)
        GPI2010.attrs['long_name'] = 'Genesis Potential Index of Emanuel2010'
        GPI2010.attrs['history'] = '|\eta|**3 * chi**(-4/3) * max((Vpot-35),0)**2 * (25+Vshear)**(-4)'
        GPI2010.to_dataset(name=dname) \
            .to_netcdf(ofile,
                encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
                unlimited_dims='time')
        print('[saved]:', ofile)
    """


if __name__ == '__main__':
    fn_ocean_org = xr.open_dataset("land.nc", decode_times=True)
    fn_ocean = fn_ocean_org.where(fn_ocean_org.land < 0.1)
    lat_ocean, lon_ocean = co.cut_co(
        fn_ocean, ulat=5, dlat=25, ulon=118, dlon=100)

    sst_org = xr.open_dataset("skt.mon.nc", decode_times=True)
    uwnd_org = xr.open_dataset("uwnd.mon.nc", decode_times=True)
    vwnd_org = xr.open_dataset("vwnd.nc", decode_times=True)
    prmsl_org = xr.open_dataset("prmsl.mon.nc", decode_times=True)
    T_org = xr.open_dataset("air.mon.nc", decode_times=True)
    rhum_org = xr.open_dataset("rhum.mon.nc", decode_times=True)

    rhum, level_data = co.cut_level(rhum_org, slice(100, None), opt=0)
    uwnd = co.cut_level(uwnd_org, level_data)
    vwnd = co.cut_level(uwnd_org, level_data)
    T = co.cut_level(uwnd_org, level_data)
    uwnd = co.cut_level(uwnd_org, level_data)

    do_tci(sst=sst_org, prmsl=prmsl_org, prs=level_data, T=T, rhum=rhum,
           uwnd=uwnd, vwnd=vwnd, lon_ocean=lon_ocean, lat_ocean=lat_ocean)
