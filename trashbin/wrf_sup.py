import xwrf
import xarray as xr
from my_junk import cut_co
def xwrf_open(file_path,proj=False):
    ds=xr.open_dataset(file_path).xwrf.postprocess()
    # Error: AttributeError: 'CRS' object has no attribute 'dtype'
    projection = ds['wrf_projection'].item()
    ds=ds.drop(['XTIME','wrf_projection']).rename({'XLAT':'lat','XLONG':'lon','Time':'time',
                                               'XLAT_U':'lat_u','XLONG_U':'lon_u',
                                               'XLAT_V':'lat_v','XLONG_V':'lon_v',})
    return (ds, projection) if proj else ds

if __name__ == "__main__":
    file_path="/work/users/tamnnm/wrf/WRF/test/RUN_ERA5_194412_10/wrfout_d02_1945-02-28_00:00:00"
    ds=xwrf_open(file_path)
    ds_new=cut_co(ds,dlat=16,ulat=18,dlon=100,ulon=105,full=True)
    print(ds_new['full'])
