import xarray as xr
import os

#To get the current directory
#cwd= os.getcwd()

#change directory
os.chdir("/work/users/student6/tam/pap25_QA_1945/cal/twcr/netcdf")

#list file within this dir
nc_file= os.listdir(os.getcwd())
nc_dts = [xr.open_dataset(filename) for filename in nc_file]

#dimensions in nc file when command ncdump -h is METADATA
# to read actual dimensions within use ncdump -h [file_name] | grep "dimensions:"
#to create actual dimensions from variable must use set_coords 
for nc in nc_dts:
    nc=nc.set_coords(['time','lat','lon','time_bnds'])

#try to merge file into one file
con_dts = xr.open_mfdataset(nc_dts,concat_dim=['time','lat','lon','time_bnds'],combine='nested')
print("yeah")

#Extract the datasets from the nested structure
#dts=merged_dataset.values()
#print(dts)

#Concatenate the datasets along other dimension
#con_dts= xr.concat(nc_dts,dim=['time'])

#save to a netcdf 
con_dts.to_netcdf('test.nc')

