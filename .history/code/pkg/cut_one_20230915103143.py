import numpy as np
import xarray as xr
import os
import constant as cs

def cut_var(dts,var=None,val_only=None):
    if var==None:
        dts_final=dts
    else:
        dts_var=dts[f'{var}']
    if val_only==None:
        dts_final=dts_var
    else:
        dts_var=dts_var.values
    return dts_final
#This module is to cut dataset according to grid and time
def cut_co(dts,ulat,dlat,ulon,dlon,opt):
    try:
        lat_co=dts.coords['lat']
        lon_co=dts.coords['lon']
    except:
        lat_co=dts.coords['latitude']
        lon_co=dts.coords['longitude']
    
    lat_data= lat_co[(lat_co >= dlat) & (lat_co <= ulat)]
    lon_data= lon_co[(lon_co >= dlon) & (lon_co <= ulon)]
    #print(lat_co,lon_co)
    #print(ulat,dlat,ulon,dlon)
    
    return lat_data,lon_data 

#cut the time coordinate according to certain range
def cut_time(dts, yr=None,mon=None,range=None,ss='None'):
    """ opt 1: choose to only cut time co or cut for one DataArray
        opt 2: choose the define time range"""
    time_co=dts.coords['time']  
    if yr == None:
        raise ValueError("must include the value of year as yr=[number]")
        #check if there is a month required 
    else:
        if mon == None:
            time_final = time_co[time_co.dt.year == yr]
            if ss ==None:
                time_final=time_final
            else:
                if ss == "DJF":
                    cond=(time_co.dt.month<=2) & (time_co.dt.year==(int(yr)+1))
                    cond2=(time_co.dt.month==12) & (time_co.dt.year== try:
	       		    time_final=time_co[cond | cond2]
    	 	    elif (ss in ['MAM,JJA,SON'] ) != True:
        		    raise ValueError("Invalid season, change option to MAM, JJA, SON")
   		        else:
        		    time_final=time_final[(time_final.dt.season==ss)]
	    elif mon != None and not isinstance(mon, (int, float)) or mon>=13:
            raise ValueError("value of month is invalid")
        else:
             time_final = time_co[(time_co.dt.year == int(yr)) & (time_co.dt.month == mon)]   
        return time_final

#level doesn't need to extract, this to directly cut the dataset
def cut_level(dts,level,ax=None):
    print(
        "Be careful with the dimension of axis you are choosing,\
        you should check the order of the dimenions\
        e.g. level,time,lat,lon then after cut lat, lon you should choose ax=0 (mean along time element)\
        It's going to return the mean at a certain level"
        )
    try:
        level_co=dts.coords['level']
        if level==None:
            level_data=level_co
        else:
            level_data=level_co[(level_co==level)]
    #choose ax=0 in this case to mean along the time and return the average on a certain level
        if ax==None:
	        return level_data
        else:
            new_dts=np.mean(new_dts,axis=ax)
    #print(data_TK_mean)
            return new_dts
    except:
        print("This dataset are not in pressure levels")

def cut_mlp(dts,time_data=None,lat_data=None,lon_data=None,lev_data=None):        
    if time_data!=None:
    	final_time=dts.sel(time=time_data)
    else:
	final_time=dts
    if lat_data!=None and lon!=None:
    	try:
        	final_latlon=final_time.sel(lon=lon_data,lat=lat_data)
    	except:
        	final_latlon=final_time.sel(longitude=lon_data,latitude=lat_data)
    else:
	final_latlon=final_time
    if level!=None:
	final_dts=final_latlon.sel(lev_data=level)
    else:
	final_dts=final_latlon
    return final_dts        

"""
if __name__ == "__main__":
    os.chdir(cs.twcr_dir)
    dts_name="shum.mon.nc"
    var_name=dts_name.split(".")[0]
    dts_org=xr.open_dataset(dts_name)[var_name]
    #call the class
    dts_co = cut_co(dts_org,ulat=40,dlat=-15,ulon=180,dlon=80,opt=1)
    dts_co_t = cut_time(dts_co,opt1=1)
    if var_name in no_lv_dts: #this has no level
        dset_final=dts_co_t    
    else:
        dset_final=np.mean(dts_co_t,axis=0)
 """
    
    
