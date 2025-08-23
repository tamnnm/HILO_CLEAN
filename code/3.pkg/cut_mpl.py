#cut the time coordinate according to certain range
import xarray as xr
"""
def year_cut(time_co,option,option2):
    if option=="f": #focus from 1943-1946
        downlim=1943
        uplim=1946
    elif option=="bss":  #base from 1943-1946
        downlim=1943
        uplim=1953
    elif option=="bg":
        downlim=1941
        uplim=1960
    elif option=="br":
        downlim=1930
        uplim=1960        
    time_data=time_co[(time_co.dt.year>=downlim) & (time_co.dt.year<=uplim)]
    #print(option2)
    if option2==None:
        time_final=time_data
    else:
        time_final=time_data[(time_data.dt.season==option2)]
    return time_final
"""
def co(main_data,lat_up,lat_down,lon_up,lon_down):
    
    try:
        lat_co=main_data.coords['lat']
        lon_co=main_data.coords['lon']
    except:
        lat_co=main_data.coords['latitude']
        lon_co=main_data.coords['longitude']
    
    lat_data= lat_co[(lat_co >= lat_down) & (lat_co <= lat_up)]
    lon_data= lon_co[(lon_co >= lon_down) & (lon_co <= lon_up)]
    #print(lat_co,lon_co)
    #print(lat_up,lat_down,lon_up,lon_down)
    
    return lat_data,lon_data
