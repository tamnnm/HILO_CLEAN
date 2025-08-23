"""
Spyder Editor

This is a temporary script file.
"""
from ast import Continue, Pass
from re import I
from selectors import EpollSelector
from tkinter import ttk
from cf_units import decode_time
from matplotlib.font_manager import ttfFontProperty
#from matplotlib.lines import _LineStyle
import pandas as pd
import matplotlib
import numpy as np
import pandas as pd
import os
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from pandas import read_csv
import csv
import xarray as xr
from matplotlib.image import imread
import numpy as np
import pandas as pd

params = {
    'axes.titlesize' :25, 
	'axes.labelsize': 25,
	'font.size': 20,
    'font.family':'serif',
	'legend.fontsize': 20,
    'legend.loc': 'upper right',
    'legend.labelspacing':0.25,
	'xtick.labelsize': 20,
	'ytick.labelsize': 20,
	'lines.linewidth': 3,
	'text.usetex': False,
	# 'figure.autolayout': True,
	'ytick.right': True,
	'xtick.top': True,

	'figure.figsize': [12, 10], # instead of 4.5, 4.5
	'axes.linewidth': 1.5,
    
	'xtick.major.size': 15,
	'ytick.major.size': 15,
	'xtick.minor.size': 5,
	'ytick.minor.size': 5,

	'xtick.major.width': 5,
	'ytick.major.width': 5,
	'xtick.minor.width': 3,
	'ytick.minor.width': 3,

	'xtick.major.pad': 10,
	'ytick.major.pad': 12,
	#'xtick.minor.pad': 14,
	#'ytick.minor.pad': 14,

	'xtick.direction': 'inout',
	'ytick.direction': 'inout',
   }
plt.clf()
matplotlib.rcParams.update(params)

#Call to have the full list of lat, lon, name and acronym of each city
Data_path="/work/users/student6/tam/pap25_QA_1945/"
cls=["tp","shum","w","z","ghw"]
temp_var=["year","tmp","tmx","tmn"]
prep_var=["year","pre","no_day","hum"]
month=["Jan","Feb","Mar","Apr","May","June","July","Aug","Sep","Oct","Nov","Dec"]
season=[None,"JJA","MAM","SON","DJF"]
option=["f","bss","base"]
var_rean_s=[]
var_rean_l=[]


#%% 5.cut year and coordinate
#cut the time coordinate according to certain range
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
#%% 6.func extract all the variables and categorize it
#ATTENTION WHEN USING CLASS: 2 steps
#STEP 1: name_of_variable=var(....)
#STEP 2: name_of_variable.function() with function in this case is var_f(SELF)
#only function use function with (SELF) 
class var():
    def __init__(self,dataset,main,factor=1):
        self.dataset=dataset
        self.main=main
        self.factor=factor
        #print(self.dtset)
        
    def var_f(self):
        dataset=self.dataset
        main=self.main
        #print(dtset)
        #return
        
        #seperate coords, attrs and variables then combine in DataArray
        def cut(self,time_data,f,ss):
            #print(self.dt)
            main_dt=self.main.sel(time=time_data)
            #print(main_dt)
            co=main_dt.coords
            atr=main_dt.attrs
            #print(self.var_name)
            #print(self.dt)
            name=f'{dataset}_{f}{ss}'
            variable=main_dt.variables[f'{dataset}']
                    #print(name)
            #reannalysis has T2m in K
                    #"""
            if dataset == "air": 
                var_fin=variable-273.15
                            #print(name,"Yay")
                        #"""
            #reannalysis and UDel has different factor for pre    
                    #print(self.factor_p)
            elif dataset=="pre":
                #print("af",self.factor_p)
                factor=self.factor_p
                var_fin=variable*factor
                        #print(name,"Yeah")
            elif self.var_name[i]=="msl":
                            #print("af",self.factor_p)
                var_fin=variable*0.01
            else:
                var_fin=variable
                    
            globals()[name]=xr.DataArray(data=var_fin,coords=co,attrs=atr)
            return
        
        #This happens since the humidity has another level from 100 to 1000 while other have it from 1 to 1000
        try:
            main=main.drop("level_2")
        except:
            pass
        
        pas_va=["lat","lon","time","stn","level"]
        list_var=list(main.variables.keys())
        for i, var in enumerate(list_var[::-1]):  # iterate over reversed list_var using enumerate()
            if var in pas_va or any(v in var for v in pas_va):  # check if var is in pas_va or contains elements from pas_va
                list_var.pop(len(list_var) - i - 1)  # remove the variable from list_var

        #print(list_var)

        for i in range(len(list_var)):
            #have to specify range(len...) since the new element will not replace the position
            #FIND: if you want to search for any conditions then you have to use AND
            if (list_var[i].find("pcp")!=-1) or (list_var[i].find("pre")!=-1) or (list_var[i]=="tcrw"):
                var="pre"
            elif list_var[i].find("sst" and "skt")!=-1:
                var="sst"
            elif list_var[i].find("air")!=-1 or list_var[i]=="air":
                var="t2m"
            elif list_var[i]=="hgt":
                var="z"
            elif list_var[i]=="uwnd":
                var="u"
            elif list_var[i]=="vwnd":
                var="v"
            elif list_var[i]=="q":
                var="shum"
            elif list_var[i]=="prmsl":
                var="msl"
            else:
                var=list_var[i]
            """
            if dtset.find("_ghw")!=-1 and list_var[i] not in var_rean_l:
                var_rean_l.append(list_var[i])
            elif dtset.find("_tp")!=-1 and list_var[i] not in var_rean_s:
                var_rean_s.append(list_var[i])
            """
            var_name.append(var)
        #print(var_name,dt)
    #Define the factor of the precipitation and maybe other
        if dtset.find("_tp")!=-1:
            if dt=="twcr":
                factor_p=10e1
            else:
                factor_p=10e3
        elif dtset=="UDel":
            factor_p=10
        else:
            factor_p=1
        #print("fp",factor_p)
           
    #in order to pass value to the following    
        self.list_var=list_var
        self.var_name=var_name
        self.factor_p=factor_p
        self.dt=dt
        #print(dt)
        time_co=main.coords['time']
        #base: 30-60 for rean, grid keep the same
        if self.city==None:
            for opt in option:
                for ss in season:
                    #print(ss)
                    if opt=="base":
                        if dt in fol_rean:
                            time_data= year_cut(time_co,'br',ss)
                        else:
                            time_data= year_cut(time_co,'bg',ss)
                    else:
                        time_data= year_cut(time_co,opt,ss)
                    
                    if opt=="f":
                        f=""
                    else:
                        f=f'_{opt}'
                    
                    if ss==None:
                        s=""
                    else:
                        s=f"_{ss}"
                    cut(self,time_data,f,s)
        else:
            time_data= year_cut(time_co,"f",None)
            cut(self,time_data,"","")
        
       
#%% 7.extract the original data through all folder into each parameter
def data_org(folder):
    if folder==fol_grid:
        folder="ob/"
        for dataset in fol_grid:
            if dataset=="GHCN":
                dset=xr.open_dataset(f'{Data_path}{folder}{dataset}.nc',decode_times=False)
                units,reference_date=dset.time.attrs['units'].split('since')
        #print(reference_date)
                if reference_date.find('19')==-1:
                    reference_date='1900-1-1 00:00:0.0'
                    dset['time']=pd.date_range(start=reference_date,periods=dset.sizes['time'],freq='M')
                    main_data=dset.precip
                    globals()['GHCN']=main_data
                    ghcn=var(dataset,main_data)
                    ghcn.var_f()
            else:
                main_data=xr.open_dataset(f'{Data_path}{folder}{dataset}.nc')
                globals()[dataset]=main_data
                grid=var(dataset,main_data)
                grid.var_f()
    else:
        for name in os.listdir(f'{Data_path}cal/'):
            if name.find(".nc")==-1:
                continue
            else:
                if 1==1:#name.find("twcr")!=-1:
                    #i=i+1
                    #print(name)
                    file=f'{Data_path}cal/{name}'
                    main_data=xr.open_dataset(file)
                    globals()[name[:-3]]=main_data
                
                    #"""
                    #Use if we need all 
                    #if name[:-3].find("twcr")!=-1:
                    rean=var(name[:-3],main_data)
                    rean.var_f()
                    #"""
                else:
                    continue
            #print(i)
    return

#Execute about fuction
#data_org(fol_grid)
data_org(fol_rean)
#print(twcr_pre)
#%%8.extract data to plot 
def data_out(dataset,pra,ax=None,city=None,season=None,t=None,level=None,base=None):
    #settle the name of the dataset
    if base==None:
        f=''
    else:
        f=f'_{base}'  
        
    if season==None or t!=None:
        s=''
    else:
        s=f'_{season}'
        
    if city==None:
        name=f'{dataset}_{pra}{f}{s}'
    else:
        name=f'{dataset}_{city}_{pra}{f}{s}'
    #print(city,name)
    main_data=globals()[name]
    #print(main_data)
    time_co=main_data.coords['time'] 
    
    #print(time_co)
    if t==None:
        time_data=time_co
    elif f'{t}'.find("_")!=-1:
        mo=t.split('_')[0]
        yr=t.split('_')[1]
        time_data= time_co[(time_co.dt.month==int(mo)) & (time_co.dt.year==int(yr))]
        #print(mo,yr)
        #print(time_data)
    elif int(t)<13: #month extract from below is string not interger
        #print(t)
        time_data=time_co[time_co.dt.month==int(t)]
        #print(time_co)
        #print(time_co)
    else:
        if season=="DJF":
            #for time in time_data:
                #if (time.dt.month==12 & time.dt.year==t) or (time_co.dt.month==(1 or 2) & time_co.dt.year==(t+1)):
            cond=(time_co.dt.month<=2) & (time_co.dt.year==(int(t)+1))
            cond2=(time_co.dt.month==12) & (time_co.dt.year==int(t))
            time_data=time_co[cond | cond2]
        else:
            time_data=time_co[time_co.dt.year==int(t)]
    
    #print(time_data)
    #sel the level
    try:
        level_co=main_data.coords['level']
        if level==None:
            level_data=level_co
        else:
            level_data=level_co[(level_co==level)]
        try:
            data_TK=main_data.sel(time=time_data,level=level_data)
        except:
            data_TK=main_data.sel(time=time_data,level=level_data)
    except:
        data_TK=main_data.sel(time=time_data)
    
    if ax==None:
        return data_TK
    else:
        data_TK_mean=np.mean(data_TK,axis=ax)
        #print(data_TK_mean)
        return data_TK_mean
#keep the data global

#%% 9.PLOT LEVEL
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
#import metpy.calc as mpcalc
#from metpy.units import units
import numpy as np
import shapefile as shp
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon

import pandas as pd

#dt: data, p: parameter; tg: time range; dlat: downlat; ulat:uplat; dlon: downlon: ulon: uplon
#ty:type ;l: pressure level; ss: season; z:potential height ;rg:range
def plot_lvl(dt,p,tg,dlat,ulat,dlon,ulon,ty=None,l=None,ss=None,z=None,rg=None,zoom=None):
    #Get the high resolution of the background
    #b_img="/work/users/student6/tam/back_pic"
    #os.environ["CARTOPY_USER_BACKGROUNDS"] = "/work/users/student6/tam/back_pic/"
    map_pro = ccrs.PlateCarree()
    fig=plt.figure()
    ax = plt.subplot(111,projection=map_pro)
    #ax.background_img(name="ETO", resolution="high")
    #ax.stock_img()
   
    #change the lat, lon data into the same name 
    def d_plt(dataset,p):
        dset_test=dataset#-twcr_sst_base
        lat_data,lon_data=co(dset_test,up_lat,down_lat,up_lon,down_lon)
        try:
            dset_plot=dset_test.sel(lon=lon_data,lat=lat_data)
        except:
            dset_plot=dset_test.sel(longitude=lon_data,latitude=lat_data)
        #ax here is for level
        #if your var is sst, don't mean again
        if p in ("sst","msl","t2m"): #this has no level
            dset_plot=dset_plot    
        else:
            dset_plot=np.mean(dset_plot,axis=0)
        return dset_plot,lat_data,lon_data
    
    if p=="w":
        name_pra="Wind"
        par="u"
        par1="v"
    else:
        if p=="sst":
            name_pra="SST"
        elif p=="msl":
            name_pra="MSLP"
        elif p=="t2m":
            name_pra="Temp"
        elif p=="z":
            name_pra="Geopotential height"
        elif p=="shum":
            name_pra="Humidity"
        par=p
        
    #limit to plot 
    down_lat=dlat
    up_lat=ulat
    down_lon=dlon
    up_lon=ulon

    if tg.find("_")!=-1:
        tg_base=tg.split("_")[0]
        yr=tg.split("_")[1]
    else:
        yr=tg
        tg_base=None
    #ax=0 here is for the time
    org_test=data_out(dt,pra=par,ax=0,season=ss,level=l,t=tg)
    org_base=data_out(dt,pra=par,ax=0,season=ss,level=l,t=tg_base,base="base")
    #twcr_sst_DJF_45=data_out(dataset="twcr",pra="sst",season="DJF",t=1945)

    #PREPARE WITH LIMIT LON,LAT
    if f'{ty}'.find("ano")!=-1:
        dset_test,lat,lon=d_plt(org_test-org_base,par)
    else:
        dset_test,lat,lon=d_plt(org_test,par)
    #dset_test_2=d_plt(twcr_test)[0]
    dset_base=d_plt(org_base,par)[0]

    
    #EXCLUSIVE FOR POTENTIAL HEIGHT
    #if you take mean in data_out it will create errors
    #"""
    if par=="z":
        if z!=None:
            while rg<=5:
                try:
                    dset_plot=dset_test.where(abs(dset_test.values - z)<=rg)
                    base_plot=dset_base.where(abs(dset_base.values - z)<=rg+3)
                    break
                except:
                    rg=rg+1
        else:
            raise Exception("There are no range here")
    else:
        dset_plot=dset_test
        base_plot=dset_base
    #"""
    #print(dset_plot)
    
    
    #PLOT
    #try:
        #CONTOUR FOR SST AND HEIGHT AND HUMIDITY
        #QUIVER FOR WIND
    #print(p)
    if p=="w":
        def quiver(axe,X,Y,lab,uwnd,vwnd,rgs,c,xpos,ypos,ot=None,ot2=None): #rgs: regridded shape, cmap=colormap, ot=others
            u_norm = uwnd.values / np.sqrt(uwnd.values ** 2.0 + vwnd.values ** 2.0)
            v_norm = vwnd.values / np.sqrt(uwnd.values ** 2.0 + vwnd.values ** 2.0)
            #print(LON.shape,LAT.shape,uwnd.shape,vwnd.shape)
                
            #Set color scale 
            """
            colors = np.arctan2(uwnd, vwnd)
            norm = Normalize()
            norm.autoscale(colors)
            colormap = cm.inferno
            """
            # we need to normalize our colors array to match it colormap domain
            # which is [0, 1]
            #if extra==None:
            ax_plot=axe
            #else:
                #ax_plot = ax.twinx()
            if rgs!=None:
                qui=ax_plot.quiver(X,Y,u_norm,v_norm,#np.arctan2(u_norm,v_norm),
                transform=map_pro,
                scale=20,
                zorder=3,
                width=0.0009,
                pivot='middle',
                headlength=1,
                minlength=3,
                headwidth=5,
                linewidth=3,
                #cmap=c,
                #color=colormap(norm(colors)),
                #angles='xy',#if you want to have color on vector, make sure u have arctan2+angles+cmap
                facecolor=c,
                #Regridding can be an effective way of visualising a vector field, particularlyif the data is dense or warped
                regrid_shape=rgs)
            else:
                qui=ax_plot.quiver(X,Y,u_norm,v_norm,#np.arctan2(u_norm,v_norm),
                scale=80,
                zorder=3,
                width=0.009,
                pivot='middle',
                #headlength=2,
                #minlength=3,
                #headwidth=5,
                linewidth=0.5,
                #cmap=c,
                #color=colormap(norm(colors)),
                #angles='xy',#if you want to have color on vector, make sure u have arctan2+angles+cmap
                facecolor=c,
                #Regridding can be an effective way of visualising a vector field, particularlyif the data is dense or warped
                #regrid_shape=rgs
                )
                
            if ot!=None:
                #windspeed

                if f'{ty}'=="clim-ano":
                    vmin=-8
                    vmax=8
                    n=9
                else:
                    vmin=1000
                    vmax=1040
                    n=9
                levels = np.linspace(vmin, vmax,n)

                if ot=="wp":
                    windspeed = np.sqrt(uwnd ** 2 + vwnd ** 2)
                    #print("wp",type(windspeed))
                    lab_c="Windspeed [m/s]"
                    #contour
                    #cb_ax = fig.add_axes([0.15, 0.1, 0.75, 0.02]) #this is replaced with pad
                #Plot the contour of background (mean sea)
                elif ot=="msl":
                    org_msl=data_out(dt,pra="msl",ax=0,season=ss,t=tg)
                    org_msl_base=data_out(dt,pra="msl",ax=0,season=ss,t=tg_base,base="base")
                    #print("base",org_msl_base)
                    if f'{ty}'=="clim-ano":
                        msl_plot=d_plt(org_msl-org_msl_base,"msl")[0]
                        if zoom==None:
                            ax_plot.quiverkey( qui, xpos, ypos, 2,label=lab, labelpos='N', labelcolor=c)
                        else:
                            pass                    
                    elif f'{ty}'.find("clim")!=-1:
                        msl_plot=d_plt(org_msl_base,"msl")[0]
                    else:
                        msl_plot=d_plt(org_msl,"msl")[0]
                    lab_c='Mean sea level pressure[hPa]'
                
                cont=msl_plot.plot.contourf(ax=ax_plot,
                                                    cmap='bwr_r',
                                                    transform=ccrs.PlateCarree(),
                                                    add_colorbar=False,
                                                    levels=levels,
                                                    vmin=vmin,
                                                    vmax=vmax)
                if ot2=="Yes":
                    fig.colorbar(ScalarMappable(norm=cont.norm, cmap=cont.cmap),
                                    ticks=levels,
                                    ax=ax_plot,
                                    label=lab_c,
                                    fraction=0.045,orientation='horizontal', extend='both'
                                    )
                """
                color_fig, color_ax = plt.subplots(figsize=(10,5))
                color_fig.colorbar(ScalarMappable(norm=cont.norm, cmap=cont.cmap),
                                ticks=levels,
                                ax=color_ax,
                                label=lab_c,
                                fraction=0.045,orientation='horizontal', extend='both'
                                )
                color_ax.axis('off')

                    # Save the legend as a separate figure
                color_fig.savefig(
                        f'{Data_path}Data/legend/{p}')
                plt.close(color_fig)
                """
                #cbar = plt.colorbar(cont,fraction=0.045,orientation='horizontal', extend='both')#, cax=cb_ax
                #cbar.set_label(label=lab, size=12)
                #tick_locator = plt.ticker.MaxNLocator(nbins=4)
                #cbar.locator = tick_locator
                #cbar.update_ticks()
                #plt.colorbar()
                return 
            else:
                pass
                return
            
            #Creat meshgrid to draw vector
        LON,LAT=np.meshgrid(lon,lat)
            
        #FOR WIND, ORG_TEST IS U ELEMENT AND ORG_TEST_2 IS V ELEMENT
        org_test_2=data_out(dt,pra=par1,ax=0,season=ss,level=l,t=tg)
        org_base_2=data_out(dt,pra=par1,ax=0,season=ss,level=l,t=tg_base,base="base")
        if f'{ty}'.find("ano")!=-1:
            dset_plot_2=d_plt(org_test_2-org_base_2,par1)[0]
        else:
            dset_plot_2=d_plt(org_test_2,par1)[0]
        #dset_test_2=d_plt(twcr_test)[0]
        dset_base_2=d_plt(org_base_2,par1)[0]
        #widths = np.linspace(0, 1, LON.size)
            
        #CALCULATE WINDSPEED: Remember to use the variable not the whole dataArray
        #uwnd=np.asarray(dset_plot.values)
        #vwnd=np.asarray(dset_plot_2.values)
            
        #Plot the vector
        #do 2 if so they can plot one or two simutanously if we need
        if f'{ty}'.find("clim")!=-1: #be careful if we don't have this the contour won't plot
            quiver(ax,LON,LAT,"clim",dset_base,dset_base_2,300,"green",1.2,0.9,ot="msl")#"jet"
        #elif f'{ty}'.find("clim")!=-1:
            #quiver(ax,LON,LAT,"clim",dset_base,dset_base_2,300,"green",1.2,0.9)#"jet"
            #if you only plot clim then remember to put ot(background contour of msl in)
        else:
            pass
            
        #Since we gonna use the ano more often and it usually goes with clim so we put ot here
        #"""
        if ty==None or f'{ty}'.find("ano")!=-1:
            if ty==None:
                lab=None
            else:
                lab="ano"
            quiver(ax,LON,LAT,lab,dset_plot,dset_plot_2,300,"black",1.2,0.8,ot="msl",ot2="Yes")#,"viridis"
        else:
            pass
            #"""
        #print(dset_base,dset_plot)
        #print("OH!")
            
    else:
    #OTHER CASE    
        if f'{ty}'.find("line")!=-1:
            pl=dset_plot.plot.contour(ax=ax,colors="red",linewidths=2,levels=1)
            p1l=base_plot.plot.contour(ax=ax,colors="gray",linewidths=2,linestyles="dashed",levels=1)
            #use when it requires proxy artists
            main_line = mlines.Line2D([], [], color='r',
                    markersize=15, label='10/1944')
            base_line = mlines.Line2D([], [], color='gray',
                    markersize=15,linestyle="dashed", label='1930-1960')
            ax.legend(handles=[main_line,base_line])

            #ax.clabel(pl, inline=True,fontsize=20,colors="white") #show the number on the line
            #ax.clabel(p1l, inline=True,fontsize=15,colors="white"

            #contour the base
        elif f'{ty}'.find("ctb")!=-1:
            base_plot.plot.contourf(ax=ax)
                
            #contour only for the main dataset or the anomaly    
        elif f'{ty}'.find("ct")!=-1 or f'{ty}'.find("ano")!=-1:
            if p=="sst" or p=="t2m":
                if f'{ty}'.find("ano")!=-1:
                    vmin=-6
                    vmax=6
                    n=25
                    t_n=13
                else:
                    vmin=-35
                    vmax=35
                    n=15
                    t_n=n
                levels = np.linspace(vmin, vmax, n)
                levels_lab=np.linspace(vmin, vmax, t_n)
                sst_plot=dset_plot.plot.contourf(ax=ax,transform=ccrs.PlateCarree(),add_colorbar=False,levels=levels)
                lab='degree Celcius'
                
                fig.colorbar(ScalarMappable(norm=sst_plot.norm, cmap=sst_plot.cmap),
                                ticks=levels_lab,
                                ax=ax,
                                label=lab,
                                fraction=0.045,orientation='horizontal', extend='both'
                                )
                #"""
                color_fig, color_ax = plt.subplots(figsize=(10,5))
                color_fig.colorbar(ScalarMappable(norm=sst_plot.norm, cmap=sst_plot.cmap),
                                ticks=levels,
                                ax=color_ax,
                                label=lab,
                                fraction=0.045,orientation='horizontal', extend='both'
                                )
                color_ax.axis('off')
                
                    # Save the legend as a separate figure
                color_fig.savefig(
                        f'{Data_path}Data/legend/{p}')
                plt.close(color_fig)
                #"""
                #cbar = plt.colorbar(sst_plot,fraction=0.045,orientation='horizontal', extend='both')#, cax=cb_ax
                #tick_locator = plt.ticker.MaxNLocator(nbins=4)
                #cbar.locator = tick_locator
                #cbar.update_ticks()
            elif p=="msl":    
                lab_c='Mean sea level pressure[hPa]'
                if f'{ty}'.find("ano")!=-1:
                    vmin=-8
                    vmax=8
                    n=9
                else:
                    vmin=1000
                    vmax=1040
                    n=9
                levels = np.linspace(vmin, vmax,n)
                cont=dset_plot.plot.contourf(ax=ax,
                                                    cmap='bwr_r',
                                                    transform=ccrs.PlateCarree(),
                                                    add_colorbar=False,
                                                    levels=levels,
                                                    vmin=vmin,
                                                    vmax=vmax)
                fig.colorbar(ScalarMappable(norm=cont.norm, cmap=cont.cmap),
                                ticks=levels,
                                ax=ax,
                                label=lab_c,
                                fraction=0.045,orientation='horizontal', extend='both'
                                )
        
        
    #set the beautiful background
    #to crop the background must use set_extend
    ax.set_extent([down_lon, up_lon,down_lat,up_lat],ccrs.PlateCarree())
        
    #All other feature (coastline,land,borders)
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))#, linewidth=borders)#resolution='10m')
    #if f'{ty}'.find('line')!=-1:
        #ax.add_feature(cfeature.LAND)#, facecolor=cfeature.COLORS["land_alt1"])
        #ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
        
    """land_50m = cfeature.NaturalEarthFeature(category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
        ax.add_feature(land_50m)"""
    #This is used for zoom in figure for ex wind factor in local area closer to VN
    if zoom!=None:
        axins = zoomed_inset_axes(ax,5,loc=1) # zoom=6
        # pcolormesh(x,y,d[i,:,:],cmap=colormap,vmin=dmin,vmax=dmax)
        # pcolormesh(x,y,z,cmap=colormap,vmin=-dmax,vmax=dmax)
        if p=="w":
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
            x=np.linspace(xlim[0],xlim[1])
            y=np.linspace(ylim[0],ylim[1])
            XZ,YZ=np.meshgrid(x,y)
            if f'{ty}'==("clim"): #be careful if we don't have this the contour won't plot
                quiver(axins,XZ,YZ,"clim",dset_base,dset_base_2,None,"green",1.05,0.9,ot="msl")#"jet"
            elif f'{ty}'.find("clim")!=-1:
                quiver(axins,XZ,YZ,"clim",dset_base,dset_base_2,None,"green",1.05,0.9)#"jet"
                    #if you only plot clim then remember to put ot(background contour of msl in)
            else:
                pass
               
                #Since we gonna use the ano more often and it usually goes with clim so we put ot here
                #"""
            if ty==None or f'{ty}'.find("ano")!=-1:
                if ty==None:
                    lab=None
                else:
                    lab="ano"
                quiver(axins,XZ,YZ,lab,dset_plot,dset_plot_2,None,"black",1.05,0.8,ot="msl")#,"viridis"
            else:
                pass
                #"""
            #print(dset_base,dset_plot)
            #print("OH!")
                
        else:
            #OTHER CASE    
            if f'{ty}'.find("line")!=-1:
                pl=dset_plot.plot.contour(ax=axins,colors="red",linewidths=2,levels=1)
                p1l=base_plot.plot.contour(ax=axins,colors="gray",linewidths=2,linestyles="dashed",levels=1)
                #use when it requires proxy artists
                main_line = mlines.Line2D([], [], color='r',
                        markersize=15, label='main')
                base_line = mlines.Line2D([], [], color='gray',
                        markersize=15,linestyle="dashed", label='base')
                ax.legend(handles=[main_line,base_line])

            #ax.clabel(pl, inline=True,fontsize=20,colors="white") #show the number on the line
            #ax.clabel(p1l, inline=True,fontsize=15,colors="white"

            #contour the base
            elif f'{ty}'.find("ctb")!=-1:
                base_plot.plot.contourf(ax=axins)
                    
                #contour only for the main dataset or the anomaly    
            elif f'{ty}'.find("ct")!=-1 or f'{ty}'.find("ano")!=-1:
                if p=="sst" or p=="t2m":
                    if f'{ty}'.find("ano")!=-1:
                        vmin=-6
                        vmax=6
                        n=25
                        t_n=13
                    else:
                        vmin=-35
                        vmax=35
                        n=15
                        t_n=n
                    levels = np.linspace(vmin, vmax, n)
                    levels_lab=np.linspace(vmin, vmax, t_n)
                    sst_plot=dset_plot.plot.contourf(ax=axins,add_colorbar=False,levels=levels)
                    lab='degree Celcius'        
            elif p=="msl":    
                if f'{ty}'.find("ano")!=-1:
                    vmin=-8
                    vmax=8
                    n=9
                else:
                    vmin=1000
                    vmax=1040
                    n=9
                levels = np.linspace(vmin, vmax,n)
                cont=dset_plot.plot.contourf(ax=axins,
                                                    cmap='bwr_r',
                                                    add_colorbar=False,
                                                    levels=levels,
                                                    vmin=vmin,
                                                    vmax=vmax)

            # clabel(C,fontsize=14,fmt='%2.1f')#,manual=True)
            # subregion of the original image
        if p!="sst":
            axins.set_xlim(100, 110)
            axins.set_ylim(5, 25)
        else:
            axins.set_xlim(100, 120)
            axins.set_ylim(5, 25)
            #this is used for set all the ticks and labels of the zoom in to appear non-visible
        axins.axes.xaxis.set_visible(False)
        axins.axes.yaxis.set_visible(False)
        plt.setp(axins,xticks=[],yticks=[])
        #axins.set_extent([down_lon, up_lon,down_lat, up_lat],ccrs.PlateCarree())
        
        #All other feature (coastline,land,borders)
        axins.coastlines(resolution='10m')
        #axins.add_feature(cfeature.LAND,)#, facecolor=cfeature.COLORS["land_alt1"])
        #axins.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
        axins.add_feature(cfeature.BORDERS,resolution='10m')#, linewidth=borders)#resolution='10m')
        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    else:
        pass
        
        
    if l==None:
        name_l=''
    else:
        if z==None:
            name_l=f'-{l}hPa'
        else:
            name_l=f'-{l}hPa-{z}m'
            
    if ss==None:
        name_s=''
    else:
        name_s=f'-{ss}'
        
    if ty!=None:
        type=f'-{ty}'
    else:
        type=""
        
    if tg==None:
        tt=f'{name_pra}{name_l}{name_s}'
    else:
        try:
            tgian=tg.replace("_","/")
        except:
            tgian=tg
        tt=f'{name_pra}{type}{name_l}{name_s}-{tgian}'
    print(tt)
            
    ax.set_title(f'{tt}',pad=25)
    #ax.legend()
        
        
    #Advantage that we can specific ticks of the grid instead of automatic one
    #"""
    ax.set_xticks(np.linspace(down_lon, up_lon,5), crs=map_pro)
    ax.set_yticks(np.linspace(down_lat, up_lat,5), crs=map_pro)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
        
    ax.tick_params(axis='both',which="major",labelsize=15)
    #change fontsize of tick when have major
        
    #Cái này tương ứng với gridliné
    #"""
        
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    
    #Must put behind all the plot or it's gonna interfere with the plot. This plot is for the island of Vietnam
    if dlon>85:
        shp_path = "/work/users/student6/tam/map/vnm_admbnda_adm0_gov_20200103.shp"
        vnmap = shp.Reader(shp_path)

        txt_shapes = []
        for vnmapshape in vnmap.shapeRecords(): 
            listx=[]
            listy=[]
            # parts contains end index of each shape part
            parts_endidx = vnmapshape.shape.parts.tolist()
            parts_endidx.append(len(vnmapshape.shape.points) - 1)
            for i in range(len(vnmapshape.shape.points)):
                x, y = vnmapshape.shape.points[i]
                if x>108:
                    if i in parts_endidx:
                        # we reached end of part/start new part
                        txt_shapes.append([listx,listy])
                        listx = [x]
                        listy = [y]
                    else:
                        # not end of part
                        listx.append(x)
                        listy.append(y)
                else:
                    continue

        for zone in txt_shapes:
            x,y = zone
            plt.plot(x,y,markeredgecolor='none',color="k",markersize=10e-6,linewidth=0.7)
            #must have linewidth and small markersize or else the graph's gonna be bunch of bold dots
        ax.text(112, 10, "Spratly Islands", fontsize=13,rotation=45,alpha=0.6) 
        ax.text(111, 17.5, "Paracel Islands", fontsize=13,alpha=0.6) 
        ax.text(110, 13, "EAST SEA", fontsize=13,rotation=90,alpha=0.6) 
    else:
        pass
    plt.grid()
    plt.tight_layout()
        
    #ax.gridlines(draw_labels=False,alpha=0.6, dms=False, x_inline=False, y_inline=False)
    #ax.xlabels_top, ax.ylabels_right = False, False
    #ax.xlabel_style, ax.ylabel_style = {'fontsize': 12}, {'fontsize': 12}
    #Gridlines does not #print longitude or latitde + can't change label
        
    #test
    #plt.savefig(f'{Data_path}/Data/test/test{tg}_{ty}_noback.jpg',format="jpg")
        
    #save for geopotential height
    #print("Yes")
    plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/final/{l}hPa_{dt}_{ty}_{tg}_{p}.jpg'),format="jpg",dpi=300)
    plt.clf()
    """
    if zoom==None:
        if p=="sst" or p=="t2m":
            plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/{p}/yr{yr}/{dt}_{ty}_{tg}.jpg'),format="jpg",dpi=300)
        #save for wind,humidity
        elif p=="z":
            plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/{p}/{z}m/y_{yr}/{dt}_{ty}_{tg}.jpg'),format="jpg")
        else:
            if dlon>85 or dlon<80:
                if ss==None:
                    plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/large/{l}hPa_{dt}_{ty}_{tg}_{p}.jpg'),format="jpg",dpi=300)
                else:
                    plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/large/{l}hPa_{dt}_{ss}_{ty}_{tg}_{p}.jpg'),format="jpg",dpi=300)
            else:    
                if ss==None:
                    plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/{p}/{l}hPa/yr{yr}/{dt}_{ty}_{tg}.jpg'),format="jpg",dpi=300)
                else:
                    plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/{p}/{l}hPa/yr{yr}/{dt}_{ss}_{ty}_{tg}.jpg'),format="jpg",dpi=300)
        plt.clf()
    else:
        plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/zoom/{p}_{yr}_{dt}_{ty}_{tg}.jpg'),format="jpg",dpi=300)
        plt.clf()
    """
        #plt.show()
    """except Exception as e:
        #print(f"{tg},{p}")
        #print(e)
        return e"""
    return


"""import os
import imageio

png_dir = '../animation/png'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('../animation/gif/movie.gif', images)"""

#test
"""
for i in range(1,13):
    for yr in range(1943,1946):
        plot_lvl("twcr",p="z",tg=f"{i}_{yr}",l=500,rg=5,z=5865,ty="line",
         dlat=-15,ulat=40,dlon=80,ulon=180)
        plot_lvl("twcr",p="z",tg=f"{i}_{yr}",l=500,rg=5,z=5870,ty="line",
         dlat=-15,ulat=40,dlon=80,ulon=180)
"""
"""
for yr in range(1943,1946):
    
    for i in range(1,13): 
        plot_lvl("twcr",p="t2m",tg=f"{i}_{yr}",ty="ano",
                    dlat=-15,ulat=40,dlon=80,ulon=155)
        plot_lvl("twcr",p="t2m",tg=f"{i}_{yr}",ty="ct",
                    dlat=-15,ulat=40,dlon=80,ulon=155)
    
    for lv in [750,850,925]:
        for i in range(1,13):
            if (i<3 or i>8) and lv==750:
                continue
            elif (3<=i and i<=8) and lv==925:
                continue
            else:
                #plot_lvl(dt="twcr",p="msl",tg=f"{i}_{yr}",l=lv,ty="ano",
                #        dlat=0,ulat=40,dlon=85,ulon=155)
                #plot_lvl(dt="twcr",p="msl",tg=f"{i}_{yr}",l=lv,ty="ct",
                #        dlat=0,ulat=40,dlon=85,ulon=155)
                            
                plot_lvl("twcr",p="w",tg=f"{i}_{yr}",l=lv,ty="clim-ano",
                    dlat=0,ulat=40,dlon=85,ulon=155)
                plot_lvl("twcr",p="w",tg=f"{i}_{yr}",l=lv,ty="ano",
                    dlat=0,ulat=40,dlon=85,ulon=155)
                #plot_lvl("twcr",p="w",tg=f"{i}_{yr}",l=lv, #ty is nothing mean plot normal
                #    dlat=0,ulat=40,dlon=85,ulon=155)     
                #plot_lvl("twcr",p="w",tg=f"{i}_{yr}",l=lv,ty="clim",
                #    dlat=0,ulat=40,dlon=85,ulon=155)
                #"""
        

"""                               
        for ss in season:
            if (ss=="DJF" or ss=="SON") and lv==750:
                continue
            elif (ss=="MAM" or ss=="JJA") and lv==925:
                continue
            else:
                plot_lvl("twcr",p="w",tg=f"{yr}",l=lv,ss=ss,ty="clim-ano",
                    dlat=0,ulat=40,dlon=85,ulon=155)
                plot_lvl("twcr",p="w",tg=f"{yr}",l=lv,ss=ss,ty="ano",
                    dlat=0,ulat=40,dlon=85,ulon=155)
                plot_lvl("twcr",p="w",tg=f"{yr}",l=lv,ss=ss,ty="clim",
                    dlat=0,ulat=40,dlon=85,ulon=155)
                plot_lvl("twcr",p="w",tg=f"{yr}",l=lv,ss=ss,
                    dlat=0,ulat=40,dlon=85,ulon=155)
"""
spc=['8_1943','4_1944','8_1944','10_1944','12_1944','2_1945']
for d in spc:
    """plot_lvl(dt="twcr",p="msl",tg=d,ty="ano",
                                dlat=5,ulat=25,dlon=100,ulon=118)
    plot_lvl(dt="twcr",p="sst",tg=d,ty="ano",
                             dlat=5,ulat=25,dlon=100,ulon=118)
    plot_lvl(dt="twcr",p="t2m",tg=d,ty="ano",
                             dlat=5,ulat=25,dlon=100,ulon=118)"""
    for lv in [750,850,925]:
        plot_lvl(dt="twcr",p="w",tg=d,l=lv,ty="clim-ano",
                                dlat=5,ulat=25,dlon=100,ulon=118)
        
#plot_lvl("twcr",p="z",tg=f"10_1944",l=500,rg=5,z=5870,ty="line",
#            dlat=-10,ulat=50,dlon=85,ulon=180)
    
    
#"""
#"""
#large
# #,zoom="Yes")
#plot_lvl(dt="twcr",p="w",tg=f"12_1944",l=925,ty="ano",
#         dlat=-15,ulat=90,dlon=0,ulon=155)#,zoom="Yes")
#plot_lvl(dt="twcr",p="w",tg=f"12_1944",l=925,
#         dlat=-15,ulat=90,dlon=0,ulon=155)#,zoom="Yes")
#"""
"""
for yr in range(1943,1946):
    for i in range(1,13):
        plot_lvl("twcr",p="z",tg=f"{i}_1944",l=500,rg=5,z=5865,ty="line",
            dlat=-15,ulat=40,dlon=85,ulon=180)
        plot_lvl("twcr",p="z",tg=f"{i}_1944",l=500,rg=5,z=5870,ty="line",
            dlat=-15,ulat=40,dlon=85,ulon=180)
"""

#plot_lvl("cera",p="w",tg="12_1944",l=925,ty="clim",
#                    dlat=0,ulat=40,dlon=0,ulon=180)
# %%
