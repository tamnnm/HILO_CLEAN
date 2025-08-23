#%% 1.IMPORT STUFF & CALL LIST NAME, LON,LAT OF CITY

#YOU HAVE TO CALL FIG!!! TO CHANGE THE WHITE PART AROUND
"""
Spyder Editor

This is a temporary script file.
"""
from ast import Continue
from re import I
from selectors import EpollSelector
from cf_units import decode_time
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

params = {
    'axes.titlesize' :25, 
	'axes.labelsize': 15,
	'font.size': 15,
    'font.family':'serif',
	'legend.fontsize': 15,
    'legend.loc': 'upper right',
    'legend.labelspacing':0.25,
	'xtick.labelsize': 15,
	'ytick.labelsize': 15,
	'lines.linewidth': 3,
	'text.usetex': False,
	# 'figure.autolayout': True,
	'ytick.right': True,
	'xtick.top': True,

	'figure.figsize': [15, 10], # instead of 4.5, 4.5
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
	'ytick.major.pad': 10,
	#'xtick.minor.pad': 14,
	#'ytick.minor.pad': 14,

	'xtick.direction': 'in',
	'ytick.direction': 'in',
   }
plt.clf()
matplotlib.rcParams.update(params)

#Call to have the full list of lat, lon, name and acronym of each city
Data_path="/work/users/student6/tam/pap25_QA_1945/"
fol_grid=['CRU','GHCN','UDel']
#fol_grid=['GHCN']
#fol_rean=['era','cera','twcr']
fol_rean=['twcr']
cls=["tp","shum","w","z","ghw"]
var_real=["pre","tmp"]
temp_var=["year","tmp","tmx","tmn"]
prep_var=["year","pre","no_day","hum"]
month=["Jan","Feb","Mar","Apr","May","June","July","Aug","Sep","Oct","Nov","Dec"]
season=[None,"JJA","MAM","SON","DJF"]
option=["f","bss","base"]
var_rean_s=[]
var_rean_l=[]

city_path="Data/City.csv"
city_list=pd.read_csv(f'{Data_path}{city_path}')
city_lat=city_list.iloc[:,1]
city_lon=city_list.iloc[:,2]
#print(city_lon[0])
city_acr=city_list.iloc[:,3]
city_full_name=city_list.iloc[:,0]
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
    def __init__(self,dtset,main,city=None,factor=1):
        self.dtset=dtset
        self.main=main
        self.city=city
        self.factor=factor
        #print(self.dtset)
        
    def var_f(self):
        dtset=self.dtset
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
            if self.dt=="GHCN":
                #print("Meh")
                if self.city==None:
                    name=f'GHCN_pre{f}{ss}'
                else:
                    name=f'GHCN_{self.city}_pre{f}{ss}'
                print(name)
                variable=main_dt
                globals()[f'{name}']=variable
            else:
                for i in range(len(self.list_var)):    
                    if self.city==None:
                        name=f'{self.dt}_{self.var_name[i]}{f}{ss}'
                    else:
                        name=f'{self.dt}_{self.city}_{self.var_name[i]}{f}{ss}'

                    variable=main_dt.variables[f'{self.list_var[i]}']
                    #print(name)
            #reannalysis has T2m in K
                    #"""
                    if self.var_name[i] in ("tmp","t2m","sst"): 
                        if self.dt in fol_grid:
                            var_fin=variable
                        else:
                            var_fin=variable-273.15
                            print(name,"Yay")
                        #"""
            #reannalysis and UDel has different factor for pre    
                    #print(self.factor_p)
                    elif self.var_name[i]=="pre":
                            #print("af",self.factor_p)
                        factor=self.factor_p
                        var_fin=variable*factor
                        print(name,"Yeah")
                    else:
                        var_fin=variable
                    
                    globals()[name]=xr.DataArray(data=var_fin,coords=co,attrs=atr)
                return
        
        #"""
        #to get the name that don't have ghw or tp just era or era_base
        if dtset.find("_")!=-1:
            dt=dtset.split("_")[0]
        #"""
        else:
            dt=dtset
        
        #This happens since the humidity has another level from 100 to 1000 while other have it from 1 to 1000
        try:
            main=main.drop("level_2")
        except:
            pass
        
        pas_va=["lat","lon","time","stn","level"]
        if dt=="GHCN":
            list_var=["precip"]
        else:
            list_var=list(self.main.variables.keys())
        print(list_var)
        var_num=len(list_var)
        #print(list_var[var_num-1])
        var_name=[]

        #IMPORTANT: remember to put the element that needs to be erased first
        i=var_num-1
        j=0
        while i>=0:
            if j==len(pas_va):
                j=0
                i=i-1
            #print("i",i,list_var[i],"j",j,pas_va[j])
            if list_var[i]==pas_va[j] or list_var[i].find(pas_va[j])!=-1:
                list_var.remove(list_var[i])
                i=i-1
                j=0
            #print("Yes")
            else:
                j=j+1 

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
            else:
                var=list_var[i]
            """
            if dtset.find("_ghw")!=-1 and list_var[i] not in var_rean_l:
                var_rean_l.append(list_var[i])
            elif dtset.find("_tp")!=-1 and list_var[i] not in var_rean_s:
                var_rean_s.append(list_var[i])
            """
            var_name.append(var)
        print(var_name,dt)
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
        
        
        #"""
        """
        if file.find("ghw")!=-1:
            level_co=main_data.coords['level']
            for lvl in level_co:
                globals()[f'{dataset}_{lvl}']=main_data.sel(level=lvl)
        else:
            continue
        """
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
                if name.find("twcr")!=-1:
                    #i=i+1
                    print(name)
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
    ret/

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
    print(city,name)
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
        print(t)
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
#import metpy.calc as mpcalc
#from metpy.units import units
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.cm as cm

def plot_lvl(dt,p,tg,dlat,ulat,dlon,ulon,ty=None,l=None,ss=None,z=None,rg=None):
    #Get the high resolution of the background
    b_img="/work/users/student6/tam/back_pic"
    os.environ["CARTOPY_USER_BACKGROUNDS"] = "/work/users/student6/tam/back_pic/"
    map_pro = ccrs.PlateCarree()
    fig=plt.figure()
    ax = plt.subplot(111,projection=map_pro)
    #ax.background_img(name="ETO", resolution="high")
    ax.stock_img()
    leg=[]
    
    def d_plt(dataset,p):
        dset_test=dataset#-twcr_sst_base
        lat_data,lon_data=co(dset_test,up_lat,down_lat,up_lon,down_lon)
        dset_plot=dset_test.sel(lon=lon_data,lat=lat_data)
        #ax here is for level
        #if your var is sst, don't mean again
        if p=="sst": #this has no level
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
        tg_base=None
    #ax=0 here is for the time
    twcr_test=data_out(dt,pra=par,ax=0,season=ss,level=l,t=tg)
    twcr_base=data_out(dt,pra=par,ax=0,season=ss,level=l,t=tg_base,base="base")
    #twcr_sst_DJF_45=data_out(dataset="twcr",pra="sst",season="DJF",t=1945)

    #PREPARE WITH LIMIT LON,LAT
    if f'{ty}'.find("ano")!=-1:
        dset_test,lat,lon=d_plt(twcr_test-twcr_base,par)
    else:
        dset_test,lat,lon=d_plt(twcr_test,par)
    #dset_test_2=d_plt(twcr_test)[0]
    dset_base,lat,lon=d_plt(twcr_base,par)

    
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
    print(dset_plot)
    
    
    #PLOT
    try:
        #CONTOUR FOR SST AND HEIGHT AND HUMIDITY
        #QUIVER FOR WIND
        print(p)
        if p=="w":
            LON,LAT=np.meshgrid(lon,lat)
            twcr_test_2=data_out(dt,pra=par1,ax=0,season=ss,level=l,t=tg)
            dset_plot_2,lat_2,lon_2=d_plt(twcr_test_2,par)
            #widths = np.linspace(0, 1, LON.size)
            plt.quiver(LON,LAT,dset_plot,dset_plot_2,transform=map_pro,scale=50, zorder=6,pivot='middle', cmap='jet' )
            plt.colorbar()
            #windspeed = (uwind ** 2 + vwind ** 2) ** 0.5
            
            
            print("OH!")
            
        else:
        #OTHER CASE    
            if f'{ty}'.find("line")!=-1:
                pl=dset_plot.plot.contour(ax=ax,colors="black",linewidths=2,levels=1)
                p1l=base_plot.plot.contour(ax=ax,colors="r",linewidths=2,linestyles="dashed",levels=1)
                #use when it requires proxy artists
                main_line = mlines.Line2D([], [], color='black',
                          markersize=15, label='main')
                base_line = mlines.Line2D([], [], color='r',
                          markersize=15,linestyle="dashed", label='base')
                ax.legend(handles=[main_line,base_line])

                #ax.clabel(pl, inline=True,fontsize=20,colors="white") #show the number on the line
                #ax.clabel(p1l, inline=True,fontsize=15,colors="white")

            #contour the base
            elif ty.find("ctb")!=-1:
                base_plot.plot.contourf(ax=ax)
                
            #contour only for the main dataset or the anomaly    
            elif ty.find("ct")!=-1 or ty.find("ano")!=-1:
                dset_plot.plot.contourf(ax=ax)
        
        

        #cb=plt.colorbar(pl, orientation='horizontal', pad=0, aspect=50)
        #cb.set_label('Temperature (C)')
        #dset_plot.plot.contourf(ax=ax)
        #dset_test.plot()
        #plt.show()
        
        #set the beautiful background
        #to crop the background must use set_extend
        ax.set_extent([down_lon, up_lon,down_lat, up_lat],ccrs.PlateCarree())
        
        #All other feature (coastline,land,borders)
        ax.coastlines()#resolution='10m')
        ax.add_feature(cfeature.LAND)#, facecolor=cfeature.COLORS["land_alt1"])
        ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
        ax.add_feature(cfeature.BORDERS)#, linewidth=borders)#resolution='10m')
        """land_50m = cfeature.NaturalEarthFeature(category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        ax.add_feature(land_50m)"""
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
        
        if tg==None:
            tt=f'{name_pra}{name_l}{name_s}'
        else:
            try:
                tgian=tg.replace("_","/")
            except:
                tgian=tg
            tt=f'{name_pra}{name_l}{name_s}-{tgian}'
            
        ax.set_title(f'{tt}',pad=15)
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
        plt.grid()
        plt.tight_layout()
        
        #ax.gridlines(draw_labels=False,alpha=0.6, dms=False, x_inline=False, y_inline=False)
        #ax.xlabels_top, ax.ylabels_right = False, False
        #ax.xlabel_style, ax.ylabel_style = {'fontsize': 12}, {'fontsize': 12}
        #Gridlines does not print longitude or latitde + can't change label
        
        #test
        plt.savefig(f'{Data_path}/Data/test/test{tg}.jpg',format="jpg",dpi=1000)
        
        #plt.savefig(os.path.abspath(rf'{Data_path}Data/ANA/{p}/{z}m/y_{yr}/{ty}_{tg}.jpg'),format="jpg")
        
        plt.clf()
        #plt.show()
    except Exception as e:
        #print(f"{tg},{p}")
        print(e)
        return e
    return

"""
i=1
while i<13:
    for k in range(1943,1946):
        plot_lvl("twcr","z",tg=f'{i}_{k}',l=500,rg=5,z=5865,ty="line",
         dlat=-13,ulat=40,dlon=80,ulon=180)
        #plt.clf()
        #plot_lvl("twcr","z",tg=f'{i}_{k}',l=500,rg=5,z=5870,ty="line",
         #dlat=-13,ulat=40,dlon=80,ulon=180)
    i=i+1
"""  

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
#for i in range(1,13):
#plot_lvl("twcr",p="z",tg=f"12_1944",l=500,rg=5,z=5865,ty="line",
         #dlat=-15,ulat=40,dlon=80,ulon=180)

plot_lvl(dt="twcr",p="w",tg=f"2_1945",l=925,
         dlat=-15,ulat=40,dlon=80,ulon=180)
# %%
