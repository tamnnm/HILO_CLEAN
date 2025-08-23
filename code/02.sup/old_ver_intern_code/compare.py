# -*- coding: utf-8 -*-
#%% 1.IMPORT STUFF & CALL LIST NAME, LON,LAT OF CITY
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

params = {
	'axes.labelsize': 20,
	'font.size': 20,
    'font.family':'serif',
	'legend.fontsize': 10,
    'legend.loc': 'upper right',
    'legend.labelspacing':0.25,
	'xtick.labelsize': 10,
	'ytick.labelsize': 10,
	'lines.linewidth': 4,
	'text.usetex': False,
	# 'figure.autolayout': True,
	'ytick.right': True,
	'xtick.top': True,

	'figure.figsize': [14, 8], # instead of 4.5, 4.5
	'axes.linewidth': 1.5,

	'xtick.major.size': 15,
	'ytick.major.size': 10,
	'xtick.minor.size': 10,
	'ytick.minor.size': 10,

	'xtick.major.width': 3,
	'ytick.major.width': 3,
	'xtick.minor.width': 3,
	'ytick.minor.width': 3,

	'xtick.major.pad': 15,
	'ytick.major.pad': 5,
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
fol_rean=['era','cera','twcr']
#fol_rean=['twcr']
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
#%% 2.EXTRACT ALL DATA - SEPERATE INTO YEARS, CITY, PARAMETERS
#create new folder within the file
#os.makedirs(direc)
#os.makedirs(temp_fol)
#os.makedirs(prep_fol)

#list for name in temp and prep
direc="/Data/DATA_REAL/Filter_Data"
temp_name=[]
prep_name=[]

all_para=temp_var[1:]+prep_var[1:]
city_T_year=[]
city_P_year=[]
All_name=[]
All=[]
year_full=np.arange(43,54)

"""Produce list to plot"""
"""for i in range(len(all_para)):
    if all_para[i] in temp_var:
        os.makedirs(f'T_pic/{all_para[i]}')
    else:
        os.makedirs(f'P_pic/{all_para[i]}')"""

"""produce list to hold the arranged data 
according to each parameter in order with the city"""
for i in range(len(temp_var[1:])):
    #print(f'{temp_var[i+1]}',f'{prep_var[i+1]}')
    globals()[f'{temp_var[i+1]}']=[]
    #os.makedirs(f'{temp_fol}/{temp_var[i+1]}')
    #os.makedirs(f'{prep_fol}/{prep_var[i+1]}')
    globals()[f'{prep_var[i+1]}']=[]
    
"""Produce list to hold data in each year"""
for i in year_full:
    globals()[f'yr_{i}_T']=[]
    globals()[f'city_{i}_T']=[]
    globals()[f'yr_{i}_P']=[]
    globals()[f'city_{i}_P']=[]
    
    #must include yr_ if we only have number then letter like 43_T is not allowed
    
#Seperate data into multiple csv 
"""
for sheet_name,df in data.items():
    name=f'{direc}{sheet_name}.csv'
    #print(name)
    
    #transfer sheet into csv - ONLY USE THIS TO EXTRACT SHEET INTO CSV
    df.to_csv(name,header=None,index=True)
"""
    
    
#use for listing all the name of the folder
for name in os.listdir(Data_path+direc):
    if name.find('.csv')==-1:
        continue
    else:
        unname=pd.read_csv(f'{Data_path}{direc}/{name}',index_col=0,header=None)
    
    """MUST DO!!!!
    in case that there were "empty"cell but not truly empty
    implement this in order to clear all those unnessacry"""
    
    if len(unname.columns>4):
        unname=unname.iloc[:,0:4]
        if name.find('_T') !=-1:
            unname.columns=temp_var
            #temp.append(unname)
            #we use temp and prep for after rearrange each city
        else:
            unname.columns=prep_var
            #prep.append(unname)
        #print(name[0:-4])
        All_name.append(name[0:-4])
        All.append(unname)
    
    #print(globals()[f'{sheet_name}'])
    #print(f'{sheet_name}')
#print(HN_T)

#"""
#this is to seperate which year+city has temp + prep folder
for i in range(len(All_name)):
    #Set out a few array
    re_arr=[] ## data of year +rearrendeg data
    year=[]
    year_fl=All[i].iloc[:,0].dropna().to_numpy()
    for yr in year_fl:
        yr=int(yr)
        year.append(yr)
    #if All_name[i]=="HN_P":
        #print(year_fl)
        #print(year)
    #set parameter name to use later
    if All_name[i].find('_T') !=-1:
        para=temp_var[1:]
        temp_name.append(All_name[i])
        city_T_year.append(year)
    else:
        para=prep_var[1:]
        prep_name.append(All_name[i])
        city_P_year.append(year)
        
    #set the parameter list to temporarily hold the rearrange data
    #for a in range(len(All[i].columns)-1):
        #globals()[f'para_{a+1}']=[] #first parameter list
    #!this is quite UNNECCESSARY
    
    #year which we have data
    
    #Split data according to the year
    for k in range(0,len(year)):
        #Add time to the dataset
        for b in range(0,12):
            if b<9:
                time=f'19{year[k]}0{b+1}01'
            else:
                time=f'19{year[k]}{b+1}01'
            All[i].loc[k*12+b,'year']=time
        
        all_pa=All[i].iloc[(k*12):((k+1)*12),1:].reset_index(drop=True)
        #print(yr)
        if All_name[i].find('_T') !=-1:
            globals()[f'city_{year[k]}_T'].append(f'{All_name[i][:-2]}')
            globals()[f'yr_{year[k]}_T'].append(all_pa)
        else:
            globals()[f'city_{year[k]}_P'].append(f'{All_name[i][:-2]}')
            globals()[f'yr_{year[k]}_P'].append(all_pa)
            
    #transfer string into timeindex
    #print("before",All[i]['year'])
    All[i]['year']=pd.to_datetime(All[i]['year'],format='%Y%m%d')
    #print("after",All[i]['year'])
    #Split data according to the parameter_city
    for j in range(len(para)):
        #print(f'{para[j]}')
        para_data=[] #To hold data of each parameter after being split
        all_pa_data=[] #to hold data of each year of all the para
        city_year=[] #hold which city in each year
        #hold full range of each para follow each city 
        globals()[f'{All_name[i][:-2]}_{para[j]}']=All[i].iloc[:,[0,j+1]].set_index('year')
        #print(f'{All_name[i][:-2]}_{para[j]}')
        
        #split data as 11 years with each para
        for k in range(0,len(year)):
            #add the data of each year according to each parameter in each city
            pa=All[i].iloc[(k*12):((k+1)*12),j+1].reset_index(drop=True)
            #print(pa)
            para_data.append(pa)
            #all_pa_data.append(all_pa)
        #re_arr.append(year,)
        #globals()[f'{All_name[i]}_{para[i]}']=pd.concat(para_data,axis=1)
        #globals()[f'{All_name[i]}_{para[i]}'].columns=year
        #print(para_data)
        
        #CONCAT transform the array into data frame
        
        city_para=pd.concat(para_data,axis=1)
        city_para.columns=year
        
        #print(All_name[i],city_para)
        #append the data according to each city in each parameter
        globals()[para[j]].append(city_para)
        globals()[f'{All_name[i][:-2]}_{para[j]}_ar']=para_data
        
    #if i==0:
#print(avg)
#print(city_43_T)
#print(yr_43_T)
#print(avg[1])
#print(len(avg[1].columns))
#print(len(daily_max))
def plot_para(z,para,year_to_plot):
    year_plot=[]
    for k in range(len(year_to_plot)):
        yr_plt=[]
        #print(year_to_plot[k])
        for j in range(len(year_to_plot[k])):
            yr_plt.append(year_to_plot[k][j]+1900)
        year_plot.append(yr_plt)
    for i in range(len(para[z].columns)):
        #test_yr=para[z].columns[i]
        #print(test_yr)
        test=para[z].iloc[:,i].to_numpy()
        if year_plot[z][i] in (1943,1944,1945):
            a=1
            if year_plot[z][i]==1943:
                c="orangered"
            elif year_plot[z][i]==1944:
                c="seagreen"
            else:
                c="indigo"
         #the first 3 years (43,44,45) #too-tired to change since HP only have 1947 go on
            plt.plot(month,test,alpha=a,color=c,marker='o')
        #elif i<8:
        else:
            a=0.25
            plt.plot(month,test,alpha=a,marker='o')
    plt.legend(year_plot[z])
    
"""
for b in range(len(all_para)):
    for a in range(len(pre)):
        #print(all_para[b])
        #print(a)
        #print(name)
        if all_para[b] in temp_var:
            folder=f'{Data_path}Data/REAL/T_pic/{all_para[b]}'
            name=temp_name[a][:-2]
            c=r'$T^o$'
            year_plot=city_T_year
        else:
            folder=f'{Data_path}Data/REAL/P_pic/{all_para[b]}'
            name=prep_name[a][:-2]
            year_plot=city_P_year
            if all_para[b]=="pre":
                c="mm"
            elif all_para[b]=="no_day":
                c="number of day"
            else:\
                c="%"
        #print(f'{name}_{all_para[b]}')
        plot_para(a,globals()[all_para[b]],year_plot)
        plt.title(f'{name}_{all_para[b]}')
        plt.xlabel('Month')
        plt.ylabel(c)
        plt.tick_params(axis='x',direction='in')
        plt.savefig(f'{folder}/{name}_{all_para[b]}.jpg',format="jpg")
        plt.clf()
"""

#%% 3.FOCUS REGION OF THE REAL DATA
def focus(dset_name,pra):
    datset=globals()[f'{dset_name}_{pra}'].reset_index()
    datset['yr']=datset['year'].dt.year
    datset=datset[(datset['yr']<1947) &(datset['yr']>1942)].set_index('year')
    datset.drop('yr', inplace=True, axis=1)
    #if pra=="tmp":
        #datset=datset+273.15
    globals()[f'{dset_name}_{pra}_focus']=datset
    return
for dset_name in All_name:
    for pra in all_para:
        #since some city doesn't have the precipitation parameter
        try:
            focus(dset_name[:-2],pra)
        except:
            pass
#%% 4.OFFICIAL FUNCTION BASED ON OBS DATA
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
        #print(i)
    return

#Execute about fuction
data_org(fol_grid)
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
    elif t.find("_")!=-1:
        mo=t.split('_')[0]
        yr=t.split('_')[1]
        time_data= time_co[(time_co.dt.month==int(mo)) & (time_co.dt.year==int(yr))]
        #print(mo,yr)
        #print(time_data)
    else:
        if season=="DJF":
            #for time in time_data:
                #if (time.dt.month==12 & time.dt.year==t) or (time_co.dt.month==(1 or 2) & time_co.dt.year==(t+1)):
            cond=(time_co.dt.month<=2) & (time_co.dt.year==(t+1))
            cond2=(time_co.dt.month==12) & (time_co.dt.year==t)
            time_data=time_co[cond | cond2]
        else:
            time_data=time_co[time_co.dt.year==t]
    
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

map_pro = ccrs.PlateCarree()
fig = plt.figure()
ax = plt.subplot(111,projection=map_pro)
ax.stock_img()


def plot_lvl(dt,p,tg,dlat,ulat,dlon,ulon,ty=None,l=None,ss=None,z=None,rg=None):
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
        p="u"
        p1="v"
    else:
        if p=="sst":
            name_pra="SST"
        elif p=="z":
            name_pra="Geopotential height"
        elif p=="shum":
            name_pra="Humidity"
        p=p
        
    #limit to plot 
    down_lat=dlat
    up_lat=ulat
    down_lon=dlon
    up_lon=ulon

    #ax=0 here is for the time
    twcr_test=data_out(dt,pra=p,ax=0,season=ss,level=l,t=tg)
    twcr_base=data_out(dt,pra=p,ax=0,season=ss,level=l,base="base")
    #twcr_sst_DJF_45=data_out(dataset="twcr",pra="sst",season="DJF",t=1945)

    #PREPARE WITH LIMIT LON,LAT
    if ty.find("ano")!=-1:
        dset_test,lat,lon=d_plt(twcr_test-twcr_base,p)
    else:
        dset_test,lat,lon=d_plt(twcr_test,p)
    #dset_test_2=d_plt(twcr_test)[0]
    dset_base,lat,lon=d_plt(twcr_base,p)

    #EXCLUSIVE FOR POTENTIAL HEIGHT
    #if you take mean in data_out it will create errors
    #"""
    if p=="z":
        if z!=None:
            while rg<=10:
                try:
                    dset_plot=dset_test.where(abs(dset_test.values - z)<=rg)
                    base_plot=dset_base.where(abs(dset_base.values - z)<=rg)
                    break
                except:
                    rg=rg+1
        else:
            raise Exception("There are no range here")
    else:
        dset_plot=dset_test
        base_plot=dset_base
    #"""

    #PLOT
    try:
        #CONTOUR FOR SST AND HEIGHT AND HUMIDITY
        #QUIVER FOR WIND
        if p=="w":
            LON,LAT=np.meshgrid(transpose(lon),transpose(lat))
            twcr_test_2=data_out(dt,pra=p1,ax=a,season=ss,level=l,t=time)
            ax.quiver(LON,LAT, dset_test, twcr_test_2, transform=map_pro)
        else:
        #OTHER CASE    
            if ty.find("line")!=-1:
                pl=dset_plot.plot.contour(ax=ax,colors="black",linewidths=2,levels=1)
                p1l=base_plot.plot.contour(ax=ax,colors="r",linewidths=2,linestyles="dashed",levels=1)
                ax.clabel(pl, inline=True,fontsize=10) #show the number on the line
            
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

        ax.coastlines()
        ax.add_feature(cfeature.LAND)#, facecolor=cfeature.COLORS["land_alt1"])
        #ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
        ax.add_feature(cfeature.BORDERS)#, linewidth=borders)
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
            
        ax.set_title(f'{tt}')
        ax.set_xticks(np.linspace(down_lon, up_lon, 5), crs=map_pro)
        ax.set_yticks(np.linspace(down_lat, down_lat, 5), crs=map_pro)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.savefig(f'{Data_path}Data/ANA/{p}/{ty}_{z}_{tg}.jpg',format="jpg")
        plt.clf()
    except:
        print(f"{tg},{p}")
    return

#for i in range(1,13):
def ani_plot(y):
    for y in range(1943,1946):
        plot_lvl("twcr","z",tg=f'{i}_{y}',l=500,rg=5,z=5865,ty="line",
         dlat=-13,ulat=40,dlon=80,ulon=180)

ani = FuncAnimation(fig,
                    ani_plot,
                    frames=3,
                    interval=100)

#%%10.extract data following each city and each parameter  
class data_city():
    def __init__(self,dataset):
        self.dataset=dataset
    def data_full(self):
        dataset=self.dataset
        #print(dataset)
        main_data=globals()[dataset]
    #Do when time is float and to define time as dataframe
#fix the range of coords:
        if dataset=="GHCN":
            rg=7
        elif dataset in fol_grid or dataset.find("twcr")!=-1:
            rg=1.5
        else:
            rg=0.15    
            
        self.main_data=main_data
        self.rg=rg
            
        def ville(self,city):
            main_data=self.main_data
            rg=self.rg            

            num=city_list.index[(city_list['AC']==city)].values[0]
            #print(num.values)
            lon=city_lon[num]
            lat=city_lat[num]
            #print(lat)
            #print(dataset)
#extract the lat,lon:
            lat_downlim=lat-rg
            lat_uplim=lat+rg
            lon_downlim=lon-rg
            lon_uplim=lon+rg
            #print(dset)
            #wE ONLY CARE ABOUT THIS IN GHCN since this has the beginning date in 1900 but it detects wrongly and state that it starts at 1800 
            #work-around when get the error 'unable to decode times units'

##Hanoi: 20.4 -22.2, 104 106
            #print(lat_downlim,lat_uplim,lon_downlim,lon_uplim)
            lat_data,lon_data=co(main_data,lat_uplim,lat_downlim,lon_uplim,lon_downlim)
            #print(lat_data,lon_data)
            try:
                data_TK_re=main_data.sel(lat=lat_data,lon=lon_data)
            except:
                data_TK_re=main_data.sel(latitude=lat_data,longitude=lon_data)
            
        #print(list(data_TK_re.variables.keys()))
#define the full data global
        
            var_city=var(dataset,data_TK_re,city)
            var_city.var_f()
            return 

        for ct in city_acr:
            ville(self,ct)
        #ville(self,"HN")

#"""
for dset in (fol_rean):
    if dset in fol_rean:
        dset_name=f'{dset}_tp'
        """
        for c in cls:
            if dset_name=='twcr_ghw':
                continue
            else:
                dset_name=f'{dset}_{c}'
        """       
    else:
        dset_name=dset
    dc=data_city(dset_name)
    dc.data_full()
#"""
"""
dset="GHCN"
dc=data_city(dset)
dc.data_full()
"""   
#%%11.plot to compare with the real data

"""
We combine all these datasets into a dataframe to easily manipulate the graph (create twinx, change the ticks,....)
"""
import matplotlib.dates as mdates
def plot_ss(ct,pra,fd):
    #"""
    """fig=plt.figure()
        ax=fig.add_subplot(111, label="1")"""
    fig, ax = plt.subplots()
    
    if pra=="t2m":
        pra_ob="tmp"
    else:
        pra_ob=pra

    dt_plt=[] #hold the column to put into a dataframe
    lg_plt=[] #hold the legend
    lines=[] #hold all the plots then extract the legend
    
    if fd=="grid":
        col=["m","#e6d800","#00bfa0"]
        fold="GRID/"
        for i in range(len(fol_grid)):
            if pra in ("t2m","tmp") and fol_grid[i]=="GHCN":
                continue
            else:  
                data_TK_mean=data_out(fol_grid[i],pra,(1,2),city=ct)            
                data_TK_plot=data_TK_mean.to_pandas()
                dts = data_TK_plot.index.to_datetime()
                ax.plot(dts,data_TK_plot,alpha=0.7,label=f'{fol_grid[i]}',marker='o')
    elif fd=="rean":
        col=["#4daf4a","#ff7f00","#e41a1c"]
        fold="REAN/"
        for i in range(len(fol_rean)):
            data_TK_mean=data_out(fol_rean[i],pra,(1,2),city=ct)            
            data_TK_plot=data_TK_mean.to_pandas()
            dt_plt.append(data_TK_plot)
            lg_plt.append(fol_rean[i]) #get the label of all fol_rean
           
    
    for i in range(len(lg_plt)):
        plot_full=ax.plot(dts,dt_plt.iloc[:,i],alpha=0.8,color=col[i],marker='o')
        lines.append(plot_full)
           
    #TO PLOT REAL AND REAN IN THE SAME GRAPH
    #The problem: even though having the same time series but when plot it appears a little bit difference from each other 
        #to merge/concat use pd.concat (can mix all series, dataframe)
    try:
        real=globals()[f'{ct}_{pra_ob}_focus']
        dt_plt=pd.concat([dt_plt,real],axis=1)
        ax2 = ax.twinx()
        g2=ax2.plot(dts,dt_plt.iloc[:,1],alpha=1,color="#0000ff",marker='o')
        lines.append(g2)
        lg_plt.append("observed")
        #ax.plot(dts,data_plot.iloc[:,0],alpha=0.8,label=f'twcr',color=col[2],marker='o')
        #ax.plot(dts,data_plot.iloc[:,1],alpha=0.8,label=f'observed',color="#0000ff",marker='o')
        subfol="real"
    except:
        subfol="no_real"
    #"""
        
    #"""
    #Solve the problem that the month and year can all appear in graph 
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b') 
    yearsFmt = mdates.DateFormatter('\n\n%Y')  # add some space for the year label
    
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthsFmt)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    #"""
    
    
    #Incase you have twinx plot but you want to present all legend
    ax.legend(lines, lg_plt, loc=0)
    
    #turn the label to 90 degrees
    if pra=="pre":
        plt.title(f'pre_{ct}')
        plt.ylabel('mm')        
    else:
        plt.title(f'temperature_{ct}')
        plt.ylabel('C')#"""
    plt.tick_params(axis='y',labelrotation=90)
    plt.tick_params(axis='x',direction='in')
    plt.xlabel('year')
    """try:
        plt.savefig(f'{Data_path}/Data/{fold}{subfol}/{pra}/{ct}.jpg',format="jpg")
    except:
        plt.savefig(f'{Data_path}/Data/{fold}{subfol}/{pra_ob}/{ct}.jpg',format="jpg")
    """
    plt.tight_layout()
    plt.show()
    #plt.clf()

#REMEMBER

#print(HN_pre_focus)
"""
for ct in city_acr:
    #plot_ss(ct,"pre","grid") #done
    #plot_ss(ct,"t2m","grid") #done
    plot_ss(ct,"pre","rean")
    plot_ss(ct,"t2m","rean")
"""

#to so sanh thi chi lay t2m con tmp(doi thanh sst) dung cho sea surface temperature
#plot_ss("HN","pre","grid")
plot_ss("HN","pre","rean")
#print(data_out("twcr","pre",(1,2),"HN"))

#%% 13.print test
print(HN_pre.values)
#HN_pre.plot()
#%% 14.INVESTIGATION ON CORRELATION BTW OBS AND REAL DATA
from sklearn.metrics import mean_squared_error
import math

def res(x,y):
    res=sst.linregress(x,y)
    return res

def RMSE(x,y):
    mse=mean_squared_error(x,y)
    rmse=math.sqrt(mse)
    return rmse

#Remember that for iloc ex: a.iloc[b] is column, a.iloc[[b]] is row

def drop(data_b,i):
    lost_index=data_b.iloc[[i]].index
    data_f=data_b.drop(labels=lost_index,axis=0)
    return data_f,lost_index
def lost(x):
    data_full=x
    lost_data=[]
    for i in range(data_full.shape[0]-1,-1,-1):
        if np.isnan(data_full.iloc[[i]]).bool()==False:
            continue
        else:
            data_end,lost_index=drop(data_full,i)
            lost_data.append(i)
    #print(lost_data)"""
    return data_end,lost_data


#To get the position of all the index in which the value is nan
HN_rain_me_REAL,lost_data=lost(HN_rain_me)

for a in range(len(lost_data)):
    GHCN_HN_prep_REAL=drop(GHCN_HN_prep,lost_data[a])
    
#print(HN_rain_me_REAL,UDel_HN_prep_REAL[0])

"""Transpose(T) since the original has 131 rows so if we arrayrize it now, there will be 131 arrays
-> therefore we have to transpose then choose the first array which is itself
Since it's [[s,f,g,h,h]] after to_numpy
"""

print(res(HN_rain_me_REAL.T.to_numpy()[0],GHCN_HN_prep_REAL[0]))

"""RESULT: 
    UDel:slope=0.8032110725771288,rvalue=0.81853306777809298,pvalue=7.509402519093109e-33 
    CRU: slope=0.6990777842302427,rvalue=0.8177035054806998, pvalue=9.795296970191982e-33
    GHCN:slope=0.5513068523875881,rvalue=0.8177035054806998, pvalue=9.795296970191982e-33
"""

#print(HN_rain_me.iloc[[0]].values)
#print(type(HN_rain_me.iloc[47].index))
#REAL=HN_rain_me.drop([47],axis=0)
    #return REAL,lost_data
#print(HN_rain_me.index)

"""
print(HN_rain_me.iloc[[1]].index.values)
#HP_rain_me.plot(color='red',label='observed')
"""
#%% 15.TEST FOR PLOT - DO NOT USE IT OFFICIALLY
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


#direction of the data of CRU
os.chdir("D:/INFERNOOOOOOOOO/B3 SA/intern/pap25_QA_1945/ob/")
dataset="UDel"#"GHCN"#
name="precip.mon.total.v501.UDel.nc"#'cru_ts4.05.1941.1950.pre.dat.nc' #"precip.mon.total.v501.UDel.nc" #"precip.mon.total.GHCN.nc"
file=f'{dataset}/{name}'

#open data set + set variables
dset=xr.open_dataset(file)#,decode_times=False)
#print(dset)
#print(dset.time)
"""
#work-around when get the error 'unable to decode times units'
units,reference_date=dset.time.attrs['units'].split('since')
#print(units)
if reference_date.find('19')==-1:
    reference_date='1900-1-1 00:00:0.0'
    dset['time']=pd.date_range(start=reference_date,periods=dset.sizes['time'],freq='M')
#"""
#Do when time is float
dset['time'] = pd.DatetimeIndex(dset['time'].values)

main_data=dset.precip
lat_co=main_data.coords['lat']
lon_co=main_data.coords['lon']
time_co=main_data.coords['time']
#print(main_data.attrs['units'])

#Hanoi: 20.4 -22.2, 104 106
lat_data= (lat_co >=-10) & (lat_co <=22.2)
lon_data= (lon_co >=103) & (lon_co <=120)
time_data= time_co[(time_co.dt.year>1942) & (time_co.dt.year<1954)]
#print(time_data)
data_TK=main_data[:,lat_data][...,lon_data].sel(time=time_data)
data_TK_re=data_TK.resample(time="M").mean()#.dropna()
#print(HN_rain_me)
plt.figure(figsize=(16, 8), dpi=150)
np.mean(data_TK_re*10,axis=(1,2)).plot()
HN_rain_me.plot(color='red',label='observe')
plt.legend(["gridded","observed"])
plt.title('title')
plt.ylabel('ylabel')
plt.xlabel('xlabel')
plt.legend()
plt.show()
plt.show()
#"""
#%%
#%%
# %%
