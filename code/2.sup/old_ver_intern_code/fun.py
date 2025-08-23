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
    return

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
            twcr_test_2=data_out(dt,pra=p1,ax=a,season=ss,level=l,t=time)
            ax.quiver(lon, lat, dset_test, twcr_test_2, transform=map_pro)
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