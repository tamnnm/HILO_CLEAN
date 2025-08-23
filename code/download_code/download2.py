#%% Define the year + function to get data
#!/usr/bin/env python

from ecmwfapi import ECMWFDataServer
import cdsapi
import numpy as np
#The data only spans from 1900 to 2010

server = ECMWFDataServer()
year_focus=[1943,1944,1945,1946]
#ls -ayear_focus=[1943]
year_full=[1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943,1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960]
#year_90=[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]
#year_00=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
month=np.arange(1,13)
a=0

def time(year):
    date=""
    for i in range(len(year)):
        for j in range(len(month)):
            if month[j]<10:
                date=date+f'{year[i]}0{month[j]}01/'
            else:
                if i==len(year)-1 and j==len(month)-1:
                    date=date+f'{year[i]}{month[j]}01'
                else:
                    date=date+f'{year[i]}{month[j]}01/'
    return date
        #print(date)
        #a=a+1
#Geo(500-200), Temp(1000), SpecHu, Prep, U-V Wind (925,850,700,200)
#ATTENTION:format=NetCDF cannot be called on ecmwf machine 
def era(year_need):
    t=time(year_need)
    print(t)
    server.retrieve({
    "class": "e2",
    "dataset": "era20c",
    "date": f'{t}',
    "expver": "1",
    "levtype": "pl",
    "levelist": "500",
    "param": "129.128",
    "stream": "moda",
    "type": "an",
    "grid": "0.125/0.125",
    #"area":"35/64/-13/161",   
    "target": "era_z",
    })
    server.retrieve({
    "class": "e2",
    "dataset": "era20c",
    "date": f'{t}',
    "expver": "1",
    "levtype": "pl",
    "levelist": "750/825/925",
    "param": "131.128/132.128/133.128",
    "stream": "moda",
    "type": "an",
    "grid": "0.125/0.125",
    #"area":"35/64/-13/161",   
    "target": "era_hw",
    })     
def cera(year_need):
    t=time(year_need)
    print(t)
    server.retrieve({
    "class": "ep",
    "dataset": "cera20c",
    "date": f'{t}' ,
    "expver": "1",
    "levtype": "pl",
    "number": "9",
    "param": "129.128","levelist": "500",
    "stream": "edmo",
    "grid": "0.125/0.125",
    #"area":"35/64/-13/161",
    "type": "an",
    "target": "cera_z",
})
    server.retrieve({
    "class": "ep",
    "dataset": "cera20c",
    "date": f'{t}' ,
    "expver": "1",
    "levtype": "pl",
    "number": "9",
    "levelist": "750/825/925",
    "param": "131.128/132.128/133.128",
    "stream": "edmo",
    "grid": "0.125/0.125",
    #"area":"35/64/-13/161",
    "type": "an",
    "target": "cera_hw",
})
#%% execution of retrieving data 
#era(year_focus,"era_famine")
#cera(year_focus,"cera_famine")
#era(year_full)
cera(year_full)
#era(year_90,"era_base")
#era(year_00,"era_base")

# %%
