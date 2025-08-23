#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import subprocess
import concurrent.futures as con_fu
#The data only spans from 1900 to 2010
import os

os.chdir('/content/drive/MyDrive/Code/wrf/ecmwf/')
path=os.getcwd()
print(path)

# %%Function:

server = ECMWFDataServer(
    url="https://api.ecmwf.int/v1",
    key="0ee4b55bb4663e3800eec2a7fcb19c18",
    email="nguyenngocminhtam1510@gmail.com",
)
year_full=[1930,1945,1954,1971,1972]
vars_pressure=[129.128,157.128,133.128,130.128,131.128,132.128,138.128,135.138]
vars_sfc=[168.128,167.128,165.128,166.128,151.128,34.128,235.128,134.128,139.128,170.128,183.128,236.128,39.128,40.128,41.128,42.128,89.228]
def generate_date_range(year):
  date=""
  date=date+f'{year}-01-01/to/{year}-12-31'
  return date

def era_retrieve(year_need):
  date_range=generate_date_range(year_need)
  for variable in vars_pressure:
    name_var = str(variable).split('.')[0] 
    filename = f'era/era_{year_need}_{name_var}.grb'
    if os.path.exists(path + filename):
      continue
    else:
      print(f"Trying to download {filename}")
      try: 
        server.retrieve({
          'class':'e2',
          'dataset':'era20c',
          'date':f'{date_range}',
          'expver':'1',
          'levtype':'pl',
          'levelist':'10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000',
          'param':f'{variable}',
          'stream':'oper',
          'time':'00:00:00/06:00:00/12:00:00/18:00:00',
          'type':'an',
          'grid':'0.125/0.125',
          'format':'grib1',
          'target':f'{path}/{filename}'
        })

        print(f"Download {filename}")
      except:
        print(f'Transfering {filename}')

  for variable in vars_sfc:
    name_var = str(variable).split('.')[0]
    filename = f'era/era_{year_need}_{name_var}.grb'
    if os.path.exists(path + filename):
      continue
    else:
      print(f"Trying to download {filename}")
      try: 
        server.retrieve({
          'class':'e2',
          'dataset':'era20c',
          'date':f'{date_range}',
          'expver':'1',
          'levtype':'sfc',
          'param': f'{variable}',
          'stream':'oper',
          'time':'00:00:00/06:00:00/12:00:00/18:00:00',
          'type':'an',
          'grid':'0.125/0.125',
          'format':'grib1',
          'target':f'{path}/{filename}'
          })

      except:
        print(f'Transfering {filename}')

  return

def cera_retrieve(year_need):
  date_range = generate_date_range(year_need)
  for variable in vars_pressure:
    name_var = str(variable).split('.')[0]
    filename = f'cera/cera_{year_need}_{name_var}.grb'
    if os.path.exists(path + filename):
      continue
    else:
      print(f"Trying to download {filename}")
      try: 
        server.retrieve({
          'class':'ep',
          'dataset':'cera20c',
          'date':f'{date_range}',
          'number':'0/1/2/3/4/5/6/7/8/9',
          'expver':'1',
          'levtype':'pl',
          'levelist':'10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000',
          'param': f'{variable}',
          'stream':'enda',
          'type':'an',
          'grid':'0.125/0.125',
          'format':'grib1',
          'target':f'{path}/{filename}'
          })
      except:
        print(f'Transfering {filename}')

  for variable in vars_sfc:
    name_var = str(variable).split('.')[0]
    filename = f'cera/cera_{year_need}_{name_var}.grb'
    if os.path.exists(path + filename):
      continue
    else:
      print(f"Trying to download {filename}")
      try: 
        server.retrieve({
          'class':'ep',
          'dataset':'cera20c',
          'date': f'{date_range}',
          'number':'0/1/2/3/4/5/6/7/8/9',
          'expver':'1',
          'levtype':'sfc',
          'param': f'{variable}',
          'stream':'enda',
          'time':'00:00:00/06:00:00/12:00:00/18:00:00',
          'type':'an',
          'grid':'0.125/0.125',
          'format':'grib1',
          'target':f'{path}/{filename}'
        })
      except:
        print(f'Transfering {filename}')
  
  return

def invariant_retrieve():
  server.retrieve({
    'class':'e2',
    'dataset':'era20c',
    'date':'1900-01-01',
    'param':'129.128',
    "levtype": "sfc",
    'stream':'oper',
    'time':'00:00:00',
    #'grid':'0.125/0.125',
    'format':'grib1',
    'target':'era/era_invariant.grb'
  })
  
  
  server.retrieve({
    'class':'ep',
    'dataset':'cera20c',
    'date':'1901-01-01',
    'number':'0',
    'param':'129.128',
    "levtype": "sfc",
    'stream':'enda',
    'time':'00:00:00',
    #'grid':'0.125/0.125',
    'format':'grib1',
    'target':'cera/cera_invariant.grb'
  })
  return
def main():
  with con_fu.ThreadPoolExecutor() as executor:
    futures =[]
    for year in year_full:
      #print(year)
      futures +=[executor.submit(era_retrieve, year)]
      #futures +=[executor.submit(cera_retrieve, year)]
    #futures +=[executor.submit(invariant_retrieve)]
    con_fu.wait(futures)

  print("useless")
  
if __name__ == "__main__":
  main()