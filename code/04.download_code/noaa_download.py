# --------------------- DOWNLOAD NOAA PARALLEL --------------------- #
var_list  = ['rhum.2m']
def download_file(year,var):
    if os.path.exists(f'{var}.{year}.nc'):
        return
    print(f"Downloaded: {year}")
    subprocess.run(f'wget https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/2mSI/{var}.{year}.nc -O {var}.{year}.nc', shell=True)

# for filename in filelist:
#     print("Start")
#     subprocess.run(f'wget {filename} -O {filename.split("/")[-1]}', shell=True)
#     print(filename.split("/")[-1])
    # subprocess.run(f'tar -xf {filename.split("/")[-1]}', shell=True)

# List of years to download
years = (1881,)

# # Use ThreadPoolExecutor to download files in parallel
with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust max_workers based on your system's capability
    executor.map(download_file, years, var_list)
# #!wget https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/miscSI/prmsl.1881.nc -o prmsl.1881.nc


def noaa_download(year):
  if os.path.exists(f'hgt.{year}.nc'):
    return

def main():
  with con_fu.ThreadPoolExecutor() as executor:
    futures =[]
    #for var in var_base:
    #  for level in level_special:
    #    futures +=[executor.submit(era5_retrieve_base,type_folder='base',var=var,level=level)]
    #for year in year_base:
    #  futures +=[executor.submit(era5_retrieve_base,typefolder='sfc_base',year=None)]
    #for year in year_full:
    #  futures +=[executor.submit(era5_retrieve, year,'synop')]
    #futures +=[executor.submit(era5_retrieve, 1964,'partial')]
    for year in range(1961,1991):
      futures += [executor.submit()]
    con_fu.wait(futures)

  print("useless")

