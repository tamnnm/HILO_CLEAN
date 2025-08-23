#merge code

import subprocess
import cdo
import constant as cs
import os
import xarray as xr

#change to the file directory
os.chdir(cs.twcr_dir)

# Initialize CDO module
cdo = cdo.Cdo()

#list file
list_file=os.listdir(cs.twcr_dir)

#creat list for variables:
list_var = []
list_dir = []

# Define input file paths
for fn in list_file:
    if fn.endswith(".nc") == True:
        if (("coords" in fn) or ("test" in fn)) ==True:
            continue
        else:
            print(fn)
            list_var.append(fn.split(".")[0])
            list_dir.append(fn) 
            print(fn.split(".")[0])
    else:
        continue

print("done")

# Define output file path
output_file = 'merged_test.nc'

# Construct CDO command and execute it
syntax_new="-selname,"
for i in range(len(list_var)):
    syntax=syntax_new
    if i != len(list_var) -1:    
        syntax_old=str(list_var[i] + ' '+list_dir[i] + ' -selname,')
    else:
        syntax_old=str(list_var[i] + ' '+list_dir[i])
    syntax_new=syntax+syntax_old
    
print(syntax_new)

#merge
#you cannot put directly into the command because it cannot decipher that syntax is string 
cdo_command= f'cdo merge {syntax_new} {output_file}'
subprocess.run(cdo_command, shell=True)