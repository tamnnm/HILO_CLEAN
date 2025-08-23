"""
import os

end=".mon.mean.nc"
new_end=".mon.nc"
os.chdir("/work/users/student6/tam/pap25_QA_1945/cal/twcr")
for file_name in os.listdir():
    if file_name.endswith(end):
        new_name=file_name.replace(end,new_end)
        os.rename(file_name,new_name)
"""        
import pyipp
version = pyipp.get_version()
print("IPP library version:", version)
