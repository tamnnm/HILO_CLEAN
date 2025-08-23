<<<<<<< HEAD
import os
import pandas as pd
=======
# Produce dataframe like this
# City | Ind | Data1 | Data2 | Data3 | Data4
# ------------------------------------------
# ALUOI| T10p| 0.5   | ....  | ....  |.....
# ------------------------------------------
# HANOI| T10p| 0.5   | ....  | ....  |.....
#  ------------------------------------------
# ALUOI| T90p| 0.5   | ....  | ....  |.....
#  ------------------------------------------
# HANOI| T90p| 0.5   | ....  | ....  |.....
# ...............

import os
import pandas as pd
from constant import *
>>>>>>> c80f4457 (First commit)
filepath = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/para/csv_file/ensemble"
rain_csv = "/work/users/tamnnm/code/01.main_code/01.city_list_obs/city/rain_city.txt"
temp_csv = "/work/users/tamnnm/code/01.main_code/01.city_list_obs/city/temp_city.txt"
file_list = [f for f in os.listdir(
    filepath) if os.path.isfile(os.path.join(filepath, f))]

rain_df = pd.read_csv(rain_csv, names=[
                      'name_station', 'first_year', 'last_year', 'lon', 'lat', 'elev', 'appendix'])
temp_df = pd.read_csv(temp_csv, names=[
                      'name_station', 'first_year', 'last_year', 'lon', 'lat', 'elev', 'appendix'])
<<<<<<< HEAD
temp_tuple = ('Tnn', 'Tnx', 'Txx', 'Txn', 'Tn10p', 'Tn90p', 'Tx90p', 'Tx10p',
              'CSDI', 'WSDI', 'SU25', 'TN20', 'DTR')
rain_tuple = ('R1mm', 'R10mm', 'R20mm',
              'R50mm', 'Rx1day', 'Rx5day', 'R99p', 'R95p', 'CDD', 'CWD', 'SDII', 'PRCPTOT')

# Mapping of categorical values to numerical values
temp_mapping = {value: idx + 1 for idx, value in enumerate(temp_tuple)}
rain_mapping = {value: idx + 1 for idx, value in enumerate(rain_tuple)}

# Reverse the mapping dictionaries
reverse_temp_mapping = {v: k for k, v in temp_mapping.items()}
reverse_rain_mapping = {v: k for k, v in rain_mapping.items()}

for file in file_list:
    file_df = pd.read_csv(f"{filepath}/{file}")
    # Add lat, lon for reference
    merge_df = pd.merge(file_df, rain_df[['name_station', 'lon', 'lat']], on='name_station',
                        how='left')

    # Create a categorical column based on the sort_series
    if 'R_' in file:
        merge_df['ind'] = merge_df['ind'].map(rain_mapping)
    else:
        merge_df['ind'] = merge_df['ind'].map(temp_mapping)

    # Sort the DataFrame by the numerical 'ind' and then by 'lat'
    sorted_df = merge_df.sort_values(
        by=['ind', 'lat', 'lon'], ascending=[True, False, True])

    # Reverse the 'ind' column back to categorical values
    if 'R_' in file:
        sorted_df['ind'] = sorted_df['ind'].map(reverse_rain_mapping)
    else:
        sorted_df['ind'] = sorted_df['ind'].map(reverse_temp_mapping)
=======


def sort_city(file_df, var, sort_ind= True):
    # Add lat, lon for reference

    merge_df = pd.merge(file_df, rain_df[['name_station', 'lon', 'lat']], on='name_station',
                        how='left')
    if sort_ind:
        # Create a categorical column based on the sort_series
        if 'R' in var:
            merge_df['ind'] = merge_df['ind'].map(rain_mapping)
        else:
            merge_df['ind'] = merge_df['ind'].map(temp_mapping)

        # Sort the DataFrame by the numerical 'ind' and then by 'lat'
        sorted_df = merge_df.sort_values(
            by=['ind', 'lat', 'lon'], ascending=[True, False, True])

        # Reverse the 'ind' column back to categorical values
        if 'R' in var:
            sorted_df['ind'] = sorted_df['ind'].map(reverse_rain_mapping)
        else:
            sorted_df['ind'] = sorted_df['ind'].map(reverse_temp_mapping)
    else:
        sorted_df = merge_df.sort_values(
            by=['lat', 'lon'], ascending=[False, True])
>>>>>>> c80f4457 (First commit)

    # Reset the index
    sorted_df.reset_index(drop=True, inplace=True)
    # Drop the specified columns
    try:
        sorted_df.drop(columns=['lon_x',
                                'lat_x', 'lat_y', 'lon_y'], inplace=True)
    except:
        sorted_df.drop(columns=['lon',
                                'lat'], inplace=True)
    # sorted_df.dropna(subset=['ind'], inplace=True)
<<<<<<< HEAD
    print(sorted_df)
    sorted_df.to_csv(f"{filepath}/{file}", index=False)
=======
    return sorted_df

if __name__ == "__main__":
    for file in file_list:
        file_df = pd.read_csv(f"{filepath}/{file}")
        # Add lat, lon for reference
        sorted_df = sort_city(file_df)
        sorted_df.to_csv(f"{filepath}/{file}", index=False)
>>>>>>> c80f4457 (First commit)
