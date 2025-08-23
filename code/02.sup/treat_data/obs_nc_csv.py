# pseudo code

# Dictionary = {
#     "station_name":
#         "no_station": int,
#         "alt_name": list of str,
#         "location": {
#             "lon": float,
#             "lat": float,
#             "elev": float
#         },
#         "variable_1": dict => Dataframe,
#         "variable_2": dict => Dataframe,
#         ...
# }

import pandas as pd
import xarray as xr
from functools import reduce
import ftfy
import unicodedata
import io
import os
import sys
import re
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import math
from my_junk import DataCache
Cache = DataCache()

# Initialize data structures
station_dict = {}  # Main dictionary to store all station data
alt_point_dict = {}  # Dictionary to store alternative point names
list_variables = []  # Track all unique variables
station_counter = 1  # Unique station IDs
csv_folder = "/data/projects/REMOSAT/tamnnm/obs/dat_mon_1920_1953/csv_files/"  # Path to CSV files
netcdf_folder = "/data/projects/REMOSAT/tamnnm/obs/dat_mon_1920_1953/netcdf/"  # Path to save NetCDF files
os.chdir(csv_folder)  # Change working directory to the csv_folder containing CSV files

var_dict = {
    "Tx": "Absolute Maximum temperature",
    "Tn": "Absolute Minimum temperature",
    "P": "Precipitation",
    "RHm": "Mean Relative Humidity",
    "Pnb": "Number of days with Precipitation",
    "Txm": "Mean Maximum temperature",
    "Tnm": "Mean Minimum temperature",
    "RHn": "Minimum Relative Humidity",
}

# Extract list of stations
station_list = [file.split("_")[0].strip().replace("-", "").replace(".", "").replace(" ", "") for file in os.listdir(csv_folder) if file.endswith(".csv")]
# Unique station names
unique_stations = set(station_list)

def build_station_groups(unique_stations):
    """Groups station names based on name containment relationships with special exceptions.

    Processes a list of station names to create logical groupings where:
    - Names that contain other names are grouped together (e.g., 'poulo' and 'poulocondore')
    - Specific exceptions are maintained as separate groups (e.g., 'pakse' and 'pakseng')
    - The shortest name in each group becomes the canonical key

    Args:
        unique_stations (list): List of station names (strings or Path objects).
                               May contain mixed case and special characters.

    Returns:
        dict: A dictionary where:
              - Keys are the shortest name in each group
              - Values are dicts with:
                * 'alt_name': List of alternative names in the group

    Example:
        >>> build_station_groups(['poulo', 'poulocondore', 'pakse', 'pakseng'])
        {
            'poulo': {'alt_name': ['poulocondore']},
            'pakse': {'alt_name': []},
            'pakseng': {'alt_name': []}
        }

    Notes:
        - Handles these special cases explicitly:
          * ('pakse', 'pakseng') → kept separate
          * ('yunanfou', 'yunnanfou') → kept separate
        - Automatically converts Path objects to strings
        - Case-sensitive for grouping but preserves original casing

    Visual flow:
    expand_group(seed="poulo")
    ├─ Finds "poulocondore" contains "poulo"
    │  └─ expand_group(now checking "poulocondore")
    │     ├─ Finds "condore" is in "poulocondore"
    │     └─ No more matches
    └─ Returns {"poulo", "poulocondore", "condore"}
    """

    # Special cases that should NOT be grouped together
    EXCEPTIONS = {
        ("pakse", "pakseng"),
        ("yunanfou", "yunnanfou"),
        ("vinh", "vinhlong")
    }

    station_dict = {}
    remaining_stations = set(station
                        for station in unique_stations)

    def should_group(a, b):
        """Check if two stations should be grouped, considering exceptions"""

        # Check if this pair is in our exceptions list
        for exception_pair in EXCEPTIONS:
            if {a,b} == set(exception_pair):
                return False

        # Default grouping logic
        return a != b and (a in b or b in a)

    def expand_group(seed, current_group, remaining):
        """Recursively expand group with exception handling"""
        new_members = set()

        if not remaining:
            return current_group

        for station in list(remaining):
            for member in current_group:
                if should_group(station, member):
                    new_members.add(station)
                    break

        if new_members:
            remaining -= new_members
            return expand_group(seed, current_group | new_members, remaining)
        return current_group

    def process_remaining(remaining):
        if not remaining:
            return station_dict

        seed = remaining.pop()
        group = expand_group(seed, {seed}, remaining.copy())
        remaining -= group

        group_str = {name for name in group}
        group_key = min(group_str, key=len)

        alt_names = [name
                    for name in group - {group_key}]

        station_dict[group_key] = {'alt_name': alt_names}
        return process_remaining(remaining)

    return process_remaining(remaining_stations)

station_name_dict = build_station_groups(unique_stations)
station_list = list(station_name_dict.keys())
# Create default dictionary for alternative points
for station in station_list:
    alt_point_dict[station] = {"primary": (), "alt_points": [], "close_points": []}


# Helper functions (to be implemented)
def read_csv(file_path: str) -> pd.DataFrame:
    with open(file_path, 'rb') as f:
        content = f.read()  # Replace degree symbol with 'o'
        lines = content.split(b'\n')

    # Metadata extraction
    metadata = ftfy.fix_text(lines[0].decode('latin-1', errors='replace'))  # Read the first line as metadata
    metadata_clean = metadata.replace('°', 'o')  # Replace degree symbol with 'o'

    # Main data processing
    clean_data = [ftfy.fix_text(line.decode('latin-1', errors='replace')) for line in lines[1:] if line.strip()]
    data_df = pd.DataFrame([line.replace('\n','').split(',') for line in clean_data]).apply(pd.to_numeric, errors='coerce')
    data_df = data_df.replace(['-', -99.99], np.nan)  # Use `-99.99` if preferred

    # # Read data into a DataFrame (skip the first row)
    # data_df = pd.read_csv(file_path, skiprows=1, header=None)

    # # Replace "-" or "-99.99" with NaN (or another value, e.g., -99.99)

    return metadata_clean, data_df.reset_index(drop=True)

def format_station_name(raw_name: str) -> str:
    """Standardize station names (e.g., remove hyphens, lowercase)."""
    # Remove leading/trailing whitespace and hypens and dots
    raw_name = raw_name.strip().replace("-", "").replace(".", "")

    if raw_name in station_list:
        return raw_name
    else:
        for station in station_list:
            if raw_name in station_name_dict[station]["alt_name"]:
                return station

    return raw_name

def dms_convert(dms_str: str) -> float:
    """Convert DMS (degrees, minutes, seconds) string to decimal degrees."""
    parts = re.split(r'[o\'"]', dms_str.strip())
    if len(parts) < 2:
        raise ValueError(f"Invalid DMS format: {dms_str}")

    degrees = float(parts[0])
    # Convert minutes to decimal (significant: 2 digits)
    minutes = float(parts[1]) / 60.0

    if len(parts) == 3 and parts[2] != '':
        seconds = float(parts[2]) / 3600.0
        minutes += seconds

    return degrees + round(minutes,2)

def extract_value(pattern, text):
    match = re.search(pattern, text)
    main_metadata = match.group(1).strip()
    return main_metadata if match else None

def extract_location_data(metadata: str) -> tuple:
    """Extract (lon, lat, elev) from file header."""
    # Implementation needed

    lon_pattern = r'(?:long\.?|longitude)\s*:?\s*[^\d]*(\d+o\d+\'?)'
    lat_pattern = r'(?:lat\.?|latitude)\s*:?\s*[^\d]*(\d+o\d+\'?)'
    alt_pattern = r'(?:alt\.?|altitude)\s*:?\s*([\d\.]+)\s*(?:m\.?|meters?)?'

    try:
        lon = dms_convert(extract_value(lon_pattern, metadata))
        lat = dms_convert(extract_value(lat_pattern, metadata))
        alt = extract_value(alt_pattern, metadata)
        lon, lat, alt = float(lon), float(lat), float(alt)
    except Exception as e:
        if "Location data not found" not in metadata:
            error = f"Error extracting location data from metadata: {metadata}"
            print(error)
            print("Exception:", e)
            # raise ValueError(f"Error extracting location data: {metadata}")
        lon = lat = alt = float('nan')  # Default to NaN if extraction fails
    return (lon, lat, alt)

def process_timeseries(year,data: pd.DataFrame, var: str) -> pd.DataFrame:
    """Convert raw data to standardized DataFrame with time index."""

    try:
        # choose 12 columns for 12 months
        data = data.iloc[:, :12]  # Select first 12 columns (months)
        Time_index = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='ME')
        data.columns = Time_index  # Set columns to time index
        # data = data.dropna(axis=1, how='all')  # Drop columns with all NaN values
        data = data.T  # Transpose to have time as index
        data.columns = [var]  # Set column name to variable
        data.index.name = 'time'  # Set index name to 'time'
    except Exception as e:
        raise ValueError(f"Error processing timeseries for year {year}: {e}")
    return data

def check_distance(point1, point2):
    """Calculate the distance between two points using Haversine formula."""
    if allnan(point1) or allnan(point2):
        return float('inf')  # Return infinity if either point is NaN
    radian_point1 = [radians(point1[1]), radians(point1[0])] # lat, lon
    radian_point2 = [radians(point2[1]), radians(point2[0])] # lat, lon
    if radian_point1 == radian_point2:
        return 0.0
    if anynan(point1) or anynan(point2): print("point 1:", point1, "point 2:", point2)
    # Haversine formula
    c = haversine_distances([radian_point1, radian_point2])[1][0]  # Get the distance in radians
    r = 6371  # Radius of Earth in kilometers
    # print(c* r, "km")  # Debug print
    return c * r  # Distance in kilometers

def allnan(series):
    """Check if all elements in a tuple are NaN."""
    return all(np.isnan(x) for x in series)

def anynan(series):
    """Check if any element in a tuple is NaN."""
    return any(np.isnan(x) for x in series)


# Process each CSV file
for file in os.listdir(csv_folder):

    if not file.endswith(".csv"): continue

    # try:
    # Parse filename
    parts = file.split(".csv")[0].split("_")
    if len(parts) < 3:
        continue  # Skip malformed filenames

    city_name, var, year = parts[0], parts[1], parts[2]

    # Skip problematic names
    if " " in city_name or "et" in year:
        continue

    # Standardize name and process data
    final_name = format_station_name(city_name)
    metadata,full_df = read_csv(file)
    if full_df.empty:
        print(f"Skipping empty file: {file}")
        continue

    print('Processing:', file, final_name, var, year)
    ts_data = process_timeseries(year,full_df, var)  # Skip header
    lon, lat, elev = extract_location_data(metadata)  # Assuming first row has location data
    print("Location:",lon, lat, elev)

    # Update dictionary
    if final_name not in station_dict:
        station_dict[final_name] = {
            "no_station": station_counter,
            "alt_name": station_name_dict[final_name]["alt_name"],
            "location": {"lon": lon, "lat": lat, "elev": elev},
            var: {year: ts_data}
        }
        alt_point_dict[final_name]["primary"] = (lon,lat,elev)  # Store primary point
        station_counter += 1
    else:
        # Check the coordinates
        primary_points = alt_point_dict[final_name]["primary"]

        used_flag = True # Flag to indicate if the destination point exists in the station_dict

        if (lon, lat, elev) == primary_points or allnan((lon, lat, elev)):
            chosen_name = final_name

        elif allnan(primary_points):
            # If primary point is NaN, update it with the new coordinates
            alt_point_dict[final_name]["primary"] = (lon, lat, elev)
            chosen_name = final_name
            station_dict[final_name]["location"] = {"lon": lon, "lat": lat, "elev": elev}

        else:
            no_used_points = len(alt_point_dict[final_name]['alt_points'])

            if no_used_points > 0:
                distances = [check_distance((lon, lat, elev), point) for point in alt_point_dict[final_name]["alt_points"]]
                closest_point = np.argmin(distances)
                ref_distance = distances[closest_point]
            else:
                ref_distance = check_distance((lon, lat, elev), primary_points)
                closest_point = -1


            if ref_distance > 15:
                print(f"Warning: {final_name} has a new point that is more than 15 km away from existing points.")
                # Create a new alternative point
                alt_point_dict[final_name]["alt_points"].append((lon, lat, elev))
                chosen_name = final_name + f"_{no_used_points+1}"
                used_flag = False
                print("Option 1")
            else:
                if closest_point >= 0: # If there are actual alternative points
                    chosen_name = final_name + f"_{closest_point+1}"
                    print("Option 2")
                else: # The new points is in 15 km from the primary point
                    chosen_name = final_name
                    alt_point_dict[final_name]["close_points"].append((lon, lat, elev))
                    print("Option 3")
        if used_flag:
            try:
                # Add variable/year data
                if var not in station_dict[chosen_name]:
                    station_dict[chosen_name][var] = {}
            except Exception as e:
                print(f"Error adding variable {var} for station {chosen_name}: {e}")
                print(station_dict.keys())
                print("primary:",primary_points)
                print("alt_points:", alt_point_dict[final_name]["alt_points"])
                print("new point:", (lon, lat, elev))
                print(ref_distance, "km")
                sys.exit(0)

            # If the chosen point already exists, update the dictionary
            station_dict[chosen_name][var][year] = ts_data
        else:
            print(f"Adding new station {chosen_name} with coordinates ({lon}, {lat}, {elev})")
            station_dict[chosen_name] = {
            "no_station": station_counter,
            "location": {"lon": lon, "lat": lat, "elev": elev},
            var: {year: ts_data},
            }
            station_counter += 1

    # Track variables
    if var not in list_variables:
        list_variables.append(var)

    # except Exception as e:
    #     print(f"Error processing {file}: {e}")

# Convert to NetCDF
for variable in list_variables:
    # Prepare metadata and timeseries
    meta_records = []
    ts_records = []

    print(f"Processing variable: {variable}")

    for station, data in station_dict.items():
        if variable in data:

            if math.isnan(data['location']['lon']) or math.isnan(data['location']['lat']):
                print(f"Skipping {station} due to missing location data.")
                continue

            # Collect metadata
            meta_records.append({
                "name_station": station,
                "no_station": data["no_station"],
                "latitude": data["location"]["lat"],
                "longitude": data["location"]["lon"],
                "elevation": data["location"]["elev"]
            })

            # Merge timeseries across years
            combined_ts = data[variable]
            combined_df = pd.concat(combined_ts.values(), axis=0)
            combined_df["no_station"] = data["no_station"]
            ts_records.append(combined_df)

    # Create xarray Dataset
    if ts_records:
        # Extract the column data for the variable (1 column only)
        combined_df = pd.concat(ts_records, axis=0).reset_index().set_index(["time", "no_station"])[variable]
        combined_da = combined_df.to_xarray()

        meta_df = pd.DataFrame(meta_records)

        Cache.save_data(meta_df, os.path.join(netcdf_folder, "meta_data.csv"))
        sys.exit()

        def creat_sup_ds(name, dtype, full_name=None):
            return xr.DataArray(
                data=meta_df[name].values,
                coords={'no_station': meta_df['no_station']},
                dims=['no_station'],
                attrs={'standard_names': name or full_name},
            ).astype(dtype)

        ds = xr.Dataset({
            variable: combined_da,
            "lat": creat_sup_ds("latitude", "float32", "latitude"),
            "lon": creat_sup_ds("longitude", "float32", "longitude"),
            "elev": creat_sup_ds("elevation", "float32", "elevation"),
            "no_station": creat_sup_ds("no_station", "int", "no_station"),
            "station_name": creat_sup_ds("name_station", "U20", "station_name")
        })

        time_min = ds.time.min().dt.year.item()
        time_max = ds.time.max().dt.year.item()
        missing_years_note = "Contains sporadic years (e.g., 1922, 1930-1953) with station-dependent gaps"

        ds.attrs.update({
            "title": f"Station data for {variable}",
            "creator": "Tam Nguyen",
            "created": pd.Timestamp.now().isoformat(),
            "long_name": var_dict.get(variable, "Unknown variable"),
            "temporal_coverage": f"{time_min}-{time_max}",
            "temporal_gaps": missing_years_note,
            "missing_data": "NaN represents missing observations (no fillValue set)",
            "data_warning": "Temporal sampling is irregular - years with no data contain no records (not zero-filled)",
        })

        # Save to NetCDF
        ds.to_netcdf(os.path.join(netcdf_folder,f"{variable}_stations_{time_min}_{time_max}_v2.nc"))