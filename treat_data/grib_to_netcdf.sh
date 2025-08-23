for file in regrid_era5_*.grib; do
  basename=$(basename "$file")
  cdo -f nc copy "$file" "${basename//.grib/}".nc
done

