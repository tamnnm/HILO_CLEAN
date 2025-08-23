for file in *.grib; do
  basename=$(basename "$file")
  cdo -f nc copy "$file" "$basename.nc"
done

