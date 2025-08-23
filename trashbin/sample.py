def set_up_plot(dlat, ulat, dlon, ulon, pad):
    map_pro = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.subplot(111, projection=map_pro)
    ax.set_xticks(np.arange(dlon, ulon, pad), crs=map_pro)
    ax.set_yticks(np.arange(dlat, ulat, pad), crs=map_pro)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # Tick for axis
    ax.tick_params(axis='both', which="major", labelsize=15)
    # change fontsize of tick when have major
    # """
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    # Draw border, coasral line,...
    ax.coastlines(resolution='10m')
    # , linewidth=borders)#resolution='10m')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    # axins.add_feature(cfeature.LAND,)#, facecolor=cfeature.COLORS["land_alt1"])
    # axins.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
    return fig, ax


shp_path = "/work/users/tamnnm/geo_info/vnm/full_shp/vnm_admbnda_adm0_gov_20200103.shp"
# vnmap = shp.Reader(shp_path)
# sharex, sharey so that the suplot use the same x-bar or y-bar. The gridspec_kw set to 0 so that it delete the space between the suplots
# plot vietnam and the station point
txt_shapes = []
for vnmapshape in vnmap.shapeRecords():
    listx = []
    listy = []
    # parts contains end index of each shape part
    parts_endidx = vnmapshape.shape.parts.tolist()
    parts_endidx.append(len(vnmapshape.shape.points) - 1)
    for i in range(len(vnmapshape.shape.points)):
        x, y = vnmapshape.shape.points[i]
        if i in parts_endidx:
            # we reached end of part/start new part
            txt_shapes.append([listx, listy])
            listx = [x]
            listy = [y]
        else:
            # not end of part
            listx.append(x)
            listy.append(y)

# """
for zone in txt_shapes:
    x, y = zone
    # Plot only the border
    ax.plot(x, y, color="k", markersize=10e-6, linewidth=0.4)
    # Fill the inside with a certainc color
    # ax.fill(x,y,facecolor='red',edgecolor="k",linewidth=0.4)
ax.text(112, 10, "Spratly Islands", fontsize=5, rotation=45)
ax.text(111, 14.5, "Paracel Islands", fontsize=5, rotation=45)
ax.text(110, 13, "EAST SEA", fontsize=5, rotation=90, alpha=0.6)
