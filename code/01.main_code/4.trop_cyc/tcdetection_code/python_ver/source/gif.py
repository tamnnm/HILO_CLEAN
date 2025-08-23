file_name = "/data/projects/REMOSAT/tamnnm/iwtrc/ASEAN/grid_0.7/prmsl.1881.nc"
file_name_1 = "/data/projects/REMOSAT/tamnnm/iwtrc/ASEAN/grid_0.7/hgt_thick.1881.nc"
data = xr.open_dataset(file_name)['prmsl'].sel(
    time=slice('1881-09-27', '1881-10-07'), longitude=slice(100, 180))

data = data.where(data < 1008)
data_1 = xr.open_dataset(file_name_1)['gh'].sel(
    time=slice('1881-09-27', '1881-10-07'), longitude=slice(100, 180)).squeeze()  # *10e5
print(data_1.max())
fig, ax = ini_plot(-5, 40, 100, 180, pad=5,
                   figsize=[12, 7], label_size=10, title="Test_trajectory", grid=True)


def plot_element(day):
    data_test = data[day, :, :]
    data_test_2 = data_1[day, :, :]
    if day == 0:
        pl_hgt = data_test.plot.contour(
            ax=ax, colors='red', linestyles='solid', levels=4)  # , linewidths=2, levels=1)
        ax.clabel(pl_hgt, inline=True, fontsize=10)
        pl_hgt_f = data_test_2.plot.contourf(
            ax=ax, cmap='RdYlBu', vmin=-2, vmax=2, levels=21, add_colorbar=False)  # , linewidths=2, levels=1)
    else:
        pl_hgt = data_test.plot.contour(
            ax=ax, colors='red', linestyles='solid', levels=4)
        ax.clabel(pl_hgt, inline=True, fontsize=10)
        pl_hgt_f = data_test_2.plot.contourf(
            ax=ax, cmap='RdYlBu', vmin=-2, vmax=2, levels=21, add_colorbar=False)
    return pl_hgt, pl_hgt_f


Nf = data['time'].size

# print(Nf)
# for i in range(Nf):
#     if i != 0:
#         for pl in pl_hgt.collections:
#             pl.remove()
#         # for pl in pl_hgt_f.collections:
#         #     pl.remove()
#         for txt in ax.texts:
#             txt.remove()
#         # cb.remove()
#     time = data['time'][i].values
#     plt.title(f'{time}', pad=25)
#     pl_hgt = plot_element(i)
#     # cb = plt.colorbar(pl_hgt_f, ax=ax)
#     plt.savefig(image_path+f"test_{i}.png")

pl_hgt, pl_hgt_f = plot_element(0)
cb = plt.colorbar(pl_hgt_f, ax=ax)


def animate(day):
    # print(data_hgt.values)
    global pl_hgt, pl_hgt_f, cb
    for pl in pl_hgt.collections:
        pl.remove()
    for pl in pl_hgt_f.collections:
        pl.remove()
    for txt in ax.texts:
        txt.remove()
    cb.remove()
    time = data['time'][day].values
    pl_hgt, pl_hgt_f = plot_element(day)
    ax.clabel(pl_hgt, inline=True, fontsize=10)
    cb = plt.colorbar(pl_hgt_f, ax=ax)
    # Set the number on the contour
    # ax.clabel(pl_hgt, inline=True, fontsize=10)
    # If the variable dimension is not important for the plot, then you can use the ... placeholder.
    plt.title(f'{time}', pad=25)
    # plt.savefig(os.path.abspath(
    #    rf'/work/users/tamnnm/code/main_code/trop_cyc/img/era5_{z_p}hPa_{day}_11.jpg'), format="jpg", dpi=300)
    # plt.close()
    return pl_hgt, pl_hgt_f


anim = FuncAnimation(fig, animate, frames=Nf, repeat=True)
# , writer=FFMpegWriter()
writervideo = PillowWriter(fps=3)
anim.save(image_path+f'HG_1881_animation_f.gif', writer=writervideo)
