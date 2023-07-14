import mat73
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pickle
import os
from pathlib import Path

import dflat.plot_utilities as gF
import dflat.data_structure as df_struct
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier

dirname = str(Path(__file__).parent) + "/"
savepath = dirname + "saved_Figs/"
lumerical_dat_path = dirname + "Lumerical/"
sim_path = dirname + "optimized_lens.pickle"

####
def load_fieldsExyz(filename):
    dat = mat73.loadmat(lumerical_dat_path + filename)
    Exyz = dat["Exy"]
    x, y, z, f = dat["x"], dat["y"], dat["z"], dat["f"]
    if Exyz.ndim == 3:
        Exyz = Exyz[:, :, :, np.newaxis]

    angle = np.arctan2(np.imag(Exyz), np.real(Exyz))
    fieldx = np.transpose(np.abs(Exyz[:, :, 0, :]) * np.exp(1j * angle[:, :, 0, :]), [2, 1, 0])
    fieldy = np.transpose(np.abs(Exyz[:, :, 1, :]) * np.exp(1j * angle[:, :, 1, :]), [2, 1, 0])

    return fieldx, fieldy, [x, y, z, f]


def load_simFields(propagation_parameters):
    # Although we could load the previously computed PSFs, lets load the lens and recompute
    # incase we want to change output grid or something else
    with open(sim_path, "rb") as fhandle:
        data = pickle.load(fhandle)
        latent_tensor = data["latent_tensor"]
        alpha = data["alpha"].numpy()
        aperture = data["aperture"]

    wavelength_set_m = propagation_parameters["wavelength_set_m"]
    point_source_locs = np.array([[0.0, 0.0, z] for z in [1e6]])
    mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense1024_U350_H600", pmin=0.042, pmax=0.875)
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)

    trans, phase = mlp_latent_layer(latent_tensor, wavelength_set_m)
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf0 = psf_intensity[:, 0:1, :, :, :].numpy()
    psf90 = psf_intensity[:, 1:2, :, :, :].numpy()
    phase0 = psf_phase[:, 0:1, :, :, :].numpy()
    phase90 = psf_phase[:, 1:2, :, :, :].numpy()

    psf45 = 0.5 * psf0 + 0.5 * psf90 + np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    psf135 = 0.5 * psf0 + 0.5 * psf90 - np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    psf_intensity = np.concatenate([psf0, psf45, psf90, psf135], axis=1)

    # Convert radial profile to 2D so we can plot in 2D
    aperture = psf_layer.aperture_trans

    trans = df_fourier.radial_2d_transform(trans)
    phase = df_fourier.radial_2d_transform(phase)
    aperture = df_fourier.radial_2d_transform(aperture)

    return np.squeeze(trans), np.squeeze(phase), aperture, np.squeeze(psf_intensity), np.squeeze(phase0), np.squeeze(phase90), alpha


def group_format_ax(axlist, lim=[-250, 250]):
    for ax in axlist:
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    return


def plot_dflat_vs_FDTD():
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": np.arange(500e-9, 601e-9, 20e-9),
            "ms_samplesM": {"x": 143, "y": 143},
            "ms_dx_m": {"x": 350e-9, "y": 350e-9},
            "radius_m": 25e-6,
            "sensor_distance_m": 0.75e-3,
            "initial_sensor_dx_m": {"x": 1.0e-6, "y": 1.0e-6},
            "sensor_pixel_size_m": {"x": 1.0e-6, "y": 1.0e-6},
            "sensor_pixel_number": {"x": 500, "y": 500},
            "radial_symmetry": True,
            "diffractionEngine": "fresnel_fourier",
        }
    )
    num_wl = len(propagation_parameters["wavelength_set_m"])
    trans, phase, aperture, psf_intensity, phase0, phase90, alpha = load_simFields(propagation_parameters)

    # Display metasurface optical response
    lx, ly = gF.get_lens_pixel_coordinates(propagation_parameters)
    extentl = [min(lx) * 1e6, max(lx) * 1e6, max(lx) * 1e6, min(lx) * 1e6]
    trans *= aperture
    phase *= aperture

    fig = plt.figure(figsize=(num_wl * 5, 20))
    ax = gF.addAxis(fig, 4, num_wl)
    for i in range(num_wl):
        ax[i].imshow(trans[i, 0], vmin=0, vmax=1, extent=extentl)
        ax[i + num_wl * 1].imshow(trans[i, 1], vmin=0, vmax=1, extent=extentl)
        ax[i + num_wl * 2].imshow(phase[i, 0], vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=extentl)
        ax[i + num_wl * 3].imshow(phase[i, 1], vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=extentl)
    plt.savefig(savepath + "dflat_lens.png")
    plt.savefig(savepath + "dflat_lens.pdf")

    # Display PSF
    net_psf = np.sum(psf_intensity * alpha[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    dx, dy = gF.get_detector_pixel_coordinates(propagation_parameters)
    extent = [min(dx) * 1e6, max(dx) * 1e6, max(dy) * 1e6, min(dy) * 1e6]
    fig = plt.figure(figsize=(num_wl * 5, 5))
    ax = gF.addAxis(fig, 1, num_wl)
    for i in range(num_wl):
        ax[i].imshow(net_psf[i], extent=extent, norm=TwoSlopeNorm(0), cmap="seismic")
    group_format_ax(ax)
    plt.savefig(savepath + "dflat_psfs.png")
    plt.savefig(savepath + "dflat_psfs.pdf")

    ########
    #### FDTD ANSYS SIMULATION
    # Display lens
    fieldx, fieldy, coord = load_fieldsExyz("fieldMS.mat")
    reffieldx, reffieldy, coord = load_fieldsExyz("ref_field.mat")
    fieldx /= reffieldx
    fieldy /= reffieldy
    idx = [0, 2, 4, 6, 8, 10]
    fieldx = fieldx[idx, :, :]
    fieldy = fieldy[idx, :, :]

    X, Y = np.meshgrid(*coord[0:2])
    X *= 1e6
    Y *= 1e6
    extent = [np.min(X), np.max(X), np.max(Y), np.min(Y)]
    aperture = np.where(np.sqrt(X**2 + Y**2) <= 25, np.ones_like(X), np.zeros_like(X))
    fig = plt.figure(figsize=(num_wl * 5, 20))
    ax = gF.addAxis(fig, 4, num_wl)
    for i in range(num_wl):
        ax[i].imshow(np.abs(fieldx[i]) * aperture, vmin=0, vmax=1, extent=extent)
        ax[i + num_wl * 1].imshow(np.abs(fieldy[i]) * aperture, vmin=0, vmax=1, extent=extent)
        ax[i + num_wl * 2].imshow(np.angle(fieldx[i]) * aperture, vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=extent)
        ax[i + num_wl * 3].imshow(np.angle(fieldy[i]) * aperture, vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=extent)
    plt.savefig(savepath + "fdtd_lens.png")
    plt.savefig(savepath + "fdtd_lens.pdf")

    # Load combined far field file
    with open(lumerical_dat_path + "far_field.mat", "rb") as fhandle:
        data = pickle.load(fhandle)
        fieldx = data["fieldx"]
        fieldy = data["fieldy"]
        coord = data["coord"]
    x, y = coord[0:2]
    x *= 1e6
    y *= 1e6
    idx = 0  # Grab far field z distance matching the simulation if multiple z slices were computed
    extent = [min(x), max(x), max(y), min(y)]

    psf0 = np.abs(fieldx[:, idx]) ** 2
    psf90 = np.abs(fieldy[:, idx]) ** 2
    phase0 = np.angle(fieldx[:, idx])
    phase90 = np.angle(fieldy[:, idx])
    psf45 = 0.5 * psf0 + 0.5 * psf90 + np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    psf135 = 0.5 * psf0 + 0.5 * psf90 - np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    fdtd_psf_intensity = np.transpose(np.stack((psf0, psf45, psf90, psf135)), [1, 0, -2, -1])
    fdtd_net_psf = np.sum(fdtd_psf_intensity * alpha[np.newaxis, :, np.newaxis, np.newaxis], axis=1)

    fig = plt.figure(figsize=(num_wl * 5, 5))
    ax = gF.addAxis(fig, 1, num_wl)
    for i in range(num_wl):
        ax[i].imshow(fdtd_net_psf[i], extent=extent, norm=TwoSlopeNorm(0), cmap="seismic")
    group_format_ax(ax)
    plt.savefig(savepath + "fdtd_psfs.png")
    plt.savefig(savepath + "fdtd_psfs.pdf")

    plt.close()

    return


if __name__ == "__main__":
    plot_dflat_vs_FDTD()
