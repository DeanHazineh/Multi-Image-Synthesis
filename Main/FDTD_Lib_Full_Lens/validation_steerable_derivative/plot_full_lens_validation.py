import mat73
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pickle
from pathlib import Path

import dflat.plot_utilities as gF
import dflat.data_structure as df_struct
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier

dirname = str(Path(__file__).parent) + "/"
savepath = dirname + "saved_Figs/"
lumerical_dat_path = dirname + "Lumerical/"
sim_path = dirname + "optimized_lens.pickle"


def load_fieldsExyz(filename):
    dat = mat73.loadmat(lumerical_dat_path + filename)
    Exyz = dat["Exy"]
    x, y, z = dat["x"], dat["y"], dat["z"]
    if Exyz.ndim == 3:
        Exyz = np.expand_dims(Exyz, 2)

    angle = np.arctan2(np.imag(Exyz), np.real(Exyz))
    fieldx = np.transpose(np.abs(Exyz[:, :, :, 0]) * np.exp(1j * angle[:, :, :, 0]), [2, 1, 0])
    fieldy = np.transpose(np.abs(Exyz[:, :, :, 1]) * np.exp(1j * angle[:, :, :, 1]), [2, 1, 0])

    return fieldx, fieldy, [x, y, z]


def load_dflatFields(propagation_parameters):
    # Although we could load the previously computed PSFs, lets load the lens and recompute
    # incase we want to change output grid or something else
    with open(sim_path, "rb") as fhandle:
        data = pickle.load(fhandle)
        latent_tensor = data["latent_tensor"]
        alpha = data["alpha"].numpy()
        aperture = data["aperture"]

    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 143, "y": 143},
            "ms_dx_m": {"x": 350e-9, "y": 350e-9},
            "radius_m": 25e-6,
            "sensor_distance_m": 1e-3,
            "initial_sensor_dx_m": {"x": 1.25e-6, "y": 1.25e-6},
            "sensor_pixel_size_m": {"x": 1.25e-6, "y": 1.25e-6},
            "sensor_pixel_number": {"x": 600, "y": 600},
            "radial_symmetry": False,
            "diffractionEngine": "fresnel_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    # df_struct.print_full_settings(propagation_parameters)

    point_source_locs = np.array([[0.0, 0.0, z] for z in [1e6]])
    mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense1024_U350_H600", pmin=0.042, pmax=0.875)
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)

    trans, phase = mlp_latent_layer(latent_tensor, [532e-9])
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf0 = psf_intensity[:, 0:1, :, :, :].numpy()
    psf90 = psf_intensity[:, 1:2, :, :, :].numpy()
    phase0 = psf_phase[:, 0:1, :, :, :].numpy()
    phase90 = psf_phase[:, 1:2, :, :, :].numpy()

    psf45 = 0.5 * psf0 + 0.5 * psf90 + np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    psf135 = 0.5 * psf0 + 0.5 * psf90 - np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    psf_intensity = np.concatenate([psf0, psf45, psf90, psf135], axis=1)

    return np.squeeze(trans), np.squeeze(phase), psf_layer.aperture_trans, np.squeeze(psf_intensity), np.squeeze(phase0), np.squeeze(phase90), alpha


def group_format_ax(axlist):
    for ax in axlist:
        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])
    return


def plot_dflat_vs_FDTD():
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 143, "y": 143},
            "ms_dx_m": {"x": 350e-9, "y": 350e-9},
            "radius_m": 25e-6,
            "sensor_distance_m": 1e-3,
            "initial_sensor_dx_m": {"x": 1.25e-6, "y": 1.25e-6},
            "sensor_pixel_size_m": {"x": 1.25e-6, "y": 1.25e-6},
            "sensor_pixel_number": {"x": 600, "y": 600},
            "radial_symmetry": False,
            "diffractionEngine": "fresnel_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    trans, phase, aperture, psf_intensity, phase0, phase90, alpha = load_dflatFields(propagation_parameters)

    dx, dy = gF.get_detector_pixel_coordinates(propagation_parameters)
    extent = [min(dx) * 1e6, max(dx) * 1e6, max(dy) * 1e6, min(dy) * 1e6]
    lx, ly = gF.get_lens_pixel_coordinates(propagation_parameters)
    extentl = [min(lx) * 1e6, max(lx) * 1e6, max(ly) * 1e6, min(ly) * 1e6]
    trans *= aperture
    phase *= aperture

    fig = plt.figure()
    ax = gF.addAxis(fig, 2, 2)
    for i in range(2):
        ax[i].imshow(trans[i], vmin=0, vmax=1, extent=extentl)
        ax[i + 2].imshow(phase[i], vmin=-np.pi, vmax=np.pi, cmap="hsv", extent=extentl)
    plt.savefig(savepath + "dflat_lens.png")
    plt.savefig(savepath + "dflat_lens.pdf")

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(psf_intensity[i], extent=extent)
    group_format_ax(ax)
    plt.savefig(savepath + "dflat_PSFs.png")
    plt.savefig(savepath + "dflat_PSFs.pdf")

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 2)
    ax[0].imshow(phase0, vmin=-np.pi, vmax=np.pi, extent=extent, cmap="hsv")
    ax[1].imshow(phase90, vmin=-np.pi, vmax=np.pi, extent=extent, cmap="hsv")
    group_format_ax(ax)
    plt.savefig(savepath + "dflat_PSFphase.png")
    plt.savefig(savepath + "dflat_PSFphase.pdf")

    #### LUMERICAL FULL FIELD CALCULATIONS
    # Load the lens field
    fieldx, fieldy, coord = load_fieldsExyz("fieldMS.mat")
    reffieldx, reffieldy, coord = load_fieldsExyz("ref_field.mat")
    fieldx /= reffieldx
    fieldy /= reffieldy
    x, y = coord[0:2]
    extent = [min(x), max(x), max(y), min(y)]
    X, Y = np.meshgrid(x, y)
    aperture = np.where(np.sqrt(X**2 + Y**2) <= 25e-6, np.ones_like(X), np.zeros_like(X))

    fig = plt.figure()
    ax = gF.addAxis(fig, 2, 2)
    ax[0].imshow(np.abs(fieldx[0]) * aperture, vmin=0, vmax=1)
    ax[1].imshow(np.abs(fieldy[0]) * aperture, vmin=0, vmax=1)
    ax[2].imshow(np.angle(fieldx[0]) * aperture, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    ax[3].imshow(np.angle(fieldy[0]) * aperture, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    plt.savefig(savepath + "fullLens_lens.png")
    plt.savefig(savepath + "fullLens_lens.pdf")

    # Load the PSF distribution
    fieldx, fieldy, coord = load_fieldsExyz("far_field.mat")
    x, y = coord[0:2]
    x *= 1e6
    y *= 1e6
    idx = 1  # we simulated extra z slices but pick the right one matching the simulation
    extent = [min(x), max(x), max(y), min(y)]

    psf0 = np.abs(fieldx[idx]) ** 2
    psf90 = np.abs(fieldy[idx]) ** 2
    phase0 = np.angle(fieldx[idx])
    phase90 = np.angle(fieldy[idx])
    psf45 = 0.5 * psf0 + 0.5 * psf90 + np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)
    psf135 = 0.5 * psf0 + 0.5 * psf90 - np.sqrt(psf0 * psf90) * np.cos(phase0 - phase90)

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 4)
    ax[0].imshow(psf0, extent=extent)
    ax[1].imshow(psf45, extent=extent)
    ax[2].imshow(psf90, extent=extent)
    ax[3].imshow(psf135, extent=extent)
    group_format_ax(ax)
    plt.savefig(savepath + "fullLens_PSFs.png")
    plt.savefig(savepath + "fullLens_PSFs.pdf")

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 2)
    ax[0].imshow(phase0, vmin=-np.pi, vmax=np.pi, extent=extent, cmap="hsv")
    ax[1].imshow(phase90, vmin=-np.pi, vmax=np.pi, extent=extent, cmap="hsv")
    group_format_ax(ax)
    plt.savefig(savepath + "fullLens_PSFphase.png")
    plt.savefig(savepath + "fullLens_PSFphase.pdf")

    #### Display difference of psf intensity
    fullLens_intensity = np.stack([psf0, psf45, psf90, psf135])
    norm_psf = psf_intensity / np.sum(psf_intensity, axis=(-2, -1), keepdims=True)
    error = np.abs(norm_psf - fullLens_intensity / np.sum(fullLens_intensity, axis=(-2, -1), keepdims=True))
    error = np.where(norm_psf / np.max(norm_psf, axis=(-2, -1), keepdims=True) > 0.05, error / norm_psf, np.empty_like(error) * np.nan)

    out = np.nanmedian(error, axis=(-2, -1))
    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        im = ax[i].imshow(error[i], cmap="magma", extent=extent, vmin=0, vmax=0.5)
    group_format_ax(ax)
    # df_plt.addColorbar(fig, ax[3], im)
    cb_ax = fig.add_axes([0.91, 0.124, 0.04, 0.754])
    fig.colorbar(im, orientation="vertical", cax=cb_ax)
    plt.savefig(savepath + "PSFError.png")
    plt.savefig(savepath + "PSFError.pdf")
    group_format_ax(ax)

    return


if __name__ == "__main__":
    plot_dflat_vs_FDTD()
    plt.close()
