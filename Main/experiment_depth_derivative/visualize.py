import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
import tensorflow as tf
import os
import sys
from pathlib import Path

import dflat.data_structure as df_struct
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier
import dflat.render_layer as df_im
import dflat.plot_utilities as gF


sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import compute_PSFs_with_interference


### Unpack some global settings
dirname = str(Path(__file__).parent) + "/output_Two_image"
datpath = dirname + "/optimized_lens.pickle"
savepath = dirname + "/post_trained_images/"
sim_params_path = dirname + "/propagation_parameters.pickle"
mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense512_U350_H600", pmin=0.042, pmax=0.875)

with open(sim_params_path, "rb") as fhandle:
    sim_prop_params = pickle.load(fhandle)

with open(datpath, "rb") as fhandle:
    data = pickle.load(fhandle)
    trans = data["transmittance"]
    phase = data["phase"]
    aperture = data["aperture"]
    alpha = data["alpha"]
    alpha_mask = data["alpha_mask"]
    latent_tensor = data["latent_tensor"]
    use_alpha = alpha * alpha_mask

###
def make_test_im():
    sub_im_nx = 1024
    sub_im_ny = 1024
    x_vect = np.arange(0, sub_im_nx, 1) - (sub_im_nx // 2)
    y_vect = np.arange(0, sub_im_ny, 1) - (sub_im_ny // 2)
    X_vect, Y_vect = np.meshgrid(x_vect, y_vect)

    im = np.where(np.sqrt(X_vect**2 + Y_vect**2) < 250, np.ones(shape=(sub_im_ny, sub_im_nx)), np.zeros(shape=(sub_im_ny, sub_im_nx)))
    im[np.isnan(im)] = 0
    return im


def plot_PSFs():
    ### Load the lens
    sim_settings = {
        "wavelength_set_m": [532e-9],
        "ms_samplesM": {"x": 571, "y": 571},
        "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
        "radius_m": 1.0e-3,
        "sensor_distance_m": 30e-3,
        "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_number": {"x": 200, "y": 200},
        "radial_symmetry": False,
        "diffractionEngine": "fresnel_fourier",
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    propagation_parameters = df_struct.prop_params(sim_settings)
    df_struct.print_full_settings(propagation_parameters)
    zlocs = [33e-3, 35e-3, 37e-3, 39e-3, 41e-3]
    num_z = len(zlocs)
    point_source_locs = np.array([[0.0, 0.0, z] for z in zlocs])

    # Let us recompute the camera PSF in case we want to change some settings for the render
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)
    trans, phase = mlp_latent_layer(latent_tensor, propagation_parameters["wavelength_set_m"])
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
    net_psf = tf.math.reduce_sum(use_alpha[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis] * psf_intensity, axis=2)

    # # Plot the net psf
    dx, dy = gF.get_detector_pixel_coordinates(propagation_parameters)
    extent = [min(dx) * 1e6, max(dx) * 1e6, max(dy) * 1e6, min(dy) * 1e6]
    fig = plt.figure(figsize=(5 * num_z, 5))
    ax = gF.addAxis(fig, 1, num_z)
    for i in range(num_z):
        ax[i].imshow(net_psf[0, 0, i], extent=extent, norm=TwoSlopeNorm(0), cmap="seismic")
        ax[i].scatter(0, 0, marker=".", color="w")
        ax[i].set_title(np.round(np.sum(net_psf[0, 0, i]), 3))
    plt.savefig(savepath + "net_psf.png")
    plt.savefig(savepath + "net_psf.pdf")

    # Plot the psfs
    iter = 0
    fig = plt.figure(figsize=(5 * num_z, 20))
    ax = gF.addAxis(fig, 4, num_z)
    for r in range(4):
        for c in range(num_z):
            ax[iter].imshow(psf_intensity[0, r, c], extent=extent)
            ax[iter].scatter(0, 0, marker=".", color="w")
            iter = iter + 1
    plt.savefig(savepath + "raw_psf.png")
    plt.savefig(savepath + "raw_psf.pdf")

    return


def plot_im_demo(SNR=30):
    sim_settings = {
        "wavelength_set_m": [532e-9],
        "ms_samplesM": {"x": 571, "y": 571},
        "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
        "radius_m": 1.0e-3,
        "sensor_distance_m": 30e-3,
        "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_number": {"x": 700, "y": 700},
        "radial_symmetry": False,
        "diffractionEngine": "fresnel_fourier",
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    propagation_parameters = df_struct.prop_params(sim_settings)
    df_struct.print_full_settings(propagation_parameters)
    zlocs = [33e-3, 35.5e-3, 37e-3, 39e-3]
    point_source_locs = np.array([[0.0, 0.0, z] for z in zlocs])

    # Recompute the PSF from the optimized shape parameters
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)
    trans, phase = mlp_latent_layer(latent_tensor, propagation_parameters["wavelength_set_m"])
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)

    ### Plot Image Example
    sensor_emv = df_im.BFS_PGE_51S5
    im_layer = df_im.Fronto_Planar_renderer_incoherent(sensor_emv)
    num_photons = df_im.SNR_to_meanPhotons(SNR, sensor_emv)
    im1 = make_test_im()[np.newaxis, np.newaxis, np.newaxis, :, :]

    meas_im = im_layer(psf_intensity, im1 * num_photons).numpy()
    meas_im = np.concatenate([np.concatenate([meas_im[0, :, 0], meas_im[0, :, 1]], axis=-2), np.concatenate([meas_im[0, :, 2], meas_im[0, :, 3]], axis=-2)], axis=-1)
    net_im = np.sum(use_alpha[0, :, np.newaxis, np.newaxis] * meas_im, axis=0)
    depthmap = [im1 * zlocs[i] for i in range(4)]
    depthmap = np.squeeze(np.concatenate((np.concatenate(depthmap[0:2], axis=-2), np.concatenate(depthmap[2:4], axis=-2)), axis=-1))

    fig = plt.figure(figsize=(40, 10))
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(meas_im[i], cmap="gray")
    plt.savefig(savepath + "meas_im.png")
    plt.savefig(savepath + "meas_im.pdf")

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 2)
    ax[0].imshow(depthmap, norm=TwoSlopeNorm(36.75e-3), cmap="magma")
    ax[1].imshow(net_im, norm=TwoSlopeNorm(0), cmap="seismic")
    plt.savefig(savepath + "net_im.png")
    plt.savefig(savepath + "net_im.pdf")

    return


def plot_phase_profile():
    fig = plt.figure(figsize=(10, 5))
    ax = gF.addAxis(fig, 1, 2)
    for i in range(2):
        ax[i].imshow(phase[0, i, :, :] * aperture[0])
    plt.savefig(savepath + "phase.png")
    plt.savefig(savepath + "phase.pdf")

    return


if __name__ == "__main__":
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plot_PSFs()
    plot_phase_profile()
    plot_im_demo()
    plt.close()
