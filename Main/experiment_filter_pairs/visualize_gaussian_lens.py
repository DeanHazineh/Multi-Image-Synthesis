import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import os
import sys

import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.neural_optical_layer as df_neural
import dflat.plot_utilities as gF
import dflat.render_layer as df_im
import dflat.datasets_image as df_test_im

sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import compute_PSFs_with_interference


### Global set
dirname = str(Path(__file__).parent) + "/gaussian_derivative/"
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
    alpha = data["alpha"]
    alpha_mask = data["alpha_mask"]
    latent_tensor = data["latent_tensor"]
    aperture_trans = data["aperture"]
    use_alpha = alpha * alpha_mask


def rotate_alpha(alpha, angle_deg):
    return np.matmul(np.array([[np.cos(angle_deg * np.pi / 180), np.sin(angle_deg * np.pi / 180)]]), alpha)


def plot_lens():
    trans, phase = mlp_latent_layer(latent_tensor, [532e-9])
    fig = plt.figure(figsize=(20, 20))
    ax = gF.addAxis(fig, 2, 2)
    ax[0].imshow(phase[0, 0, :, :] * aperture_trans[0], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    ax[1].imshow(phase[0, 1, :, :] * aperture_trans[0], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    ax[2].imshow(trans[0, 0, :, :] * aperture_trans[0], vmin=0, vmax=1)
    ax[3].imshow(trans[0, 1, :, :] * aperture_trans[0], vmin=0, vmax=1)
    plt.savefig(savepath + "lens.png")
    plt.savefig(savepath + "lens.pdf")

    return


def plot_psfs():
    wavelength_set_m = [532e-9]
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1.0e-3,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 1.5e-6, "y": 1.5e-6},
            "sensor_pixel_size_m": {"x": 1.5e-6, "y": 1.5e-6},
            "sensor_pixel_number": {"x": 400, "y": 400},
            "radial_symmetry": False,
            "diffractionEngine": "fresnel_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    df_struct.print_full_settings(propagation_parameters)

    point_source_locs = np.array([[0.0, 0.0, 1e6]])
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)
    trans, phase = mlp_latent_layer(latent_tensor, wavelength_set_m)
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
    phase0 = psf_phase[:, 0:1, :, :, :]
    phase90 = psf_phase[:, 1:2, :, :, :]

    # Get additional rotation angles
    new_alpha = np.concatenate((use_alpha, rotate_alpha(use_alpha, 45), rotate_alpha(use_alpha, 135)), axis=0)
    new_alpha = new_alpha / np.linalg.norm(new_alpha, axis=1, ord=2, keepdims=True)
    net_psf = tf.math.reduce_sum(new_alpha[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis] * psf_intensity, axis=2)

    # Show captured PSF and phase
    dx, dy = gF.get_detector_pixel_coordinates(propagation_parameters)
    extent = [min(dx) * 1e6, max(dx) * 1e6, max(dy) * 1e6, min(dy) * 1e6]
    fig = plt.figure(figsize=(40, 10))
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(psf_intensity[0, i, 0], extent=extent)
        ax[i].scatter(0, 0, marker=".", color="w")
    plt.savefig(savepath + "raw_psfs.png")
    plt.savefig(savepath + "raw_psfs.pdf")

    fig = plt.figure(figsize=(20, 10))
    ax = gF.addAxis(fig, 1, 2)
    ax[0].imshow(phase0[0, 0, 0, :, :], extent=extent, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    ax[1].imshow(phase90[0, 0, 0, :, :], extent=extent, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    plt.savefig(savepath + "raw_phases.png")
    plt.savefig(savepath + "raw_phases.pdf")

    # Show composite PSFs
    fig = plt.figure(figsize=(40, 10))
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(net_psf[i, 0, 0], norm=TwoSlopeNorm(0), cmap="seismic")
        ax[i].set_title(np.array2string(np.round(new_alpha[i, :], 3)))
    plt.savefig(savepath + "net_psfs.png")
    plt.savefig(savepath + "net_psfs.pdf")

    return


def plot_sim_images(SNR):
    wavelength_set_m = [532e-9]
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1.0e-3,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_number": {"x": 500, "y": 500},
            "radial_symmetry": False,
            "diffractionEngine": "fresnel_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    df_struct.print_full_settings(propagation_parameters)

    point_source_locs = np.array([[0.0, 0.0, 1e6]])
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)

    trans, phase = mlp_latent_layer(latent_tensor, wavelength_set_m)
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
    # Get additional rotation angles
    new_alpha = np.concatenate((use_alpha, rotate_alpha(use_alpha, 45), rotate_alpha(use_alpha, 135)), axis=0)

    ### Plot Image Example
    sensor_emv = df_im.BFS_PGE_51S5
    im_layer = df_im.Fronto_Planar_renderer_incoherent(sensor_emv)
    test_im = df_test_im.get_grayscale_image("cameraman.png", {"x": 2000, "y": 2000})  # Ny x Nx x 1
    test_im = test_im[np.newaxis, np.newaxis, np.newaxis, :, :, 0]
    test_im = test_im / np.max(test_im)

    num_photons = df_im.SNR_to_meanPhotons(SNR, sensor_emv)
    meas_im = im_layer(psf_intensity, test_im * num_photons)
    net_im = tf.math.reduce_sum(new_alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis] * meas_im, axis=1)

    # Plot net images
    fig = plt.figure(figsize=(40, 10))
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(net_im[i, 0], norm=TwoSlopeNorm(0), cmap="seismic")
    plt.savefig(savepath + "net_ims" + str(SNR) + ".png")
    plt.savefig(savepath + "net_ims" + str(SNR) + ".pdf")

    fig = plt.figure(figsize=(40, 10))
    ax = gF.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(meas_im[0, i, 0], cmap="gray")
    plt.savefig(savepath + "raw_ims" + str(SNR) + ".png")
    plt.savefig(savepath + "raw_ims" + str(SNR) + ".pdf")

    return


if __name__ == "__main__":

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plot_lens()
    plot_psfs()
    plot_sim_images(30)
    plt.close()
