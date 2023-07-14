import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import os
import sys

import dflat.data_structure as df_struct
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier
import dflat.plot_utilities as gF
import dflat.render_layer as df_im
import dflat.datasets_image as df_test_im
from dflat.render_layer.core import get_rgb_bar_CIE1931

sys.path.append(os.path.dirname(sys.path[0]))
from src.load_ARAD_sample import load_test_ims
from src.Multi_Image_Synthesis_Optimizers import compute_PSFs_with_interference

### Global settings
dirname = str(Path(__file__).parent) + "/output"
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
    use_alpha = alpha * alpha_mask

### Plot Functions for Paper
def plot_psfs():
    wavelength_set_m = np.array([500e-9, 525e-9, 550e-9, 575e-9, 600e-9])
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1e-3,
            "sensor_distance_m": 30e-3,
            "initial_sensor_dx_m": {"x": 1.0e-6, "y": 1.0e-6},
            "sensor_pixel_size_m": {"x": 1.0e-6, "y": 1.0e-6},
            "sensor_pixel_number": {"x": 800, "y": 800},
            "radial_symmetry": True,
            "diffractionEngine": "fresnel_fourier",
        }
    )
    df_struct.print_full_settings(propagation_parameters)
    point_source_locs = np.array([[0.0, 0.0, 1e6]])
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)

    # recompute the opical response given computed lens
    trans, phase = mlp_latent_layer(latent_tensor, propagation_parameters["wavelength_set_m"])
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
    psf_decomp = use_alpha[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis] * psf_intensity
    net_psf = tf.math.reduce_sum(psf_decomp, axis=2)
    energy = np.squeeze(tf.math.reduce_sum(psf_intensity, axis=(-2, -1)).numpy())

    dx, dy = gF.get_detector_pixel_coordinates(propagation_parameters)
    extent = [min(dx) * 1e6, max(dx) * 1e6, max(dy) * 1e6, min(dy) * 1e6]
    num_wl = len(wavelength_set_m)
    fig = plt.figure(figsize=(5 * num_wl, 20))
    ax = gF.addAxis(fig, 4, num_wl)
    iter = 0
    for p in range(4):
        for i in range(num_wl):
            ax[iter].imshow(psf_intensity[i, p, 0], extent=extent)
            ax[iter].set_title(str(np.round(energy[i, p], 3)))
            ax[iter].axis("off")
            iter = iter + 1
    plt.savefig(savepath + "psfs.png")
    plt.savefig(savepath + "psfs.pdf")

    fig = plt.figure(figsize=(5 * num_wl, 20))
    ax = gF.addAxis(fig, 4, num_wl)
    iter = 0
    for p in range(4):
        for i in range(num_wl):
            ax[iter].imshow(psf_decomp[0, i, p, 0], extent=extent, norm=TwoSlopeNorm(0), cmap="seismic")
            iter = iter + 1
    plt.savefig(savepath + "psf_decomp.png")
    plt.savefig(savepath + "psf_decomp.pdf")

    return


def im_demo(SNR=30):
    wavelength_set_m = np.array([500e-9, 525e-9, 550e-9, 575e-9, 600e-9])
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1e-3,
            "sensor_distance_m": 30e-3,
            "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_number": {"x": 800, "y": 800},
            "radial_symmetry": True,
            "diffractionEngine": "fresnel_fourier",
        }
    )
    num_wl = len(wavelength_set_m)
    point_source_locs = np.array([[0.0, 0.0, 1e6]])
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)

    trans, phase = mlp_latent_layer(latent_tensor, propagation_parameters["wavelength_set_m"])
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)

    ### Plot Image Example
    sensor_emv = df_im.BFS_PGE_51S5
    im_layer = df_im.Fronto_Planar_renderer_incoherent(sensor_emv)
    num_photons = df_im.SNR_to_meanPhotons(SNR, sensor_emv)
    test_im = df_test_im.get_grayscale_image("cameraman.png", {"x": 2000, "y": 2000})  # Ny x Nx x 1
    test_im = test_im[np.newaxis, np.newaxis, np.newaxis, :, :, 0]
    test_im = test_im / np.max(test_im)

    imout = im_layer(psf_intensity, test_im * num_photons)
    net_im = tf.math.reduce_sum(imout * use_alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis], axis=1)

    fig = plt.figure(figsize=(5 * num_wl, 20))
    ax = gF.addAxis(fig, 4, num_wl)
    iter = 0
    for p in range(4):
        for i in range(num_wl):
            ax[iter].imshow(imout[i, p, 0], cmap="gray")
            iter = iter + 1
    plt.savefig(savepath + "im_set.png")
    plt.savefig(savepath + "im_set.pdf")

    fig = plt.figure(figsize=(5 * num_wl, 20))
    ax = gF.addAxis(fig, 1, num_wl)
    for p in range(num_wl):
        ax[p].imshow(net_im[i, 0], norm=TwoSlopeNorm(0), cmap="seismic")
    plt.savefig(savepath + "net_im.png")
    plt.savefig(savepath + "net_im.pdf")

    return


def color_sim(SNR=30):
    wavelength_set_m = np.arange(500e-9, 610e-9, 10e-9)
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1e-3,
            "sensor_distance_m": 30e-3,
            "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_number": {"x": 700, "y": 700},
            "radial_symmetry": True,
            "diffractionEngine": "fresnel_fourier",
        }
    )
    point_source_locs = np.array([[0.0, 0.0, 1e6]])

    mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense256_U350_H600", pmin=0.042, pmax=0.875)
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)
    trans, phase = mlp_latent_layer(latent_tensor, propagation_parameters["wavelength_set_m"])
    psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)

    hsi_im, rgb_im = load_test_ims(wavelength_set_m * 1e9, {"x": 2000, "y": 2000})
    mf_bar = get_rgb_bar_CIE1931(wavelength_set_m * 1e9)
    sensor_emv = df_im.BFS_PGE_51S5
    im_layer = df_im.Fronto_Planar_renderer_incoherent(sensor_emv)
    num_photons = df_im.SNR_to_meanPhotons(SNR, sensor_emv)

    for i in range(hsi_im.shape[0]):
        test_im = np.transpose(hsi_im[i], [2, 0, 1])[:, np.newaxis, np.newaxis, :, :]
        imout = im_layer(psf_intensity, test_im * num_photons, rfft=True)
        net_im = tf.math.reduce_sum(imout * use_alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis], axis=1)

        im_rgb = tf.linalg.matmul(tf.transpose(imout, [1, 2, 3, 4, 0]), mf_bar)
        im_rgb -= np.min(im_rgb, axis=(-3, -2), keepdims=True)
        im_rgb /= np.max(im_rgb, axis=(-3, -2), keepdims=True)

        fig = plt.figure(figsize=(30, 5))
        ax = gF.addAxis(fig, 1, 6)
        ax[0].imshow(im_rgb[0, 0, :, :, :])
        ax[1].imshow(im_rgb[1, 0, :, :, :])
        ax[2].imshow(im_rgb[2, 0, :, :, :])
        ax[3].imshow(im_rgb[3, 0, :, :, :])
        ax[4].imshow(np.sum(net_im, axis=0)[0], norm=TwoSlopeNorm(0), cmap="seismic")
        ax[5].imshow(rgb_im[i])
        plt.savefig(savepath + f"netim_arad{i}.png")
        plt.savefig(savepath + f"netim_arad{i}.pdf")

    return


if __name__ == "__main__":
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plot_psfs()
    im_demo(SNR=30)
    color_sim()
    plt.close()
