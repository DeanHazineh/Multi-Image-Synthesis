import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pickle
from pathlib import Path
import cv2 as cv2
import sys
import os

import dflat.data_structure as df_struct
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier
import dflat.plot_utilities as gF

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from src.Multi_Image_Synthesis_Optimizers import compute_PSFs_with_interference

dirname = str(Path(__file__).parent) + "/"
savepath = dirname + "saved_Figs/"
exp_dat_path = "Experimental_meas/iter19Frameid-0-time-0.pkl"
sim_dat_path = dirname + "optimized_lens.pickle"


def rotate_alpha(alpha, angle_deg):
    return np.matmul(np.array([[np.cos(angle_deg * np.pi / 180), np.sin(angle_deg * np.pi / 180)]]), alpha)


def load_exp_PSF(alpha):
    ### Get Experiment File
    with open(dirname + exp_dat_path, "rb") as fhandle:
        data = pickle.load(fhandle)
        raw_images = data["raw_images"]
        images = data["converted_images"]
        images_split = data["converted_sub_images"][:, [3, 1, 0, 2], :, :]  # Reorder polarization idx
        dark_img = data["dark_image"]
        badPixelMask = data["badPixelIdx"]

    pol_Images = np.sum(images_split, axis=0)
    init_shape = pol_Images.shape
    pol_Images = np.transpose(cv2.resize(np.transpose(pol_Images, [1, 2, 0]), (init_shape[-1] * 2, init_shape[-2] * 2), interpolation=cv2.INTER_CUBIC), [2, 0, 1])
    _, cidxy, cidxx = np.unravel_index(pol_Images.argmax(), pol_Images.shape)

    num_pix = 300
    pol_Images = pol_Images[:, cidxy - num_pix // 2 : cidxy + num_pix // 2, cidxx - num_pix // 2 : cidxx + num_pix // 2]
    pol_Images = np.flip(np.rot90(pol_Images, k=-1, axes=(-2, -1)), axis=-2)  # I mount lens flipped over when imaging
    net_psf_exp = np.sum(pol_Images[np.newaxis, :, :, :] * alpha[:, :, np.newaxis, np.newaxis], axis=1, keepdims=True)

    extent_exp = np.array([-num_pix // 2, num_pix // 2, -num_pix // 2, num_pix // 2]) * 3.5
    return pol_Images, extent_exp, net_psf_exp


def load_sim_PSF():
    ### Get Simulation
    with open(sim_dat_path, "rb") as fhandle:
        data = pickle.load(fhandle)
        latent_tensor = data["latent_tensor"]
        alpha = np.stack([data["alpha1"], data["alpha2"]])
        alpha = np.concatenate((alpha, np.array([[0], [0]])), axis=1)
        alpha3 = rotate_alpha(alpha, 45)
        alpha4 = rotate_alpha(alpha, 135)
        alpha = np.concatenate((alpha, alpha3, alpha4), axis=0)

    sim_settings = {
        "wavelength_set_m": [532e-9],
        "ms_samplesM": {"x": 571, "y": 571},
        "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
        "radius_m": 1.0e-3,
        "sensor_distance_m": 45e-3,
        "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_number": {"x": 300, "y": 300},
        "radial_symmetry": False,
        "diffractionEngine": "fresnel_fourier",
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    propagation_parameters = df_struct.prop_params(sim_settings)
    mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense512_U350_H600", pmin=0.042, pmax=0.875)
    psf_layer = df_fourier.PSF_Layer(propagation_parameters)

    trans, phase = mlp_latent_layer(latent_tensor, [532e-9])
    psf_intensity, psf_phase = psf_layer([trans, phase], [[0.0, 0.0, 1e6]], batch_loop=False)
    psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
    print(psf_intensity.shape, alpha.shape)

    net_psf = tf.math.reduce_sum(psf_intensity * alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis], axis=1)

    lx, ly = gF.get_detector_pixel_coordinates(propagation_parameters)
    sim_extent = np.array([min(lx), max(lx), min(ly), max(ly)]) * 1e6

    return psf_intensity, sim_extent, net_psf, alpha


def compare_exp_sim_psf():
    psf_intensity, sim_extent, net_psf, alpha = load_sim_PSF()
    pol_Images, extent_exp, net_psf_exp = load_exp_PSF(alpha)

    #### Figures
    fig = plt.figure(figsize=(20, 10))
    ax = gF.addAxis(fig, 2, 4)
    for i in range(4):
        ax[i].imshow(pol_Images[i], extent=extent_exp)
        ax[i + 4].imshow(psf_intensity[0, i, 0], extent=sim_extent)
    for axis in ax:
        axis.set_xlim([-400, 400])
        axis.set_ylim([-400, 400])
    plt.savefig(savepath + "psfs.png")
    plt.savefig(savepath + "psfs.pdf")

    fig = plt.figure(figsize=(20, 10))
    ax = gF.addAxis(fig, 2, 4)
    for i in range(4):
        ax[i].imshow(net_psf_exp[i, 0], norm=TwoSlopeNorm(0), cmap="seismic")
        ax[i + 4].imshow(net_psf[i, 0], norm=TwoSlopeNorm(0), cmap="seismic")
    plt.savefig(savepath + "net_psfs.png")
    plt.savefig(savepath + "net_psfs.pdf")
    return


if __name__ == "__main__":
    compare_exp_sim_psf()
    plt.close()
