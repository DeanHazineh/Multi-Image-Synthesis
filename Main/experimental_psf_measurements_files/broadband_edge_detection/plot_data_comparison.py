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
exp_dat_path = "Experimental_meas/iter39Frameid-0-time-0.pkl"
sim_dat_path = dirname + "optimized_lens.pickle"


def load_exp_PSF():
    ### Get Experiment File
    with open(dirname + exp_dat_path, "rb") as fhandle:
        data = pickle.load(fhandle)
        raw_images = data["raw_images"]
        images = data["converted_images"]
        images_split = data["converted_sub_images"][:, [3, 1, 0, 2], :, :]  # Reorder polarization idx
        dark_img = data["dark_image"]
        badPixelMask = data["badPixelIdx"]

    # use the simulation alpha
    with open(sim_dat_path, "rb") as fhandle:
        data = pickle.load(fhandle)
        alpha = data["alpha"].numpy()
        alpha = np.array([alpha[0], 0.0, alpha[1], 0.0])[np.newaxis, :]

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
        alpha = data["alpha"].numpy()
        alpha = np.array([alpha[0], 0.0, alpha[1], 0.0])[np.newaxis, :]

    sim_settings = {
        "wavelength_set_m": [532e-9],
        "ms_samplesM": {"x": 571, "y": 571},
        "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
        "radius_m": 1.0e-3,
        "sensor_distance_m": 45e-3,
        "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
        "sensor_pixel_number": {"x": 300, "y": 300},
        "radial_symmetry": True,
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
    net_psf = tf.math.reduce_sum(psf_intensity * alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis], axis=1)

    lx, ly = gF.get_detector_pixel_coordinates(propagation_parameters)
    sim_extent = np.array([min(lx), max(lx), min(ly), max(ly)]) * 1e6

    return psf_intensity, sim_extent, net_psf


def compare_exp_sim_psf():
    pol_Images, extent_exp, net_psf_exp = load_exp_PSF()
    psf_intensity, sim_extent, net_psf = load_sim_PSF()

    #### Figures
    fig = plt.figure(figsize=(40, 20))
    ax = gF.addAxis(fig, 2, 4)
    for i in range(4):
        ax[i].imshow(pol_Images[i], extent=extent_exp)
        ax[i + 4].imshow(psf_intensity[0, i, 0], extent=sim_extent)
    for axis in ax:
        axis.set_xlim([-450, 450])
        axis.set_ylim([-450, 450])
    plt.savefig(savepath + "psfs.png")
    plt.savefig(savepath + "psfs.pdf")

    fig = plt.figure(figsize=(10, 20))
    ax = gF.addAxis(fig, 2, 1)
    ax[0].imshow(net_psf_exp[0, 0], norm=TwoSlopeNorm(0), cmap="seismic")
    ax[1].imshow(net_psf[0, 0], norm=TwoSlopeNorm(0), cmap="seismic")
    plt.savefig(savepath + "net_psfs.png")
    plt.savefig(savepath + "net_psfs.pdf")

    return


if __name__ == "__main__":
    compare_exp_sim_psf()
    plt.close()
    