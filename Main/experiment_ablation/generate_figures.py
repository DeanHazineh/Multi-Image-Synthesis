import sys
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.ndimage import gaussian_filter
from copy import deepcopy as copy
import glob

import dflat.data_structure as df_struct
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier
import dflat.plot_utilities as df_plt
import dflat.render_layer as df_im

from run_ablation_design import create_target

sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import compute_PSFs_with_interference


# Define some global settings
dat_path = str(Path(__file__).parent) + "/"
propagation_parameters = df_struct.prop_params(
    {
        "wavelength_set_m": [532e-9],
        "ms_samplesM": {"x": 285, "y": 285},
        "ms_dx_m": {"x": 350e-9, "y": 350e-9},
        "sensor_distance_m": 1.0e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 500, "y": 500},
        "radial_symmetry": False,
        "diffractionEngine": "fresnel_fourier",
    }
)
point_source_locs = np.array([[0.0, 0.0, 1e6]])
wavelength_set_m = propagation_parameters["wavelength_set_m"]
mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense512_U350_H600", pmin=0.042, pmax=0.875)
psf_layer = df_fourier.PSF_Layer(propagation_parameters)


def plot_dataset(sub_fold, mode="metasurface"):
    folds = [d for d in glob.glob(dat_path + sub_fold + "*") if os.path.isdir(d)]
    sensor_emv = df_im.BFS_PGE_51S5
    noiseless_emv = copy(sensor_emv)
    noiseless_emv["shot_noise"] = False
    noiseless_emv["dark_noise"] = False
    signal_scale = 5e5
    target_PSFs = create_target(20, propagation_parameters)

    for sim_fold in folds:
        savepath = dat_path + sub_fold + sim_fold.split("/")[-1]
        # Rather than just plot the saved PSFs, we are going to load in the optimized metalens shapes and recompute the
        # PSFs in case we want to render the PSFs under different conditions (different engine or resolution, etc)
        with open(sim_fold + "/optimized_lens.pickle", "rb") as fhandle:
            data = pickle.load(fhandle)
            latent_tensor = data["latent_tensor"]
            alpha = data["alpha"]
            alpha_mask = data["alpha_mask"]
            use_alpha = alpha * alpha_mask

        # Note that we again have to do it a little different for the ideal vs metasurface case
        if mode == "metasurface":
            out = mlp_latent_layer(latent_tensor, wavelength_set_m)
            psf_intensity, psf_phase = psf_layer(out, point_source_locs, batch_loop=False)
            psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
        else:
            phase = tf.convert_to_tensor(latent_tensor, tf.float64)
            trans = tf.ones_like(latent_tensor)
            psf_intensity, psf_phase = psf_layer([trans, phase], point_source_locs, batch_loop=False)

        # Comapare the target signal to the realized signal via the PSNR metric
        target_PSFs = df_im.photons_to_ADU(target_PSFs * signal_scale, noiseless_emv, clip_zero=False)
        psf_decomp = use_alpha[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis] * df_im.photons_to_ADU(psf_intensity * signal_scale, sensor_emv)
        net_psf = tf.math.reduce_sum(psf_decomp, axis=2)
        net_psf = net_psf / tf.norm(net_psf, axis=[-2, -1], keepdims=True) * tf.norm(target_PSFs, axis=(-2, -1), keepdims=True)

        MSE = tf.math.reduce_mean((target_PSFs - net_psf) ** 2)
        PSNR = 20 * tf.experimental.numpy.log10(tf.math.reduce_max(target_PSFs) / tf.math.sqrt(MSE))
        energy = tf.reduce_sum(psf_intensity, axis=(-2, -1)).numpy()

        fig = plt.figure()
        ax = df_plt.addAxis(fig, 1, 1)
        ax[0].imshow(net_psf[0, 0, 0], norm=TwoSlopeNorm(0), cmap="seismic")
        ax[0].set_title(str(np.round(PSNR, 3)))
        plt.savefig(savepath + "net.png")
        plt.savefig(savepath + "net.pdf")

        fig = plt.figure()
        ax = df_plt.addAxis(fig, 2, 2)
        max_val = tf.math.reduce_max(psf_decomp).numpy()
        min_val = tf.math.reduce_min(psf_decomp).numpy()
        for i in range(4):
            try:  # We need this catch statement so we dont get errors if the MIS decomposition algorithm failed
                ax[i].imshow(psf_decomp[0, 0, i, 0], norm=TwoSlopeNorm(0, vmin=min_val, vmax=max_val), cmap="seismic")
                ax[i].set_title(str(np.round(energy[0, i, 0], 3)))
            except:
                continue
        plt.savefig(savepath + "decomposition.png")
        plt.savefig(savepath + "decomposition.pdf")

        fig = plt.figure(figsize=(10, 10))
        ax = df_plt.addAxis(fig, 1, 1)
        ax[0].imshow(target_PSFs[0, 0, 0], norm=TwoSlopeNorm(0), cmap="seismic")
        plt.savefig(savepath + "target_filter.png")
        plt.savefig(savepath + "target_filter.pdf")

        plt.close()
    return


if __name__ == "__main__":
    plot_dataset("output_ideal/", mode="ideal")
    plot_dataset("output_metasurface/", mode="metasurface")
