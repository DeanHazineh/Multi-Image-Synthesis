import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
import os

import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.datasets_metasurface_cells as df_library
import dflat.datasets_image as df_images
import dflat.optimization_helpers as df_opt

sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import MIS_Optimizer_Images


def create_LoG(sigma1, sigma2, propagation_parameters):
    x_sens, y_sens = df_fourier.get_detector_pixel_coordinates(propagation_parameters)
    x_sens, y_sens = np.meshgrid(x_sens * 1e6, y_sens * 1e6)

    G1 = 1 / (2 * np.pi * sigma1**2) * np.exp(-0.5 * (x_sens**2 + y_sens**2) / sigma1**2)
    G2 = 1 / (2 * np.pi * sigma2**2) * np.exp(-0.5 * (x_sens**2 + y_sens**2) / sigma2**2)
    filt = G1 - G2

    return filt[tf.newaxis, :, :]


def optimize(SNR=30):
    savepath = str(Path(__file__).parent) + "/output/"
    lagrange_energy = 7
    bias_scale = 2e2

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
            "sensor_pixel_number": {"x": 700, "y": 700},
            "radial_symmetry": True,
            "diffractionEngine": "fresnel_fourier",
        }
    )
    point_source_locs = np.array([[0.0, 0.0, z] for z in [1e6]])

    # Target Image stack
    target_filt = create_LoG(20, 25, propagation_parameters)
    test_im = df_images.get_grayscale_image("cameraman.png", {"x": 1800, "y": 1800})  # Ny x Nx x 1
    test_im = tf.convert_to_tensor(np.transpose(test_im, [2, 0, 1]), dtype=tf.float64)
    alpha_mask = np.array([1.0, 1.0, 1.0, 1.0])[np.newaxis, :]
    init_alpha = np.array([[1.0, -0.5, -1.0, 0.5]])

    # Lookup Initialization
    focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(propagation_parameters, [500e-9, 500e-9], [1.0, 0.6], [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
    _, init_norm_param = df_library.optical_response_to_param([focus_trans], [focus_phase], [500e-9], "Nanofins_U350nm_H600nm", reshape=True, fast=True)

    # Create optimizer
    pipeline = MIS_Optimizer_Images(
        alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, test_im, target_filt, SNR, lagrange_energy, bias_scale, savepath, saveAtEpochs=100
    )
    pipeline.customLoad()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    df_opt.run_pipeline_optimization(pipeline, optimizer, num_epochs=1000, loss_fn=None, allow_gpu=True)
    pipeline.save_optimized_lens()
    pipeline.save_prop_params()
    # pipeline.save_gds(size_offset=0)

    return


if __name__ == "__main__":
    optimize(SNR=30)
