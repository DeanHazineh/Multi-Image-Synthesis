import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import sys

import dflat.datasets_metasurface_cells as df_library
import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.optimization_helpers as df_opt

sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import MIS_Optimizer_PSF


def create_steerable_1Derivative_pair(propagation_parameters, sigma):
    x_sens, y_sens = df_fourier.get_detector_pixel_coordinates(propagation_parameters)
    x_sens, y_sens = np.meshgrid(x_sens * 1e6, y_sens * 1e6)

    Gx = (-x_sens / sigma**2) / (2 * np.pi * sigma**2) * np.exp(-0.5 * (x_sens**2 + y_sens**2) / sigma**2)
    Gy = (-y_sens / sigma**2) / (2 * np.pi * sigma**2) * np.exp(-0.5 * (x_sens**2 + y_sens**2) / sigma**2)

    Gx = Gx[np.newaxis, np.newaxis, np.newaxis, :, :]
    Gy = Gy[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.concatenate((Gx, Gy), axis=0)


def create_quadrature_pair(propagation_parameters, sigma):
    fx = 1 / 50
    x_sens, y_sens = df_fourier.get_detector_pixel_coordinates(propagation_parameters)
    x_sens, y_sens = np.meshgrid(x_sens * 1e6, y_sens * 1e6)

    G = np.exp(-(x_sens**2 + y_sens**2) / 2 / sigma**2)
    G1 = G * np.cos(2 * np.pi * x_sens * fx)[np.newaxis, np.newaxis, np.newaxis, :, :]
    G2 = G * np.sin(2 * np.pi * x_sens * fx)[np.newaxis, np.newaxis, np.newaxis, :, :]

    return np.concatenate((G1, G2), axis=0)


def optimize_gaussian_derivative():
    savepath = str(Path(__file__).parent) + "/gaussian_derivative/"
    saveAtEpochs = 100
    lagrange_energy = 10
    bias_scale = 2e2
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1.0e-3,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 1.25e-6, "y": 1.25e-6},
            "sensor_pixel_size_m": {"x": 1.25e-6, "y": 1.25e-6},
            "sensor_pixel_number": {"x": 1500, "y": 1500},
            "radial_symmetry": False,
            "diffractionEngine": "ASM_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )

    df_struct.print_full_settings(propagation_parameters)
    point_source_locs = np.array([[0.0, 0.0, 1e6]])
    target_psf = create_steerable_1Derivative_pair(propagation_parameters, sigma=30)

    # Lookup Initialization
    focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(propagation_parameters, [532e-9, 532e-9], [0.7, 0.7], [{"x": -40e-6, "y": 0}, {"x": 40e-6, "y": 0}])
    _, init_norm_param = df_library.optical_response_to_param([focus_trans], [focus_phase], [532e-9], "Nanofins_U350nm_H600nm", reshape=True, fast=True)
    alpha_mask = np.array([[1.0, 1.0, 1.0, 1.0]])
    init_alpha = np.array([[1.0, 0.5, -1.0, -0.5], [1.0, 0.5, -1.0, -0.5]])

    ### Create the pipeline and run optimization
    pipeline = MIS_Optimizer_PSF(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_psf, lagrange_energy, bias_scale, savepath, saveAtEpochs)
    pipeline.customLoad()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    df_opt.run_pipeline_optimization(pipeline, optimizer, num_epochs=2000, loss_fn=None, allow_gpu=True)
    pipeline.save_prop_params()
    pipeline.save_optimized_lens()
    # pipeline.save_gds(size_offset=20e-9, tag="MIS_Steerable_20nmOffset_ZLCalb")

    return


def optimize_quadrature_pair():
    savepath = str(Path(__file__).parent) + "/quadrature_pair/"
    saveAtEpochs = 100
    lagrange_energy = 4
    bias_scale = 2e2
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": 1.0e-3,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_number": {"x": 600, "y": 600},
            "radial_symmetry": False,
            "diffractionEngine": "fresnel_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )

    df_struct.print_full_settings(propagation_parameters)
    point_source_locs = np.array([[0.0, 0.0, 1e6]])
    target_psf = create_quadrature_pair(propagation_parameters, sigma=50)

    # Lookup Initialization
    focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(propagation_parameters, [532e-9, 532e-9], [0.7, 0.7], [{"x": -40e-6, "y": 0}, {"x": 40e-6, "y": 0}])
    _, init_norm_param = df_library.optical_response_to_param([focus_trans], [focus_phase], [532e-9], "Nanofins_U350nm_H600nm", reshape=True, fast=True)
    alpha_mask = np.array([[1.0, 1.0, 1.0, 1.0]])
    init_alpha = np.array([[1.0, 0.5, -1.0, -0.5], [1.0, 0.5, -1.0, -0.5]])

    ### Create the pipeline and run optimization
    pipeline = MIS_Optimizer_PSF(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_psf, lagrange_energy, bias_scale, savepath, saveAtEpochs)
    pipeline.customLoad()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    df_opt.run_pipeline_optimization(pipeline, optimizer, num_epochs=1000, loss_fn=None, allow_gpu=True)

    pipeline.save_prop_params()
    pipeline.save_optimized_lens()
    # pipeline.save_gds(size_offset=0)

    return


if __name__ == "__main__":
    optimize_quadrature_pair()
    optimize_gaussian_derivative()
