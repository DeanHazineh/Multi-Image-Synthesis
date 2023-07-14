import os
import sys
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

import dflat.data_structure as df_struct
import dflat.optimization_helpers as df_opt

sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import MIS_Optimizer_PSF, MIS_Optimizer_PSF_IDEAL


def create_target(sigma, propagation_parameters):
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]
    cidxy, cidxx = sensor_pixel_number["y"] // 2, sensor_pixel_number["x"] // 2
    pulse = np.zeros((sensor_pixel_number["y"], sensor_pixel_number["x"]))
    pulse[cidxy, cidxx] = 1.0

    Gs = gaussian_filter(pulse, sigma, order=(2, 2))
    return Gs[np.newaxis, np.newaxis, np.newaxis, :, :]


def optimize(savepath, lagrange_energy, bias_scale, alpha_mask, epochs, mode="metasurface"):
    # If optic is metasurface, let us optimize a real nanofin metasurface with the MLP
    # If optic is ideal, let us optimize four decoupled phase profiles
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Set up the propagation problem
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
    df_struct.print_full_settings(propagation_parameters)

    # Define the field intensity target
    target_PSFs = create_target(20, propagation_parameters)

    # Set up the optimizer
    ms_cells = propagation_parameters["ms_samplesM"]
    saveAtEpochs = 100
    if mode == "metasurface":  # Optimize over a real metasurface with the interference constraint (nanofins)
        init_norm_param = np.random.random((2, ms_cells["y"], ms_cells["x"]))
        init_alpha = np.array([[1.0, 0.5, -1.0, -0.5]])
        pipeline = MIS_Optimizer_PSF(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_PSFs, lagrange_energy, bias_scale, savepath, saveAtEpochs)
    else:  # Optimize over fully decoupled phase profiles (four)
        init_alpha = np.array([[1.0, 1.0, -1.0, -1.0]])
        init_norm_param = np.random.random([4, ms_cells["y"], ms_cells["x"]])
        pipeline = MIS_Optimizer_PSF_IDEAL(
            alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_PSFs, lagrange_energy, bias_scale, savepath, saveAtEpochs
        )

    # Run the training
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    df_opt.run_pipeline_optimization(pipeline, optimizer, num_epochs=epochs, loss_fn=None, allow_gpu=True)
    pipeline.save_optimized_lens()
    pipeline.save_prop_params()

    return


if __name__ == "__main__":
    savepath = str(Path(__file__).parent)
    alpha_mask = np.array([1.0, 1.0, 1.0, 1.0])[np.newaxis, :]

    # Sweep the abaltion study with different values of the regularizers (including 0)
    # Also compare the case of the true metalens system vs the idealized four image multi-synthesis
    for Energy_coeff in [0, 5]:
        for bias_coeff in [0, 5e3]:
            optimize(savepath + f"/output_metasurface/Regularer_E{Energy_coeff}_B{bias_coeff}/", Energy_coeff, bias_coeff, alpha_mask, epochs=1000, mode="metasurface")
            optimize(savepath + f"/output_ideal/Regularer_E{Energy_coeff}_B{bias_coeff}/", Energy_coeff, bias_coeff, alpha_mask, epochs=1000, mode="ideal")
