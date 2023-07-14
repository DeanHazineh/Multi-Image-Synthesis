import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import sys

import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.datasets_metasurface_cells as df_library
import dflat.optimization_helpers as df_opt

sys.path.append(os.path.dirname(sys.path[0]))
from src.Multi_Image_Synthesis_Optimizers import MIS_Optimizer_PSF


def create_steerable_1Derivatives(propagation_parameters, sigma_um_list, rotation_deg_list):
    x_sens, y_sens = df_fourier.get_detector_pixel_coordinates(propagation_parameters)
    x_sens, y_sens = np.meshgrid(x_sens * 1e6, y_sens * 1e6)

    G_list = []
    for i in range(len(sigma_um_list)):
        Gx = (-x_sens / sigma_um_list[i] ** 2) / (2 * np.pi * sigma_um_list[i] ** 2) * np.exp(-0.5 * (x_sens**2 + y_sens**2) / sigma_um_list[i] ** 2)
        Gy = (-y_sens / sigma_um_list[i] ** 2) / (2 * np.pi * sigma_um_list[i] ** 2) * np.exp(-0.5 * (y_sens**2 + x_sens**2) / sigma_um_list[i] ** 2)
        G = Gx * np.cos(rotation_deg_list[i] * np.pi / 180) + Gy * np.sin(rotation_deg_list[i] * np.pi / 180)
        G_list.append(G)

    G_list = np.stack(G_list)
    return G_list[np.newaxis, np.newaxis, :, :, :]


def optimize(alpha_mask, tag):
    savepath = str(Path(__file__).parent) + "/" + tag + ""
    lagrange_energy = 2
    bias_scale = 1e2
    propagation_parameters = df_struct.prop_params(
        {
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
    )

    df_struct.print_full_settings(propagation_parameters)
    zlocs = [33e-3, 35e-3, 37e-3, 39e-3, 41e-3]
    point_source_locs = np.array([[0.0, 0.0, z] for z in zlocs])
    target_PSFs = create_steerable_1Derivatives(propagation_parameters, [30, 30, 30, 30, 30], [0, 45 / 2, 45, 67.5, 90])

    # Lookup Initialization
    focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(propagation_parameters, [532e-9, 532e-9], [31e-3, 31e-3], [{"x": -50e-6, "y": 0}, {"x": 50e-6, "y": 0}])
    _, init_norm_param = df_library.optical_response_to_param([focus_trans], [focus_phase], [532e-9], "Nanofins_U350nm_H600nm", reshape=True, fast=True)
    alpha_mask = np.array(alpha_mask)[np.newaxis, :]
    init_alpha = np.array([[1.0, 0.5, -1.0, -0.5]])

    # Create the pipeline, load checkpoint, and visualize test image, run optimization
    pipeline = MIS_Optimizer_PSF(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_PSFs, lagrange_energy, bias_scale, savepath, saveAtEpochs=250)
    pipeline.customLoad()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    df_opt.run_pipeline_optimization(pipeline, optimizer, num_epochs=3000, loss_fn=None, allow_gpu=True)
    pipeline.save_optimized_lens()
    pipeline.save_prop_params()
    # pipeline.save_gds(size_offset=0)

    return


if __name__ == "__main__":
    optimize([1.0, 0.0, 1.0, 0.0], "output_Two_image/")
    #optimize([1.0, 1.0, 1.0, 1.0], "output_Four_image/")
