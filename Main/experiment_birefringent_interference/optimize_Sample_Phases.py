import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path

import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.plot_utilities as df_plt

dirname = str(Path(__file__).parent) + "/"


def run_optimization(init_phase, savepath, field_prop, target_intensity, loss_fun="L1", saveAtStep=100):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    phase_tensor = tf.Variable(init_phase, dtype=tf.float64, trainable=True)
    trans_tensor = tf.Variable(np.ones_like(init_phase), dtype=tf.float64, trainable=False)
    target_intensity = target_intensity / tf.reduce_sum(target_intensity)  # Normalize to unit area
    aperture = field_prop.aperture_trans
    norm_by = np.sum(aperture)

    # GD optimization
    iter = 1000
    if loss_fun == "L1":
        loss_fn = lambda psf, target: tf.math.reduce_sum(tf.math.abs(psf - target))
    else:
        loss_fn = lambda psf, target: -tf.math.reduce_sum(psf * target / tf.norm(target) / tf.norm(psf)) - 0.2 * tf.math.reduce_sum(psf)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=25, decay_rate=0.8)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    loss_history = []
    for i in tqdm(range(iter + 1)):
        with tf.GradientTape() as tape:
            out_trans, out_phase = field_prop((trans_tensor, phase_tensor))
            psf = out_trans**2 / norm_by
            loss = loss_fn(tf.squeeze(psf), target_intensity)

        gradients = tape.gradient(loss, [phase_tensor])
        optimizer.apply_gradients(zip(gradients, [phase_tensor]))
        loss_history.append(loss)

        if np.mod(i, saveAtStep) == 0:
            fig = plt.figure(figsize=(30, 5))
            ax = df_plt.addAxis(fig, 1, 6)
            ax[0].plot(loss_history)
            ax[1].imshow(np.squeeze(trans_tensor * aperture))
            ax[2].imshow(np.squeeze(np.angle(np.exp(1j * phase_tensor.numpy()))), cmap="hsv")
            ax[3].imshow(np.squeeze(out_trans))
            ax[3].set_title(np.sum(out_trans**2) / norm_by)
            ax[4].imshow(target_intensity)
            ax[4].set_title(np.sum(target_intensity.numpy()))
            ax[5].imshow(np.squeeze(out_phase), cmap="hsv")
            plt.savefig(savepath + "epoch" + str(i) + "chkpoint.png")
            plt.close()

    # at the end save a pdf so we can use it later
    fig = plt.figure(figsize=(20, 10))
    ax = df_plt.addAxis(fig, 1, 2)
    ax[0].imshow(np.squeeze(out_trans))
    ax[0].set_title(np.sum(out_trans**2) / norm_by)
    ax[1].imshow(np.squeeze(out_phase), cmap="hsv")
    plt.savefig(savepath + "opt_field.pdf")

    return


def run_SGD_Inverse_Sampler():
    prop_params = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 1000, "y": 1000},
            "ms_dx_m": {"x": 1e-6, "y": 1e-6},
            "radius_m": 1000 * 1e-6 / 2,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_number": {"x": 500, "y": 500},
            "radial_symmetry": False,
            "diffractionEngine": "ASM_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    df_struct.print_full_settings(prop_params)
    field_prop = df_fourier.Propagate_Planes_Layer(prop_params)

    # Define the target intensity disk
    sensor_pixel_number = prop_params["sensor_pixel_number"]
    rdisk = 250e-6
    xd, yd = np.meshgrid(*df_plt.get_detector_pixel_coordinates(prop_params))
    target = np.zeros((sensor_pixel_number["y"], sensor_pixel_number["x"]))
    target[np.where(np.sqrt(xd**2 + yd**2) <= rdisk)] = 1.0
    target_intensity = tf.convert_to_tensor(target, dtype=tf.float64)

    # Define focusing lens as starting profile
    lens = df_fourier.focus_lens_init(prop_params, [532e-9], [30e-3], [{"x": 0, "y": 0}])
    init_phase = lens[1]
    savepath = dirname + "output_SGD_Phase/focus_init"
    run_optimization(init_phase, savepath + "_L1/", field_prop, target_intensity, loss_fun="L1")
    run_optimization(init_phase, savepath + "_L2/", field_prop, target_intensity, loss_fun="L2")

    # Run some random starts with random phase
    repeat_num = 5
    for j in range(repeat_num):
        np.random.seed(j)
        init_phase = np.random.rand(*init_phase.shape) * 2 * np.pi
        savepath = dirname + "output_SGD_Phase/random_phase"
        run_optimization(init_phase, savepath + f"_L1_{j}/", field_prop, target_intensity, loss_fun="L1")
        run_optimization(init_phase, savepath + f"_L2_{j}/", field_prop, target_intensity, loss_fun="L2")

    return


if __name__ == "__main__":
    run_SGD_Inverse_Sampler()
