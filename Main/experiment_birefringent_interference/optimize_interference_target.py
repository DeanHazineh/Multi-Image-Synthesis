import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
from pathlib import Path

from PIL import Image
import cv2 as cv

import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.plot_utilities as df_plt
from dflat.datasets_image import load_grayscale_fromPath

dirname = str(Path(__file__).parent) + "/"


def run_optimization(init_phase, savepath, field_prop, target_intensity, target_interference, saveAtStep=100, iter=2000):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    phase_tensor = tf.Variable(init_phase, dtype=tf.float64, trainable=True)
    trans_tensor = tf.Variable(np.ones_like(init_phase), dtype=tf.float64, trainable=False)
    target_interference = tf.convert_to_tensor(target_interference, dtype=tf.float64)
    target_intensity = np.stack([target_intensity, target_intensity], axis=0)
    target_intensity = tf.expand_dims(target_intensity / tf.reduce_sum(target_intensity, axis=(1, 2), keepdims=True), axis=0)
    aperture = field_prop.aperture_trans
    norm_by = np.sum(aperture)

    # GD optimization
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    loss_history = []
    for i in tqdm(range(iter + 1)):
        with tf.GradientTape() as tape:
            out_trans, out_phase = field_prop((trans_tensor, phase_tensor))
            out_trans = out_trans**2 / norm_by
            interf1 = 0.5 * out_trans[:, 0:1, :, :] + 0.5 * out_trans[:, 1:2, :, :]
            interf2 = tf.math.sqrt(out_trans[:, 0:1, :, :] * out_trans[:, 1:2, :, :]) * tf.math.cos(out_phase[:, 0:1, :, :] - out_phase[:, 1:2, :, :])
            int45 = interf1 + interf2

            loss = tf.math.reduce_sum(tf.math.abs(out_trans[0, :, :, :] - target_intensity)) + tf.math.reduce_sum(
                tf.math.abs(int45 / tf.math.reduce_sum(int45) - target_interference / tf.math.reduce_sum(target_interference))
            )

        gradients = tape.gradient(loss, [phase_tensor])
        optimizer.apply_gradients(zip(gradients, [phase_tensor]))
        loss_history.append(loss)
        psf_intensity = tf.concat([out_trans[:, 0:1, :, :], out_trans[:, 1:2, :, :], interf1 + interf2, interf1 - interf2], axis=1)

        if np.mod(i, saveAtStep) == 0:
            fig = plt.figure(figsize=(10, 15))
            ax = df_plt.addAxis(fig, 3, 2)
            ax[0].imshow(psf_intensity[0, 0, :, :])
            ax[0].set_title(np.sum(psf_intensity[0, 0]))
            ax[1].imshow(psf_intensity[0, 1, :, :])
            ax[1].set_title(np.sum(psf_intensity[0, 1]))
            ax[2].imshow(np.squeeze(np.angle(np.exp(1j * out_phase[0, 0].numpy()))), cmap="hsv")
            ax[3].imshow(np.squeeze(np.angle(np.exp(1j * out_phase[0, 1].numpy()))), cmap="hsv")

            ax[4].imshow(psf_intensity[0, 2, :, :])
            ax[5].imshow(psf_intensity[0, 3, :, :])
            plt.savefig(savepath + "epoch" + str(i) + "chkpoint.png")
            plt.close()

            fig = plt.figure(figsize=(5, 5))
            ax = df_plt.addAxis(fig, 1, 1)
            ax[0].plot(loss_history)
            plt.savefig(savepath + "loss_history.png")
            plt.close()

    # at the end save a pdf so we can use it later
    lens_phase = np.angle(np.exp(1j * phase_tensor.numpy()))
    out_phase = np.angle(np.exp(1j * out_phase[0].numpy()))

    fig = plt.figure(figsize=(5, 10))
    ax = df_plt.addAxis(fig, 2, 1)
    ax[0].imshow(lens_phase[0], cmap="hsv")
    ax[1].imshow(lens_phase[1], cmap="hsv")
    plt.savefig(savepath + "lens_phase.png")
    plt.savefig(savepath + "lens_phase.pdf")

    fig = plt.figure(figsize=(20, 5))
    ax = df_plt.addAxis(fig, 1, 4)
    for i in range(4):
        ax[i].imshow(psf_intensity[0, i])
    plt.savefig(savepath + "out_int.png")
    plt.savefig(savepath + "out_int.pdf")

    fig = plt.figure(figsize=(10, 5))
    ax = df_plt.addAxis(fig, 1, 2)
    ax[0].imshow(out_phase[0], cmap="hsv")
    ax[1].imshow(out_phase[1], cmap="hsv")
    plt.savefig(savepath + "out_phase.png")
    plt.savefig(savepath + "out_phase.pdf")

    # at end, save the output field so we can use it to plot
    data = {"target_intensity": target_intensity, "target_interference": target_interference, "lens_phase": lens_phase, "psf_intensity": psf_intensity, "out_phase": out_phase}
    with open(savepath + "data.pickle", "wb") as fhandle:
        pickle.dump(data, fhandle)

    return


def optimize_interference():
    # Define simulation settings
    prop_params = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 1500, "y": 1500},
            "ms_dx_m": {"x": 1e-6, "y": 1e-6},
            "radius_m": 1500 * 1e-6 / 2,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_number": {"x": 1500, "y": 1500},
            "radial_symmetry": False,
            "diffractionEngine": "ASM_fourier",
            "automatic_upsample": False,
        }
    )
    df_struct.print_full_settings(prop_params)
    field_prop = df_fourier.Propagate_Planes_Layer(prop_params)

    # Define the target intensity disk
    rdisk = 1500e-6 / 2
    xd, yd = np.meshgrid(*df_plt.get_detector_pixel_coordinates(prop_params))
    sensor_pixel_number = prop_params["sensor_pixel_number"]
    target = np.zeros((sensor_pixel_number["y"], sensor_pixel_number["x"]))
    target[np.where(np.sqrt(xd**2 + yd**2) <= rdisk)] = 1.0
    target_intensity = tf.convert_to_tensor(target, dtype=tf.float64)

    # Define interference target
    img_path = dirname + "Harvard_University_shield.png"
    out = load_grayscale_fromPath(img_path, sensor_pixel_number, resize_method="pad", shrink_img_scale=0.70, invert=False)
    interf_target = np.squeeze(out / np.max(out))

    # Define random starting phase profiles
    ms_samplesM = prop_params["ms_samplesM"]
    init_phase = np.random.rand(2, ms_samplesM["y"], ms_samplesM["x"]) * 2 * np.pi
    savepath = dirname + "output_interf/"
    run_optimization(init_phase, savepath, field_prop, target_intensity, interf_target, saveAtStep=200, iter=3000)

    return


if __name__ == "__main__":
    optimize_interference()
