import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import dflat.optimization_helpers as df_opt
import dflat.neural_optical_layer as df_neural
import dflat.fourier_layer as df_fourier
import dflat.tools as df_tools
import dflat.GDSII_utilities as df_gds
import dflat.plot_utilities as gF
import dflat.render_layer as df_im


def compute_PSFs_with_interference(psf_intensity, psf_phase):
    interf1 = 0.5 * psf_intensity[:, 0:1, :, :, :] + 0.5 * psf_intensity[:, 1:2, :, :, :]
    interf2 = tf.math.sqrt(psf_intensity[:, 0:1, :, :, :] * psf_intensity[:, 1:2, :, :, :]) * tf.math.cos(psf_phase[:, 0:1, :, :, :] - psf_phase[:, 1:2, :, :, :])
    psf_intensity = tf.concat([psf_intensity[:, 0:1, :, :, :], interf1 + interf2, psf_intensity[:, 1:2, :, :, :], interf1 - interf2], axis=1)
    return psf_intensity


class MIS_Optimizer_Base(df_opt.Pipeline_Object):
    def __init__(self, alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, energy_scale, bias_scale, savepath, saveAtEpochs):
        super().__init__(savepath, saveAtEpochs)

        # Initialize computational tensors used later
        self.energy_max = tf.convert_to_tensor(np.array([1.0, 2.0, 1.0, 2.0]), dtype=tf.float64)[tf.newaxis, :, tf.newaxis]
        self.alpha_mask = tf.convert_to_tensor(alpha_mask, dtype=tf.float64)
        self.energy_scale = tf.convert_to_tensor(energy_scale, dtype=tf.float64)
        self.bias_scale = tf.convert_to_tensor(bias_scale, tf.float64)
        self.hold_bias = []
        self.tf_one = tf.constant(1.0, dtype=tf.float64)

        # Create computational pipeline elements
        self.propagation_parameters = propagation_parameters
        self.point_source_locs = point_source_locs
        self.mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense512_U350_H600", pmin=0.042, pmax=0.875)
        self.psf_layer = df_fourier.PSF_Layer(propagation_parameters)

        # Define trainable metasurface tensor and synthesis weights
        init_latent_tensor = tf.clip_by_value(df_tools.param_to_latent(init_norm_param), -2.5, 2.5)
        self.latent_tensor_variable = tf.Variable(init_latent_tensor, trainable=True, dtype=tf.float64, name="metasurface_latent_tensor")
        self.alpha = tf.Variable(init_alpha, trainable=True, dtype=tf.float64, name="TPS_scalar")

        # Hold the individual loss components for debugging
        self.psf_loss_vector = []
        self.energy_loss_vector = []
        self.bias_loss_vector = []

    def save_optimized_lens(self):
        ### Get the optimized optic to package and save
        wavelength_set_m = self.propagation_parameters["wavelength_set_m"]
        trans, phase = self.mlp_latent_layer(self.latent_tensor_variable, wavelength_set_m)
        shape_vector = self.mlp_latent_layer.latent_to_unnorm_shape(self.latent_tensor_variable)
        aperture = self.psf_layer.aperture_trans

        data = {
            "latent_tensor": self.latent_tensor_variable,
            "alpha": self.alpha,
            "alpha_mask": self.alpha_mask,
            "transmittance": trans,
            "phase": phase,
            "aperture": aperture,
            "shape_vector": shape_vector,
        }
        data_path = self.savepath + "optimized_lens.pickle"
        with open(data_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return shape_vector.numpy(), aperture.numpy()

    def save_prop_params(self):
        with open(self.savepath + "propagation_parameters.pickle", "wb") as handle:
            pickle.dump(self.propagation_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def save_gds(self, size_offset, tag=None):
        shape_array, aperture = self.save_optimized_lens()
        shape_array -= size_offset  # Subtract size offset based on fabrication calibrations

        # for radial optimization, convert to 2D
        if self.propagation_parameters["radial_symmetry"]:
            shape_array = np.squeeze(df_fourier.radial_2d_transform(shape_array), 1)
            aperture = np.squeeze(df_fourier.radial_2d_transform(aperture), 1)

        # Note creating the GDS file for large metasurfacs may take some time
        # We should define the ms grid to an integer factor of meta-atom cells (ideally unity factor)
        ms_dx_m = self.propagation_parameters["ms_dx_m"]
        cell_size = 350e-9
        upsample_x, upsample_y = ms_dx_m["x"] / cell_size, ms_dx_m["y"] / cell_size

        if not np.isclose(upsample_x, upsample_y, 1e-5):
            raise ValueError("Use the same discretization along x and y for this code!")

        if not np.isclose(np.mod(ms_dx_m["x"], cell_size), 0.0, 1e-5):
            print(np.mod(ms_dx_m["x"], cell_size), ms_dx_m["x"], cell_size)
            raise ValueError("Make metasurface grid an integer multiple of the meta-atom pitch!")

        shape_array, aperture = df_gds.upsample_with_cv2([shape_array, aperture], upsample_factor=upsample_x)
        boolean_mask = np.isclose(aperture, np.ones_like(aperture), 1e-3)
        rotation = np.zeros((1, shape_array.shape[-2], shape_array.shape[-1]))  # No rotations optimized here
        df_gds.assemble_nanofin_gdsII(shape_array, rotation, cell_size, self.savepath + "nanofins_gds", boolean_mask, gds_unit=1e-6, gds_precision=1e-9, tag=tag)

        return

    def compute_regularization_loss(self, psf_intensity, use_alpha):

        psf_shape = psf_intensity.shape
        H = tf.transpose(tf.reshape(psf_intensity, [psf_shape[0], psf_shape[1], psf_shape[2], -1]), [0, 2, 1, 3])
        R = tf.linalg.matmul(H, tf.transpose(H, [0, 1, 3, 2]))
        mask = tf.clip_by_value(-1.0 * tf.linalg.matmul(tf.transpose(use_alpha), use_alpha), 0, tf.float64.max)[tf.newaxis, tf.newaxis, :, :]
        bias_loss = self.bias_scale**2 * tf.math.reduce_sum(tf.math.abs(mask * R))

        energy_loss = self.energy_scale**2 * tf.math.reduce_sum((self.energy_max - tf.math.reduce_sum(psf_intensity, axis=[-2, -1])))

        # Move computed variables to attributes so we can plot during checkpoints
        self.energy_loss_vector.append(energy_loss.numpy())
        self.bias_loss_vector.append(bias_loss.numpy())
        self.psf_intensity = psf_intensity.numpy()

        return energy_loss + bias_loss

    def visualizeTrainingCheckpoint(self, epoch_str, saveaspdf=False):
        # Designed this code to work for either the broadband study or depth study where one of the two dimensions are singular
        # Transpose so polarization batch is the first dimension so we can reuse the wavelength or depth simulation code
        pix_num = self.propagation_parameters["sensor_pixel_number"]
        pixy, pixx = pix_num["y"], pix_num["x"]
        use_alpha = self.alpha * self.alpha_mask
        use_alpha = use_alpha.numpy()

        psf_intensity = np.reshape(np.transpose(self.psf_intensity, [1, 0, 2, 3, 4]), [4, -1, pixy, pixx])
        num_im = self.net_psf.shape[0]
        net_psf = np.reshape(self.net_psf, [num_im, -1, pixy, pixx])
        num_batch = net_psf.shape[1]

        ### Plot the set of net psfs and targets
        fig = plt.figure(figsize=(num_batch * 10, num_im * 10))
        ax = gF.addAxis(fig, num_im, num_batch)
        iter = 0
        for r in range(num_im):
            for c in range(num_batch):
                ax[iter].imshow(net_psf[r, c, :, :], norm=TwoSlopeNorm(0), cmap="seismic")
                ax[iter].set_title(np.array2string(np.round(use_alpha[r, :], 3)))
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "NetPSF_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "NetPSF_epoch" + epoch_str + ".png")
        plt.close()

        # Plot the component PSFs
        fig = plt.figure(figsize=(10 * num_batch, 40))
        ax = gF.addAxis(fig, 4, num_batch)
        iter = 0
        for r in range(4):
            for c in range(num_batch):
                im = ax[iter].imshow(psf_intensity[r, c, :, :])
                ax[iter].set_title(f"Energy: {np.sum(psf_intensity[r,c]):2.2f}")
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "PSF_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "PSF_epoch" + epoch_str + ".png")
        plt.close()

        # Plot the component PSFs with decomposition scaling
        decomp = psf_intensity * use_alpha.T[:, :, np.newaxis, np.newaxis]
        min_val = np.minimum(-1e-6, np.min(decomp))
        max_val = np.maximum(1e-6, np.max(decomp))
        fig = plt.figure(figsize=(10 * num_im, 40))
        ax = gF.addAxis(fig, 4, num_im)
        iter = 0
        for r in range(4):
            for c in range(num_im):
                im = ax[iter].imshow(decomp[r, c, :, :], norm=TwoSlopeNorm(0, min_val, max_val), cmap="seismic")
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "PSFDecomposition_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "PSFDecomposition_epoch" + epoch_str + ".png")
        plt.close()

        # Plot the loss
        fig = plt.figure(figsize=(15, 15))
        ax = gF.addAxis(fig, 1, 1)
        total_loss = np.array(self.psf_loss_vector) + np.array(self.energy_loss_vector) + np.array(self.bias_loss_vector)
        ax[0].plot(total_loss, "k-", label="total loss")
        ax[0].plot(self.psf_loss_vector, "b--", label="psf loss")
        ax[0].plot(self.bias_loss_vector, "r--", label="bias loss")
        ax[0].plot(self.energy_loss_vector, "g--", label="energy loss")
        gF.formatPlots(fig, ax[0], None, addLegend=True)
        plt.savefig(self.savepath + "/pdf_images/" + "epoch" + epoch_str + "CheckpointVisualize" + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "epoch" + epoch_str + "CheckpointVisualize" + ".png")
        plt.close()

        print("Saving Lens")
        self.save_optimized_lens()

        return


class MIS_Optimizer_PSF(MIS_Optimizer_Base):
    def __init__(self, alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_psf, energy_scale, bias_scale, savepath, saveAtEpochs):
        super().__init__(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, energy_scale, bias_scale, savepath, saveAtEpochs)
        self.target_psf = tf.convert_to_tensor(target_psf, dtype=tf.float64)

    def __call__(self):
        wavelength_set_m = self.propagation_parameters["wavelength_set_m"]

        # Compute the psfs
        out = self.mlp_latent_layer(self.latent_tensor_variable, wavelength_set_m)
        psf_intensity, psf_phase = self.psf_layer(out, self.point_source_locs, batch_loop=False)
        psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)
        use_alpha = self.alpha * self.alpha_mask
        net_psf = tf.math.reduce_sum(use_alpha[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis] * psf_intensity, axis=2)

        psf_loss = tf.math.reduce_sum(tf.math.abs(net_psf / tf.norm(net_psf, axis=[-2, -1], keepdims=True) - self.target_psf / tf.norm(self.target_psf, axis=[-2, -1], keepdims=True)))
        self.psf_loss_vector.append(psf_loss.numpy())
        self.net_psf = net_psf.numpy()

        # Compute loss
        return psf_loss + self.compute_regularization_loss(psf_intensity, use_alpha)

    def visualizeTrainingCheckpoint(self, epoch_str, saveaspdf=False):
        # Designed this code to work for either the broadband study or depth study where one of the two dimensions are singular
        # Transpose so polarization batch is the first dimension so we can reuse the wavelength or depth simulation code
        pix_num = self.propagation_parameters["sensor_pixel_number"]
        pixy, pixx = pix_num["y"], pix_num["x"]
        use_alpha = self.alpha * self.alpha_mask
        use_alpha = use_alpha.numpy()

        psf_intensity = np.reshape(np.transpose(self.psf_intensity, [1, 0, 2, 3, 4]), [4, -1, pixy, pixx])
        num_im = self.net_psf.shape[0]
        net_psf = np.reshape(self.net_psf, [num_im, -1, pixy, pixx])
        num_batch = net_psf.shape[1]

        ### Plot the set of net psfs and targets
        fig = plt.figure(figsize=(num_batch * 10, num_im * 10))
        ax = gF.addAxis(fig, num_im, num_batch)
        iter = 0
        for r in range(num_im):
            for c in range(num_batch):
                ax[iter].imshow(net_psf[r, c, :, :], norm=TwoSlopeNorm(0), cmap="seismic")
                ax[iter].set_title(np.array2string(np.round(use_alpha[r, :], 3)))
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "NetPSF_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "NetPSF_epoch" + epoch_str + ".png")
        plt.close()

        # Plot the component PSFs
        fig = plt.figure(figsize=(10 * num_batch, 40))
        ax = gF.addAxis(fig, 4, num_batch)
        iter = 0
        for r in range(4):
            for c in range(num_batch):
                im = ax[iter].imshow(psf_intensity[r, c, :, :])
                ax[iter].set_title(f"Energy: {np.sum(psf_intensity[r,c]):2.2f}")
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "PSF_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "PSF_epoch" + epoch_str + ".png")
        plt.close()

        # Plot the component PSFs with decomposition scaling
        decomp = psf_intensity * use_alpha.T[:, :, np.newaxis, np.newaxis]
        min_val = np.minimum(-1e-6, np.min(decomp))
        max_val = np.maximum(1e-6, np.max(decomp))
        fig = plt.figure(figsize=(10 * num_im, 40))
        ax = gF.addAxis(fig, 4, num_im)
        iter = 0
        for r in range(4):
            for c in range(num_im):
                im = ax[iter].imshow(decomp[r, c, :, :], norm=TwoSlopeNorm(0, min_val, max_val), cmap="seismic")
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "PSFDecomposition_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "PSFDecomposition_epoch" + epoch_str + ".png")
        plt.close()

        # Plot the loss
        fig = plt.figure(figsize=(15, 15))
        ax = gF.addAxis(fig, 1, 1)
        total_loss = np.array(self.psf_loss_vector) + np.array(self.energy_loss_vector) + np.array(self.bias_loss_vector)
        ax[0].plot(total_loss, "k-", label="total loss")
        ax[0].plot(self.psf_loss_vector, "b--", label="psf loss")
        ax[0].plot(self.bias_loss_vector, "r--", label="bias loss")
        ax[0].plot(self.energy_loss_vector, "g--", label="energy loss")
        gF.formatPlots(fig, ax[0], None, addLegend=True)
        plt.savefig(self.savepath + "/pdf_images/" + "epoch" + epoch_str + "CheckpointVisualize" + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "epoch" + epoch_str + "CheckpointVisualize" + ".png")
        plt.close()

        print("Saving Lens")
        self.save_optimized_lens()

        return


class MIS_Optimizer_PSF_IDEAL(MIS_Optimizer_Base):
    # This class optimizes over an idealized scenario where you have four fully decoupled phase modulation patterns at the optic
    # and thus four fully decoupled PSFS. It used as a comparitive investigation for the ablation and regularization study

    def __init__(self, alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, target_psf, energy_scale, bias_scale, savepath, saveAtEpochs):
        super().__init__(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, energy_scale, bias_scale, savepath, saveAtEpochs)
        self.target_psf = tf.convert_to_tensor(target_psf, dtype=tf.float64)

    def __call__(self):
        # Compute the psfs in this case with decompouled hologram phase masks
        phase = self.latent_tensor_variable
        psf_intensity, psf_phase = self.psf_layer([tf.ones_like(phase), phase], self.point_source_locs, batch_loop=False)
        use_alpha = self.alpha * self.alpha_mask
        net_psf = tf.math.reduce_sum(use_alpha[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis] * psf_intensity, axis=2)

        psf_loss = tf.math.reduce_sum(tf.math.abs(net_psf / tf.norm(net_psf, axis=[-2, -1], keepdims=True) - self.target_psf / tf.norm(self.target_psf, axis=[-2, -1], keepdims=True)))
        self.psf_loss_vector.append(psf_loss.numpy())
        self.net_psf = net_psf.numpy()

        # Compute loss
        return psf_loss + self.compute_regularization_loss(psf_intensity, use_alpha)

    def save_optimized_lens(self):
        ### Get the optimized optic to package and save
        phase = self.latent_tensor_variable
        trans = tf.ones_like(phase)
        aperture = self.psf_layer.aperture_trans
        data = {
            "latent_tensor": self.latent_tensor_variable,
            "alpha": self.alpha,
            "alpha_mask": self.alpha_mask,
            "transmittance": trans,
            "phase": phase,
            "aperture": aperture,
        }
        data_path = self.savepath + "optimized_lens.pickle"
        with open(data_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return


class MIS_Optimizer_Images(MIS_Optimizer_Base):
    def __init__(self, alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, test_im, target_filt, SNR, energy_scale, bias_scale, savepath, saveAtEpochs):
        super().__init__(alpha_mask, init_alpha, init_norm_param, point_source_locs, propagation_parameters, energy_scale, bias_scale, savepath, saveAtEpochs)
        sensor_emv = df_im.BFS_PGE_51S5
        num_wl = len(propagation_parameters["wavelength_set_m"])
        num_photons = df_im.SNR_to_meanPhotons(SNR, sensor_emv)
        test_im = test_im / tf.reduce_max(test_im) * num_photons
        filt_image = df_im.general_convolve(test_im, target_filt, rfft=True)
        filt_image_stack = tf.tile(filt_image, [num_wl, 1, 1])

        self.mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense256_U350_H600", pmin=0.042, pmax=0.875)
        self.im_layer = df_im.Fronto_Planar_renderer_incoherent(sensor_emv)
        self.test_im = test_im[tf.newaxis, tf.newaxis, :, :, :]
        self.targ_im = filt_image_stack[:, tf.newaxis, :, :]
        self.num_wl = num_wl

    def __call__(self):
        # Compute the PSFs
        wavelength_set_m = self.propagation_parameters["wavelength_set_m"]
        out = self.mlp_latent_layer(self.latent_tensor_variable, wavelength_set_m)
        psf_intensity, psf_phase = self.psf_layer(out, self.point_source_locs, batch_loop=False)
        psf_intensity = compute_PSFs_with_interference(psf_intensity, psf_phase)

        use_alpha = self.alpha * self.alpha_mask
        out_im = self.im_layer(psf_intensity, self.test_im, rfft=True)
        net_im = tf.math.reduce_sum(use_alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis] * out_im, axis=1)

        # Compute image loss
        im_loss = tf.math.reduce_sum(tf.math.abs(net_im / tf.norm(net_im, axis=(-2, -1), keepdims=True) - self.targ_im / tf.norm(self.targ_im, axis=(-2, -1), keepdims=True)))
        self.psf_loss_vector.append(im_loss.numpy())  # resuse the psf variable name althugh we are storing images
        self.net_im = net_im.numpy()

        # Compute loss
        return im_loss + self.compute_regularization_loss(psf_intensity, use_alpha)

    def visualizeTrainingCheckpoint(self, epoch_str, saveaspdf=False):
        # Designed this code to work for (one of) either broadband study or depth study where one dimension is singular
        # Transpose so polarization batch is the first dimension so we can reuse the wavelength or depth simulation code
        pix_num = self.propagation_parameters["sensor_pixel_number"]
        pixy, pixx = pix_num["y"], pix_num["x"]
        im_shape = self.net_im.shape
        imy, imx = im_shape[-2], im_shape[-1]
        num_im = im_shape[0]

        use_alpha = self.alpha * self.alpha_mask
        net_psf = tf.math.reduce_sum(use_alpha[:, :, tf.newaxis, tf.newaxis, tf.newaxis] * self.psf_intensity, axis=1, keepdims=True).numpy()
        net_psf = np.reshape(net_psf, [num_im, -1, pixy, pixx])
        psf_intensity = np.reshape(self.psf_intensity, [num_im, -1, pixy, pixx])
        net_im = self.net_im
        num_batch = net_im.shape[1]

        ### Plot the set of net images
        fig = plt.figure(figsize=(num_batch * 10, num_im * 10))
        ax = gF.addAxis(fig, num_im, num_batch)
        iter = 0
        for r in range(num_im):
            for c in range(num_batch):
                ax[iter].imshow(net_im[r, c, :, :], norm=TwoSlopeNorm(0), cmap="seismic")
                ax[iter].set_title(np.array2string(np.round(use_alpha[0, :], 3)))
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "NetIm_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "NetIm_epoch" + epoch_str + ".png")

        ### Plot the set of net images
        psf_set = np.concatenate((psf_intensity, net_psf), axis=1)
        fig = plt.figure(figsize=(num_im * 10, 5 * 10))
        ax = gF.addAxis(fig, 5, num_im)
        iter = 0
        for r in range(5):
            for c in range(num_im):
                if r < 4:
                    ax[iter].imshow(psf_set[c, r, :, :])
                else:
                    ax[iter].imshow(psf_set[c, r, :, :], norm=TwoSlopeNorm(0), cmap="seismic")
                iter = iter + 1
        plt.savefig(self.savepath + "/pdf_images/" + "NetPsf_epoch" + epoch_str + ".pdf")
        plt.savefig(self.savepath + "/png_images/" + "NetPsf_epoch" + epoch_str + ".png")

        # Plot the loss
        fig = plt.figure(figsize=(15, 15))
        ax = gF.addAxis(fig, 1, 1)
        total_loss = np.array(self.psf_loss_vector) + np.array(self.energy_loss_vector) + np.array(self.bias_loss_vector)
        ax[0].plot(total_loss, "k-", label="total loss")
        ax[0].plot(self.psf_loss_vector, "b--", label="psf loss")
        ax[0].plot(self.bias_loss_vector, "r--", label="bias loss")
        ax[0].plot(self.energy_loss_vector, "g--", label="energy loss")
        gF.formatPlots(fig, ax[0], None, addLegend=True)
        plt.savefig(self.savepath + "/png_images/" + "epoch" + epoch_str + ".png")

        plt.close()
        print("Saving Lens")
        self.save_optimized_lens()

        return
