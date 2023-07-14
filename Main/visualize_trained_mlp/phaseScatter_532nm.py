import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import dflat.neural_optical_layer as df_neural
import dflat.plot_utilities as df_plt

dirname = str(Path(__file__).parent) + "/"

### Get nanofin library dataset and the corresponding neural model (wrapped as a mlp layer)
neural_model = df_neural.MLP_Layer("MLP_Nanofins_Dense1024_U350_H600")

### Get the optical response for single wavelength so we can display joint polarization phase coverage
Lx, Ly = np.meshgrid(np.arange(60e-9, 301e-9, 1e-9), np.arange(60e-9, 301e-9, 1e-9))
shape_vec = np.stack((Lx, Ly), axis=0)
param_vec = neural_model.shape_to_param(shape_vec)
trans, phase = neural_model(param_vec, [532e-9])
trans = trans**2  # convert transmittance to transmission percent matching the fdtd train data
data_shape = phase.shape

param_vec = param_vec.numpy()
avg_trans = np.clip(np.mean(trans[0, :, :, :], axis=0), 0, 1)

clist = np.stack((param_vec[0, :, :].flatten(), np.zeros(data_shape[-2] * data_shape[-1]), param_vec[1, :, :].flatten(), avg_trans.flatten()))
phasex = phase[0, 0, :, :].numpy().flatten()
phasey = phase[0, 1, :, :].numpy().flatten()
cbar_dat = np.transpose(np.reshape(clist[0:3, :], [3, data_shape[-2], data_shape[-1]]), [1, 2, 0])

fig = plt.figure(figsize=(15, 30))
ax = df_plt.addAxis(fig, 1, 2)
ax[0].scatter(phasex, phasey, s=11, marker=".", c=clist.T)
df_plt.formatPlots(fig, ax[0], None, setAspect="equal")
ax[1].imshow(cbar_dat, extent=[np.min(Lx), np.max(Lx), np.max(Ly), np.min(Ly)])
df_plt.formatPlots(fig, ax[1], None, setAspect="equal")
plt.savefig(dirname + "png_img/scatterphase.png")
plt.savefig(dirname + "pdf_img/scatterphase.pdf")
