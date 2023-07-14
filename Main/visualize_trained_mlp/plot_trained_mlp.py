import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import dflat.neural_optical_layer as df_neural
import dflat.plot_utilities as df_plt
import dflat.datasets_metasurface_cells as df_library

dirname = str(Path(__file__).parent) + "/"


def view_trained_MLP_vs_Data():
    savepath = dirname

    ### Get nanofin library dataset and the corresponding neural model (wrapped as a mlp layer)
    library = df_library.Nanofins_U350nm_H600nm()
    lib_trans = library.transmission
    lib_phase = library.phase
    lib_phase = np.arctan2(np.sin(lib_phase), np.cos(lib_phase))  # this is the phase wrap used during training time
    wx, wy, wl = library.param1, library.param2, library.param3

    neural_model = df_neural.MLP_Layer("MLP_Nanofins_Dense1024_U350_H600")

    ### Plot transmission and phase (loop over wavelength)
    for wl_use in [532e-9]:
        wl_idx = np.argmin(np.abs(wl - wl_use))

        Lx, Ly = np.meshgrid(np.arange(60e-9, 301e-9, 1e-9), np.arange(60e-9, 301e-9, 1e-9))
        shape_vec = np.stack((Lx, Ly), axis=0)
        param_vec = neural_model.shape_to_param(shape_vec)
        trans, phase = neural_model(param_vec, [wl_use])
        trans = trans**2  # convert transmittance to transmission percent matching the fdtd train data

        fig = plt.figure(figsize=(20, 20))
        ax = df_plt.addAxis(fig, 2, 2)
        im1 = ax[0].imshow(lib_trans[0, :, :, wl_idx], vmin=0, vmax=1)
        im2 = ax[1].imshow(lib_trans[1, :, :, wl_idx], vmin=0, vmax=1)
        im3 = ax[2].imshow(trans[0, 0, :, :], vmin=0, vmax=1)
        im4 = ax[3].imshow(trans[0, 1, :, :], vmin=0, vmax=1)
        df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True)
        df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True, rmvyLabel=True)
        df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9)
        df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9, rmvyLabel=True)
        plt.savefig(savepath + "/png_img/" + f"spatial_slice_trans_{wl_use*1e9}.png")
        plt.savefig(savepath + "/pdf_img/" + f"spatial_slice_trans_{wl_use*1e9}.pdf")

        fig = plt.figure(figsize=(20, 20))
        ax = df_plt.addAxis(fig, 2, 2)
        im1 = ax[0].imshow(lib_phase[0, :, :, wl_idx], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        im2 = ax[1].imshow(lib_phase[1, :, :, wl_idx], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        im3 = ax[2].imshow(phase[0, 0, :, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        im4 = ax[3].imshow(phase[0, 1, :, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True)
        df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True, rmvyLabel=True)
        df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9)
        df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9, rmvyLabel=True)
        plt.savefig(savepath + "/png_img/" + f"spatial_slice_phase_{wl_use*1e9}.png")
        plt.savefig(savepath + "/pdf_img/" + f"spatial_slice_phase_{wl_use*1e9}.pdf")

    ### Plot wavelength slice
    yidx = 24
    print("y Slice: ", wy[yidx])
    Lx, Ly = np.meshgrid(np.arange(60e-9, 301e-9, 1e-9), wy[yidx])
    shape_vec = np.stack((Lx, Ly), axis=0)
    param_vec = neural_model.shape_to_param(shape_vec)
    Ll = np.arange(310e-9, 751e-9, 1e-9)
    trans, phase = neural_model(param_vec, Ll)
    trans = trans**2  # convert transmittance to transmission percent matching the fdtd train data

    fig = plt.figure(figsize=(15, 15))
    ax = df_plt.addAxis(fig, 2, 2)
    im1 = ax[0].imshow(lib_trans[0, 24, :, :].T, vmin=0, vmax=1)
    im2 = ax[1].imshow(lib_trans[1, 24, :, :].T, vmin=0, vmax=1)
    im3 = ax[2].imshow(trans[:, 0, 0, :], vmin=0, vmax=1)
    im4 = ax[3].imshow(trans[:, 1, 0, :], vmin=0, vmax=1)
    df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True)
    df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True, rmvyLabel=True)
    df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9)
    df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9, rmvyLabel=True)
    plt.savefig(savepath + "/png_img/" + "spectral_slice_trans.png")
    plt.savefig(savepath + "/pdf_img/" + "spectral_slice_trans.pdf")

    fig = plt.figure(figsize=(15, 15))
    ax = df_plt.addAxis(fig, 2, 2)
    im1 = ax[0].imshow(lib_phase[0, 24, :, :].T, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    im2 = ax[1].imshow(lib_phase[1, 24, :, :].T, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    im3 = ax[2].imshow(phase[:, 0, 0, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    im4 = ax[3].imshow(phase[:, 1, 0, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True)
    df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True, rmvyLabel=True)
    df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9)
    df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9, rmvyLabel=True)
    plt.savefig(savepath + "/png_img/" + "spectral_slice_phase.png")
    plt.savefig(savepath + "/pdf_img/" + "spectral_slice_phase.pdf")

    plt.close()

    return


if __name__ == "__main__":
    view_trained_MLP_vs_Data()
