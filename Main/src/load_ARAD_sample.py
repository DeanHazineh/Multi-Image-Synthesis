from scipy import interpolate
import numpy as np
from PIL import Image
import cv2 as cv
from pathlib import Path
import os
import mat73

from dflat.datasets_image import cv_aspectResize

def get_path_to_data(folder_name: str):
    resource_path = Path(__file__).parent
    return str(resource_path.joinpath(folder_name)) + "/"


def load_mat_dat(path):
    # Note: not sure what norm_factor is used for
    # Upon inspection, bands are provided in units of nm and range from 400 to 700 nm for Arad1k
    # with a 10 nm step size
    mat_dat = mat73.loadmat(path)
    bands = mat_dat["bands"]
    cube = mat_dat["cube"]
    norm_factor = mat_dat["norm_factor"]

    return bands, cube


def interpolate_HS_Cube(new_channels_nm, hs_cube, hs_bands):
    # Throw an error if we try to extrapolate
    if (min(new_channels_nm) < min(hs_bands) - 1) or (max(new_channels_nm) > max(hs_bands) + 1):
        raise ValueError("In generator, extrapoaltion of the ARAD dataset outside of measurement data is not allowed")

    interpfun = interpolate.interp1d(hs_bands, hs_cube, axis=-1, kind="linear", assume_sorted=True, fill_value="extrapolate", bounds_error=False)
    resampled = interpfun(new_channels_nm)

    return resampled


def load_test_ims(channels_nm, sensor_dim=None, resize_by="crop"):
    ARAD_DATA_PATH = get_path_to_data("ARAD_Sample/")

    files = os.listdir(ARAD_DATA_PATH)
    file_names = list(set([f[:-4] for f in files]))
    file_names.sort()

    hsi_dat = []
    rgb_dat = []
    for fn in file_names:
        bands, cube = load_mat_dat(ARAD_DATA_PATH + fn + ".mat")
        cube = interpolate_HS_Cube(channels_nm, cube, bands)
        rgb_image = np.asarray(Image.open(ARAD_DATA_PATH + fn + ".jpg"))

        # allow for data resizing with aspect ratio preservation
        if sensor_dim is not None:
            cube = cv_aspectResize(cube, sensor_dim, resize_by)
            rgb_image = cv_aspectResize(rgb_image, sensor_dim, resize_by)

        # I found an issue where some Hyperspectral value in the data set were negative (I don't know which is negative)
        # So to be safe and correct this issue, I want to clip below zero
        cube = np.clip(cube, 0, 1)
        hsi_dat.append(cube)
        rgb_dat.append(rgb_image)

    hsi_dat = np.stack(hsi_dat)
    rgb_dat = np.stack(rgb_dat)

    return hsi_dat, rgb_dat
