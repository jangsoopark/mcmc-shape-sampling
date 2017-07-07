
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion

import numpy as np


def create_narrow_band(psi, band_thickness):

    se = np.ones((band_thickness, band_thickness))

    psi_dilated = binary_dilation(psi > 0, se)
    psi_eroded = binary_erosion(psi > 0, se)

    narrow_band = psi_dilated - psi_eroded

    return narrow_band
