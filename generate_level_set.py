
from scipy.ndimage.morphology import distance_transform_edt

import numpy as np


def generate_level_set(psi_):

    psi = np.double( (psi_ > 0) * (distance_transform_edt(1 - (psi_ < 0)) - 0.5) - (psi_ < 0) * (distance_transform_edt(1 - (psi_ > 0)) - 0.5))

    return psi
