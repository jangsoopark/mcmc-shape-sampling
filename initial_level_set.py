
from scipy.ndimage.morphology import distance_transform_edt

import matplotlib.pyplot as plt

import numpy as np

from generate_level_set import *


def initial_level_set(circle_radius, size_i, size_j):

    centers = plt.ginput(1)[0]
    
    centers = np.round_(centers)
    c1 = centers[:]

    psi = np.zeros((size_i, size_j))
    c = np.sqrt(2) / 2
    s = np.sqrt(1 - c**2)


    for i in range(size_i):

        for j in range(size_j):

            ic = i - c1[1]
            jc = j - c1[0]

            if np.sqrt( ( (ic * c + jc * s) / 2.0)**2 + (-jc * s + ic * c)**2 ) < circle_radius:
                psi[i, j] = 1

    psi = -2 * psi + 1

    psi = generate_level_set(psi)

    return psi
