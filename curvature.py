
import matplotlib.pyplot as plt

import numpy as np

eps = np.finfo(float).eps


def curvature(image):

    image = np.double(image)
    m, n = image.shape

    padded = np.pad(image, [1, 1], 'constant', constant_values=(1))

    fy = padded[2: , 1: n + 1] - padded[0: m, 1: n + 1]
    fx = padded[1: m + 1, 2: ] - padded[1: m + 1, 0: n]

    fyy = padded[2: , 1: n + 1] - padded[0: m, 1: n + 1] - 2 * image
    fxx = padded[1: m + 1, 2: ] - padded[1: m + 1, 0: n] - 2 * image

    fxy = 0.25 * (padded[2: , 2: ] - padded[0: m, 2: ] - padded[2: , 0: n] + padded[0: m, 0: n])

    G = np.sqrt((fx**2 + fy**2))
    K = (fxx * fy**2 - 2 * fxy * fx * fy + fyy * fx **2) / ((fx**2 + fy**2  + eps)**(1.5))

    KG = K * G
    KG[0, :] = eps
    KG[-1, :] = eps
    KG[:, 0] = eps
    KG[:, -1] = eps

    KG = KG / np.absolute(KG).max().max()

    return KG
