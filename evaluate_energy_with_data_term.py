
import numpy as np


def evaluate_energy_with_data_term(test_image, psi, mask):

    test_image = test_image / test_image.max()
    binary_curve = psi < 0

    binary_curve_in = binary_curve * (1 - mask)
    temp1 = test_image[np.where(binary_curve_in == 1)]
    c1 = np.mean(temp1)

    binary_curve_out = binary_curve_in + mask
    temp2 = test_image[np.where(binary_curve_out == 0)]
    c2 = np.mean(temp2)

    term1 = ((temp1 - c1) ** 2).sum()
    term2 = ((temp2 - c2) ** 2).sum()

    minus_log_p_of_data = term1 + term2

    return minus_log_p_of_data
