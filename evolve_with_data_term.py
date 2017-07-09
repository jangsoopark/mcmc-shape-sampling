from six.moves import xrange

from utils import *

import numpy as np


def evolve_with_data_term(image, psi, narrow_band, training_image_matrix, pose_for_each_class_vector,
                          num_of_classes, num_of_shapes_in_each_class, dt,
                          current_iteration, last_iteration, kappa, size_i, size_j):
    size = size_i * size_j

    temp_training_image = np.zeros((num_of_shapes_in_each_class, size))

    num_of_all_training_shapes = num_of_classes * num_of_shapes_in_each_class

    training_image = training_image_matrix.reshape((num_of_all_training_shapes, size))
    pose_for_each_class = pose_for_each_class_vector.reshape((num_of_classes, 4))

    x_coord, y_coord, i_coord, j_coord, universe, r = compute_parameters(size_i, size_j)

    ct_pos = 0
    ct_neg = 0

    for i in range(size):
        if psi[i] < 0:
            ct_neg += 1
        elif psi[i] > 0:
            ct_pos += 1

    if ct_pos == 0 or ct_neg == 0:
        print("one region has disappeared; we stop the curve evolution")

    f = calculate_image_force(image, psi, size)

    for i in range(size):
        if narrow_band[i] == 1:
            psi[i] += dt * f[i] + kappa[i]

    if current_iteration == last_iteration:
        for j in range(num_of_classes):
            for k in range(num_of_shapes_in_each_class):
                temp_training_image[k, :] = training_image[j * num_of_shapes_in_each_class + k]

            update_pose(temp_training_image, psi, num_of_shapes_in_each_class, pose_for_each_class[j],
                        r, i_coord, j_coord, x_coord, y_coord, size, size_i, size_j, )

            for k in range(4):
                pose_for_each_class_vector[j * 4 + k] = pose_for_each_class[j][k]

    return psi, pose_for_each_class
