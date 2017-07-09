
from utils import *


def evaluate_energy_with_shape_prior(psi,
                                     training_phi_matrix,
                                     num_of_classes, num_of_shapes_in_each_class,
                                     pose, selected_class_id, size_i, size_j):
    size = size_i * size_j

    num_of_all_training_shapes = num_of_classes * num_of_shapes_in_each_class
    x_coord, y_coord, i_coord, j_coord, universe, r = compute_parameters(size_i, size_j)

    training_phi = training_phi_matrix.reshape((num_of_all_training_shapes, size))

    kernel_size = shape_kernel_size(training_phi, num_of_all_training_shapes)

    energy = compute_shape_energy(
        psi, pose, training_phi, num_of_all_training_shapes, kernel_size,
        selected_class_id, num_of_shapes_in_each_class, x_coord, y_coord, size_i, size_j)

    return energy
