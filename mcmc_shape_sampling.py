from six.moves import xrange

from utils import *

import numpy as np


def perform_random_selection(phi, pose, training_phi, num_shapes, k_size,
                             random_number_for_class_selection,
                             random_number_for_total_shapes,
                             random_number_array,
                             num_classes, num_shapes_in_each_class,
                             selected_shape_ids,
                             current_selected_class_id,
                             current_selected_shape_id,
                             previous_selected_shape_id,
                             accepted_count,
                             sample_iter, all_pose,
                             x_coord, y_coord,
                             size_i, size_j):
    class_weights = np.zeros(num_classes)
    shape_weights = np.zeros(num_shapes_in_each_class)

    filling = 0.0
    p_of_phi = 0
    counter = 0
    loop = True

    if sample_iter == 1:

        for i in xrange(num_shapes):
            counter += 1

            tilde_phi, domain = scaled_align(phi, all_pose[int(i / num_shapes_in_each_class)], filling,
                                             x_coord, y_coord, size_i, size_j)

            dist = compute_distance(tilde_phi, training_phi[i])
            weight = gaussian_kernel(dist, k_size)

            p_of_phi += weight

            if counter == num_shapes_in_each_class:
                counter = 0
                class_weights[int(i / num_shapes_in_each_class)] = p_of_phi

        for i in xrange(num_classes):
            class_weights[i] = class_weights[i] / class_weights[num_classes - 1]

        i = 0
        while loop:
            if random_number_for_class_selection <= class_weights[i]:
                randomly_selected_class_id = i
                loop = False
            i += 1
        current_selected_class_id = randomly_selected_class_id

        for i in xrange(4):
            pose[i] = all_pose[current_selected_class_id, i]

    tilde_phi, domain = scaled_align(phi, pose, filling, x_coord, y_coord, size_i, size_j)

    p_of_phi = 0
    counter = 0

    for i in xrange(current_selected_class_id * num_shapes_in_each_class,
                    (current_selected_class_id + 1) * num_shapes_in_each_class):

        dist = compute_distance(tilde_phi, training_phi[i, :])
        weight = gaussian_kernel(dist, k_size)
        p_of_phi += weight
        shape_weights[counter] = p_of_phi
        counter += 1

    for i in xrange(num_shapes_in_each_class):
        shape_weights[i] = shape_weights[i] / shape_weights[num_shapes_in_each_class - 1]

    p_of_selecting_shapes = 1.0

    for i in xrange(random_number_for_total_shapes):

        loop = True
        counter = 0

        while loop:
            if random_number_array[i] < shape_weights[counter]:
                selected_shape_ids[i] = current_selected_class_id * num_shapes_in_each_class + counter
                current_selected_shape_id[i] = selected_shape_ids[i]

                dist = compute_distance(tilde_phi, training_phi[int(selected_shape_ids[i])])
                p_of_selecting_shapes *= gaussian_kernel(dist, k_size)

                loop = False

            counter += 1

    p_forward = p_of_selecting_shapes

    p_reverse = 1.0
    if accepted_count != 0:
        for i in xrange(random_number_for_total_shapes):
            dist = compute_distance(tilde_phi, training_phi[previous_selected_shape_id[i]])
            p_reverse *= gaussian_kernel(dist, k_size)

    return selected_shape_ids, p_forward, p_reverse, current_selected_class_id, current_selected_shape_id


def add_shape_force_to_data_force(shape_f, f, size, alpha, beta):
    max_data_f = 0.0
    max_shape_f = 0.0

    for i in xrange(size):

        if f[i] != 0:
            if np.fabs(f[i]) > max_data_f:
                max_data_f = np.fabs(f[i])
            if np.fabs(shape_f[i]) > max_shape_f:
                max_shape_f = np.fabs(shape_f[i])

    internal_factor = max_shape_f / max_data_f

    for i in xrange(size):
        f[i] = beta * f[i] + alpha * shape_f[i] / internal_factor

    return f


def mcmc_shape_sampling(image, psi,
                        narrow_band, training_phi_matrix,
                        pose_for_each_class_vector,
                        num_of_classes, num_of_shapes_in_each_class,
                        dt, alpha, sampling_iteration,
                        random_number_for_class_selection,
                        random_number_for_total_shapes,
                        random_number_array,
                        p_forward, p_reverse,
                        current_selected_class_id,
                        current_selected_shape_id,
                        previous_selected_shape_id,
                        accepted_count, pose,
                        k, beta, size_i, size_j):
    size = size_i * size_j

    selected_shape_ids = np.zeros(random_number_for_total_shapes)

    num_of_all_training_shapes = num_of_classes * num_of_shapes_in_each_class

    training_phi = training_phi_matrix.reshape((num_of_all_training_shapes, size))
    pose_for_each_class = pose_for_each_class_vector.reshape((num_of_classes, 4))

    x_coord, y_coord, i_coord, j_coord, universe, r = compute_parameters(size_i, size_j)

    ct_pos = 0
    ct_neg = 0
    for i in xrange(size):
        if psi[i] < 0:
            ct_neg += 1
        elif psi[i] > 0:
            ct_pos += 1

    if ct_pos == 0 or ct_neg == 0:
        print('one region has disappeared; we stop the curve evolution')

    kernel_size = shape_kernel_size(training_phi, num_of_all_training_shapes)
    if k == 0:
        result = perform_random_selection(
            psi, pose, training_phi, num_of_all_training_shapes, kernel_size,
            random_number_for_class_selection, random_number_for_total_shapes,
            random_number_array, num_of_classes, num_of_shapes_in_each_class,
            selected_shape_ids,
            current_selected_class_id, current_selected_shape_id,
            previous_selected_shape_id,
            accepted_count,
            sampling_iteration, pose_for_each_class,
            x_coord, y_coord,
            size_i, size_j)
        selected_shape_ids, p_forward, p_reverse, current_selected_class_id, current_selected_shape_id = result

    f = calculate_image_force(image, psi, size)
    shape_f = compute_shape_force(
        psi, pose, training_phi,
        kernel_size, random_number_for_total_shapes,
        current_selected_shape_id,
        x_coord, y_coord, size_i, size_j)

    f = add_shape_force_to_data_force(shape_f, f, size, alpha, beta)

    for i in range(size):
        if narrow_band[i] == 1:
            psi[i] += dt * f[i]

    return psi, p_forward, p_reverse, current_selected_class_id, current_selected_shape_id
