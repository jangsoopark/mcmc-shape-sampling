import matplotlib.pyplot as plt
import numpy as np
import sys

g_kernel = lambda a, b: 1/np.sqrt(np.pi)/b * np.exp(-(a**2)/2/b/b)
heavi_side = lambda a: int(a > 1)


def compute_distance_template(phi1, phi2, size):

    area = 0

    for i in range(size):
        if phi1[i] > 0 > phi2[i] or phi1[i] < 0 < phi2[i]:
            area += 1

    return area


def shape_kernel_size(training_phi, num_shapes, size):

    sum_sq = 0.0
    sum_ = 0.0

    dist_matrix = np.zeros((num_shapes, num_shapes))

    for i in range(num_shapes):
        for j in range(num_shapes):
            dist_matrix[i][j] = compute_distance_template(training_phi[:, i], training_phi[:, j], size)
            sum_sq += dist_matrix[i, j] * dist_matrix[i, j]
            sum_ += dist_matrix[i, j]
    avg = sum_ / (num_shapes ** 2)
    sigma = np.sqrt(sum_sq / (num_shapes ** 2) - avg ** 2)

    return sigma


def scale_level_set_function(phi, factor, size):

    for i in range(size):
        phi[i] *= factor

    return phi


def inverse_tp(psi, p, filling, x_coord, y_coord, size, size_i, size_j):

    a = p[0]
    b = p[1]
    theta = p[2]
    h = p[3]

    ic = (size_i + 1) >> 1
    jc = (size_j + 1) >> 1

    new_psi = np.zeros(size)
    domain = np.zeros(size)

    for i in range(size):

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        ii = int(np.floor((cos_theta * (x_coord[i] + a) - sin_theta * (y_coord[i] + b)) * h + ic + 0.5))
        jj = int(np.floor((sin_theta * (x_coord[i] + a) + cos_theta * (y_coord[i] + b)) * h + jc + 0.5))

        if 0 < ii <= size_i and 0 < jj <= size_j:
            domain[i] = 1
            new_psi[i] = psi[ii - 1 + (jj - 1) * size_i]
        else:
            domain[i] = 0
            new_psi[i] = filling

    return new_psi, domain


def inverse_tp_sdf(psi, p, filling, x_coord, y_coord, size, size_i, size_j):

    new_psi, domain = inverse_tp(psi, p, filling, x_coord, y_coord, size, size_i, size_j)
    return scale_level_set_function(new_psi, 1.0 / p[3], size)


def tp(psi, p, filling, x_coord, y_coord, size, size_i, size_j):

    a = p[0]
    b = p[1]
    theta = p[2]
    h = p[3]

    ic = (size_i + 1) >> 1
    jc = (size_j + 1) >> 1

    new_psi = np.zeros(size)
    domain = np.zeros(size)

    for i in range(size):

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        ii = int(np.floor((cos_theta * x_coord[i] + sin_theta * y_coord[i]) / h + ic - a + 0.5))
        jj = int(np.floor((-sin_theta * x_coord[i] + cos_theta * y_coord[i]) / h + jc - b + 0.5))

        if 0 < ii <= size_i and 0 < jj <= size_j:
            domain[i] = 1
            new_psi[i] = psi[ii - 1 + (jj - 1) * size_i]
        else:
            domain[i] = 0
            new_psi[i] = filling

    return new_psi, domain


def tp_sdf(psi, p, filling, x_coord, y_coord, size, size_i, size_j):

    new_psi, domain = tp(psi, p, filling, x_coord, y_coord, size, size_i, size_j)
    return scale_level_set_function(new_psi, p[3], size), domain


def calc_gradient(a, ax, ay, i_coord, j_coord, size, size_i, size_j):

    for k in range(size):
        i = i_coord[k]
        j = j_coord[k]

        if i == 1:
            ax[k] = a[k + 1] - a[k]
        elif i == size_i:
            ax[k] = a[k] - a[k - 1]
        else:
            ax[k] = (a[k + 1] - a[k - 1]) / 2

        if j == 1:
            ay[k] = a[k + size_i] - a[k]
        elif j == size_j:
            ay[k] = a[k] - a[k - size_i]
        else:
            ay[k] = (a[k + size_i] - a[k - size_i]) / 2

    return a, ax, ay


def update_pose(training_image, psi, num_shapes, pose, r, i_coord, j_coord, x_coord, y_coord, size, size_i, size_j):

    num_pose = 4
    ratio_threshold = 0.01

    grad_tilde_image = np.zeros((num_pose, size))
    grad_e_pose = np.zeros(num_pose)
    normalizer = np.zeros(num_pose)

    filling = 0
    factor = 1

    ratio = 1.0
    prev_energy = sys.maxsize

    image = np.zeros(size)
    image_x = np.zeros(size)
    image_y = np.zeros(size)

    num1 = np.zeros(num_pose)
    num3 = np.zeros(num_pose)

    for i in range(size):
        if psi[i] < 0:
            image[i] = 1.0
        else:
            image[i] = 0.0

    image, image_x, image_y = calc_gradient(image, image_x, image_y, i_coord, j_coord, size, size_i, size_j)

    ct = 0
    while ratio > ratio_threshold:

        tp_image_x, domain = tp(image_x, pose, filling, x_coord, y_coord, size, size_i, size_j)
        tp_image_y, domain = tp(image_y, pose, filling, x_coord, y_coord, size, size_i, size_j)

        tilde_image, _ = tp(image, pose, filling, x_coord, y_coord, size, size_i, size_j)

        sin_theta = np.sin(pose[2])
        cos_theta = np.cos(pose[2])
        h = pose[3]

        for i in range(size):
            grad_tilde_image[0][i] = -tp_image_x[i]
            grad_tilde_image[1][i] = -tp_image_y[i]
            grad_tilde_image[2][i] = (tp_image_x[i] * (-sin_theta * x_coord[i] + cos_theta * y_coord[i])
                                      + tp_image_y[i] * (-cos_theta * x_coord[i] - sin_theta * y_coord[i])) / h
            grad_tilde_image[3][i] = -(tp_image_x[i] * (cos_theta * x_coord[i] + sin_theta * y_coord[i])
                                       + tp_image_y[i] * (-sin_theta * x_coord[i] + cos_theta * y_coord[i])) / h / h

        energy = 0

        for i in range(num_shapes):

            for l in range(num_pose):
                num1[l] = 0
                num3[l] = 0
            num2 = 0
            denom = 0

            for j in range(size):

                sum_ = tilde_image[j] + training_image[i][j]
                diff = tilde_image[j] - training_image[i][j]
                num2 += diff ** 2
                denom += sum_ ** 2
                
                for l in range(num_pose):

                    num1[l] += diff * grad_tilde_image[l][j]
                    num3[l] += sum_ * grad_tilde_image[l][j]

            energy += num2 / denom

            for l in range(num_pose):
                grad_e_pose[l] += 2 * num1[l] / denom - 2 * num2 * num3[l] / denom / denom

        max_r = 0
        for i in range(size):
            if tilde_image[i] > 0 and r[i] > max_r:
                max_r = r[i]

        normalizer[0] = np.fabs(grad_e_pose[0])
        normalizer[1] = np.fabs(grad_e_pose[1])
        normalizer[2] = max_r * np.fabs(grad_e_pose[2])
        normalizer[3] = max_r * np.fabs(grad_e_pose[3])

        for l in range(num_pose):
            if normalizer[l] != 0:
                grad_e_pose[l] /= normalizer[l]

        for l in range(num_pose):
            pose[l] -= factor * grad_e_pose[l]

        if ct > 0 and energy > prev_energy:
            factor /= 2

        if ct >= 1:
            ratio = np.fabs(energy - prev_energy) / prev_energy

        prev_energy = energy
        ct += 1

    return pose


def calculate_image_force(image, psi, size):

    sum1 = 0
    sum2 = 0

    area1 = 0
    area2 = 0

    f = np.zeros(size)
    for i in range(size):

        if psi[i] < 0.0:
            sum1 += image[i]
            area1 += 1
        else:
            sum2 += image[i]
            area2 += 1

    c1 = sum1 / area1
    c2 = sum2 / area2

    for i in range(size):
        f[i] = -(2 * image[i] - c1 - c2) * (c1 - c2)

    return f


def compute_xyij_coordinate_and_universe_and_r(size, size_i, size_j):

    x_coord = np.zeros(size)
    y_coord = np.zeros(size)
    r = np.zeros(size)
    i_coord = np.zeros(size)
    j_coord = np.zeros(size)
    universe = np.zeros(size)
    
    for i in range(size):

        ii = i % size_i + 1
        jj = int(np.floor(i / size_i) + 1)

        i_coord[i] = ii
        j_coord[i] = jj

        x_coord[i] = ii - (size_i + 1.0) / 2.0
        y_coord[i] = jj - (size_j + 1.0) / 2.0

        r[i] = np.sqrt(x_coord[i] ** 2 + y_coord[i] ** 2)
        universe[i] = 1

    return x_coord, y_coord, i_coord, j_coord, universe, r


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
                             size, size_i, size_j):

    class_weights = np.zeros(num_classes)
    shape_weights = np.zeros(num_shapes_in_each_class)

    filling = 0.0
    p_of_phi = 0
    counter = 0
    loop = True

    if sample_iter == 1:

        for i in range(num_shapes):
            counter += 1

            tilde_phi, domain = tp_sdf(phi, all_pose[int(i / num_shapes_in_each_class)], filling,
                                       x_coord, y_coord, size, size_i, size_j)
            
            dist = compute_distance_template(tilde_phi, training_phi[:, i], size)
            weight = g_kernel(dist, k_size)

            p_of_phi += weight

            if counter == num_shapes_in_each_class:
                counter = 0
                class_weights[int(i/num_shapes_in_each_class)] = p_of_phi

        for i in range(num_classes):
            class_weights[i] = class_weights[i] / class_weights[num_classes - 1]

        i = 0
        while loop:
            if random_number_for_class_selection <= class_weights[i]:
                randomly_selected_class_id = i
                loop = False
            i += 1
        current_selected_class_id = randomly_selected_class_id

        for i in range(4):
            pose[i] = all_pose[current_selected_class_id, i]

    tilde_phi, domain = tp_sdf(phi, pose, filling, x_coord, y_coord, size, size_i, size_j)

    p_of_phi = 0
    counter = 0

    for i in range(current_selected_class_id * num_shapes_in_each_class,
                   (current_selected_class_id + 1) * num_shapes_in_each_class):
        dist = compute_distance_template(tilde_phi, training_phi[:, i], size)
        weight = g_kernel(dist, k_size)
        p_of_phi += weight
        shape_weights[counter] = p_of_phi
        counter += 1

    for i in range(num_shapes_in_each_class):
        shape_weights[i] = shape_weights[i] / shape_weights[num_shapes_in_each_class - 1]

    p_of_selecting_shapes = 1.0

    for i in range(random_number_for_total_shapes):

        loop = True
        counter = 0

        while loop:
            if random_number_array[i] < shape_weights[counter]:

                selected_shape_ids[i] = current_selected_class_id * num_shapes_in_each_class + counter
                current_selected_shape_id[i] = selected_shape_ids[i]

                dist = compute_distance_template(tilde_phi, training_phi[:, int(selected_shape_ids[i])], size)
                p_of_selecting_shapes *= g_kernel(dist, k_size)

                loop = False

            counter += 1

    p_forward = p_of_selecting_shapes

    p_reverse = 1.0
    if accepted_count != 0:
        for i in range(random_number_for_total_shapes):
            dist = compute_distance_template(tilde_phi, training_phi[:, previous_selected_shape_id[i]], size)
            p_reverse *= g_kernel(dist, k_size)

    return selected_shape_ids, p_forward, p_reverse, current_selected_class_id, current_selected_shape_id


def compute_shape_force(phi, pose, training_phi, k_size,
                        random_number2, selected_shape_ids, x_coord, y_coord, size, size_i, size_j):

    filling = 0.0
    tilde_force = np.zeros(size)

    tilde_phi, domain = tp_sdf(phi, pose, filling, x_coord, y_coord, size, size_i, size_j)

    p_of_phi = 0

    for i in range(random_number2):
        dist = compute_distance_template(tilde_phi, training_phi[:, selected_shape_ids[i]], size)
        weight = g_kernel(dist, k_size)

        p_of_phi += weight
        for j in range(size):
            if domain[j]:
                tilde_force[j] += -weight * (
                    1 - 2 * heavi_side(training_phi[j, selected_shape_ids[i]])) / random_number2

    p_of_phi /= random_number2
    factor = 1 / (p_of_phi * random_number2)
    for i in range(size):
        tilde_force[i] *= factor

    shape_f = inverse_tp_sdf(tilde_force, pose, filling, x_coord, y_coord, size, size_i, size_j)

    return shape_f


def add_shape_force_to_data_force(shape_f, f, size, alpha, beta):

    max_data_f = 0.0
    max_shape_f = 0.0

    for i in range(size):

        if f[i] != 0:
            if np.fabs(f[i]) > max_data_f:
                max_data_f = np.fabs(f[i])
            if np.fabs(shape_f[i]) > max_shape_f:
                max_shape_f =np.fabs(shape_f[i])

    internal_factor = max_shape_f / max_data_f

    for i in range(size):
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
    
    training_phi = training_phi_matrix.reshape((size, num_of_all_training_shapes))
    pose_for_each_class = pose_for_each_class_vector.reshape((num_of_classes, 4))

    x_coord, y_coord, i_coord, j_coord, universe, r = compute_xyij_coordinate_and_universe_and_r(size, size_i, size_j)

    ct_pos = 0
    ct_neg = 0
    for i in range(size):
        if psi[i] < 0:
            ct_neg += 1
        elif psi[i] > 0:
            ct_pos += 1

    if ct_pos == 0 or ct_neg == 0:
        print('one region has disappeared; we stop the curve evolution')
    
    kernel_size = shape_kernel_size(training_phi, num_of_all_training_shapes, size)
    if k == 0:
        selected_shape_ids, p_forward, \
            p_reverse, current_selected_class_id, current_selected_shape_id = perform_random_selection(
                psi, pose, training_phi, num_of_all_training_shapes, kernel_size,
                random_number_for_class_selection, random_number_for_total_shapes,
                random_number_array, num_of_classes, num_of_shapes_in_each_class,
                selected_shape_ids,
                current_selected_class_id, current_selected_shape_id,
                previous_selected_shape_id,
                accepted_count,
                sampling_iteration, pose_for_each_class,
                x_coord, y_coord,
                size, size_i, size_j)

    f = calculate_image_force(image, psi, size)
    shape_f = compute_shape_force(
        psi, pose, training_phi,
        kernel_size, random_number_for_total_shapes,
        current_selected_shape_id,
        x_coord, y_coord, size, size_i, size_j)
    f = add_shape_force_to_data_force(shape_f, f, size, alpha, beta)

    for i in range(size):
        if narrow_band[i] == 1:
            psi[i] += dt * f[i]

    return psi, p_forward, p_reverse, current_selected_class_id, current_selected_shape_id

