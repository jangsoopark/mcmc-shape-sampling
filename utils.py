
from six.moves import xrange

import numpy as np
import sys


def gaussian_kernel(a, b):
    return 1 / (np.sqrt(np.pi) * b) + np.exp(-(a ** 2) / (2 * b ** 2))


def heavi_side(a):
    return int(a > 1)


def scale_level_set(phi, factor):
    size = phi.shape[0]

    for i in xrange(size):
        phi[i] *= factor

    return phi


def compute_distance(phi1, phi2):
    size = phi1.shape[0]

    area = 0
    for i in xrange(size):
        if phi1[i] > 0 > phi2[i] or phi1[i] < 0 < phi2[i]:
            area += 1

    return area


def shape_kernel_size(training_phi, num_shapes):
    sum_square = 0.
    sum_ = 0.

    distance_matrix = np.zeros((num_shapes, num_shapes))
    for i in xrange(num_shapes):
        for j in xrange(num_shapes):
            distance_matrix[i][j] = compute_distance(training_phi[i], training_phi[j])
            sum_square += distance_matrix[i, j] ** 2
            sum_ += distance_matrix[i, j]

    avg = sum_ / (num_shapes ** 2)
    sigma = np.sqrt(sum_square / (num_shapes ** 2) - avg ** 2)

    return sigma


def compute_parameters(size_i, size_j):

    size = size_i * size_j
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


def align(psi, pose, filling, x_coord, y_coord, size_i, size_j):

    size = size_i * size_j

    a = pose[0]
    b = pose[1]
    theta = pose[2]
    h = pose[3]

    ic = (size_i + 1) >> 1
    jc = (size_j + 1) >> 1

    new_psi = np.zeros(size)
    domain = np.zeros(size)

    for i in xrange(size):

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


def inverse_align(psi, pose, filling, x_coord, y_coord, size_i, size_j):

    size = size_i * size_j

    a = pose[0]
    b = pose[1]
    theta = pose[2]
    h = pose[3]

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


def scaled_align(psi, pose, filling, x_coord, y_coord, size_i, size_j):

    new_psi, domain = align(psi, pose, filling, x_coord, y_coord, size_i, size_j)
    return scale_level_set(new_psi, pose[3]), domain


def inverse_scaled_align(psi, pose, filling, x_coord, y_coord, size_i, size_j):

    new_psi, domain = inverse_align(psi, pose, filling, x_coord, y_coord, size_i, size_j)
    return scale_level_set(new_psi, pose[3]), domain


def calc_gradient(a, ax, ay, i_coord, j_coord, size_i, size_j):

    size = size_i * size_j

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


def compute_shape_energy(phi, pose, training_phi, num_shapes, kernel_size, current_class_id, num_shapes_in_class,
                         x_coord, y_coord, size_i, size_j):

    filling = 0.
    shape_force = np.zeros(num_shapes)

    tilde_phi, domain = scaled_align(phi, pose, filling, x_coord, y_coord, size_i, size_j)

    p_of_phi = 0.

    for i in xrange(num_shapes):

        dist = compute_distance(tilde_phi, training_phi[i])
        weight = gaussian_kernel(dist, kernel_size)

        p_of_phi += weight
        shape_force[i] = p_of_phi

    for i in xrange(num_shapes):
        shape_force[i] = shape_force[i] / shape_force[num_shapes - 1]

    p_of_phi = 0.
    start = current_class_id * num_shapes_in_class
    end = (current_class_id + 1) * num_shapes_in_class
    for i in xrange(start, end):
        if i == 0:
            p_of_phi += shape_force[i]
        else:
            p_of_phi += (shape_force[i] - shape_force[i - 1])

    p_of_phi = p_of_phi / num_shapes_in_class
    return p_of_phi


def compute_shape_force(phi, pose, training_phi, kerner_size, random_number, selected_shape_ids,
                        x_coord, y_coord, size_i, size_j):

    size = size_i * size_j

    filling = 0.
    tilde_force = np.zeros(size)

    tilde_phi, domain = scaled_align(phi, pose, filling, x_coord, y_coord, size_i, size_j)

    p_of_phi = 0.
    for i in xrange(random_number):

        dist = compute_distance(tilde_phi, training_phi[selected_shape_ids[i]])
        weight = gaussian_kernel(dist, kerner_size)

        p_of_phi += weight
        for j in xrange(size):
            if domain[j]:
                tilde_force[j] += -weight * (
                    1 - 2 * heavi_side(training_phi[selected_shape_ids[i], j])
                ) / random_number

    p_of_phi /= random_number
    factor = 1 / (p_of_phi * random_number)
    for i in xrange(size):
        tilde_force[i] *= factor

    shape_force, domain = inverse_scaled_align(tilde_force, pose, filling, x_coord, y_coord, size_i, size_j)

    return shape_force


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


def update_pose(training_image, psi, num_shapes, pose, r, i_coord, j_coord, x_coord, y_coord, size, size_i, size_j, ):
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

    for i in xrange(size):
        if psi[i] < 0:
            image[i] = 1.0
        else:
            image[i] = 0.0

    image, image_x, image_y = calc_gradient(image, image_x, image_y, i_coord, j_coord, size_i, size_j)

    ct = 0
    while ratio > ratio_threshold:

        tp_image_x, domain = align(image_x, pose, filling, x_coord, y_coord, size_i, size_j)
        tp_image_y, domain = align(image_y, pose, filling, x_coord, y_coord, size_i, size_j)

        tilde_image, _ = align(image, pose, filling, x_coord, y_coord, size_i, size_j)

        sin_theta = np.sin(pose[2])
        cos_theta = np.cos(pose[2])
        h = pose[3]

        for i in xrange(size):
            grad_tilde_image[0][i] = -tp_image_x[i]
            grad_tilde_image[1][i] = -tp_image_y[i]
            grad_tilde_image[2][i] = (tp_image_x[i] * (-sin_theta * x_coord[i] + cos_theta * y_coord[i])
                                      + tp_image_y[i] * (-cos_theta * x_coord[i] - sin_theta * y_coord[i])) / h
            grad_tilde_image[3][i] = -(tp_image_x[i] * (cos_theta * x_coord[i] + sin_theta * y_coord[i])
                                       + tp_image_y[i] * (-sin_theta * x_coord[i] + cos_theta * y_coord[i])) / h / h

        energy = 0

        for i in xrange(num_shapes):

            for l in xrange(num_pose):
                num1[l] = 0
                num3[l] = 0
            num2 = 0
            denom = 0

            for j in xrange(size):

                sum_ = tilde_image[j] + training_image[i][j]
                diff = tilde_image[j] - training_image[i][j]
                num2 += diff ** 2
                denom += sum_ ** 2

                for l in xrange(num_pose):
                    num1[l] += diff * grad_tilde_image[l][j]
                    num3[l] += sum_ * grad_tilde_image[l][j]

            energy += num2 / denom

            for l in xrange(num_pose):
                grad_e_pose[l] += (2 * num1[l] / denom - 2 * num2 * num3[l] / denom / denom)

        max_r = 0
        for i in xrange(size):
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

