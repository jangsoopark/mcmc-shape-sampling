
import numpy as np
import sys


def scale_level_set_function(phi, factor, size):

    for i in range(size):
        phi[i] *= factor

    return phi


def tp(psi, p, filling, x_coord, y_coord, size, size_i, size_j):

    a = p[0]
    b = p[1]
    theta = p[2]
    h = p[3]

    ic = (size_i + 1) >> 1
    jc = (size_j + 1) >> 1

    new_psi = np.zeros( size)
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
    return scale_level_set_function(new_psi, p[3], size)


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
            ay[k] = (a[k + size_i] -a[k - size_i]) / 2

    return a, ax, ay


def update_pose(training_image, psi, num_shapes, pose, r, i_coord, j_coord, x_coord, y_coord, size, size_i, size_j, num_of_classes):

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
                grad_e_pose[l] += (2 * num1[l] / denom - 2 * num2 * num3[l] / denom / denom)

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



def evolve_with_data_term(image, psi,
                          narrow_band,
                          training_image_matrix, training_phi_matrix,
                          pose_for_each_class_vector,
                          num_of_classes, num_of_shapes_in_each_class,
                          dt,
                          current_iteration, last_iteration, kappa, size_i, size_j):
    
    size = size_i * size_j
    
    temp_training_image = np.zeros((num_of_shapes_in_each_class, size))
    psi_aligned = np.zeros((num_of_classes, size))

    num_of_all_training_shapes = num_of_classes * num_of_shapes_in_each_class

    training_image = training_image_matrix.reshape((num_of_all_training_shapes, size))
    training_phi = training_phi_matrix.reshape((num_of_all_training_shapes, size))
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
                        r, i_coord, j_coord, x_coord, y_coord, size, size_i, size_j, num_of_classes)
            psi_aligned = tp_sdf(psi, pose_for_each_class[j], 0.0, x_coord, y_coord, size, size_i, size_j)

            for k in range(4):
                pose_for_each_class_vector[j * 4 + k] = pose_for_each_class[j][k]

    return psi, pose_for_each_class
