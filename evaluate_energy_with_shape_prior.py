import numpy as np

g_kernel = lambda a, b: 1 / np.sqrt(np.pi) / b * np.exp(-(a ** 2) / 2 / b / b)
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


def compute_shape_energy(phi, pose, training_phi, num_shapes, k_size,
                         current_class_id, num_shapes_in_each_class, x_coord, y_coord, size, size_i, size_j):
    filling = 0.0
    shape_force = np.zeros(num_shapes)

    tilde_phi, domain = tp_sdf(phi, pose, filling, x_coord, y_coord, size, size_i, size_j)

    p_of_phi = 0

    for i in range(num_shapes):
        dist = compute_distance_template(tilde_phi, training_phi[:, i], size)
        weight = g_kernel(dist, k_size)

        p_of_phi += weight
        shape_force[i] = p_of_phi

    for i in range(num_shapes):
        shape_force[i] = shape_force[i] / shape_force[num_shapes - 1]

    p_of_phi = 0
    for i in range(current_class_id * num_shapes_in_each_class, (current_class_id + 1) * num_shapes_in_each_class):

        if i == 0:
            p_of_phi += shape_force[i]
        else:
            p_of_phi += (shape_force[i] - shape_force[i - 1])

    p_of_phi = p_of_phi / num_shapes_in_each_class
    return p_of_phi


def evaluate_energy_with_shape_prior(psi,
                                     training_phi_matrix,
                                     num_of_classes, num_of_shapes_in_each_class,
                                     pose, selected_class_id, size_i, size_j):
    size = size_i * size_j

    num_of_all_training_shapes = num_of_classes * num_of_shapes_in_each_class
    x_coord, y_coord, i_coord, j_coord, universe, r = compute_xyij_coordinate_and_universe_and_r(size, size_i, size_j)

    training_phi = training_phi_matrix.reshape((size, num_of_all_training_shapes))

    kernel_size = shape_kernel_size(training_phi, num_of_all_training_shapes, size)

    energy = compute_shape_energy(
        psi, pose, training_phi, num_of_all_training_shapes, kernel_size,
        selected_class_id, num_of_shapes_in_each_class, x_coord, y_coord, size, size_i, size_j)

    return energy
