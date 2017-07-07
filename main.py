from scipy.io import loadmat

import matplotlib.pyplot as plt

import numpy as np

from initial_level_set import *
from generate_level_set import *
from create_narrow_band import *
from curvature import *
from evaluate_energy_with_data_term import *

from evolve_with_data_term import evolve_with_data_term
from mcmc_shape_sampling import mcmc_shape_sampling
from evaluate_energy_with_shape_prior import evaluate_energy_with_shape_prior

image_id = '01'
dataset_name = 'aircraft'

test_image_path = './%s/testImages/testImage%s' % (dataset_name, image_id)

if dataset_name == 'aircraft':
    training_image_path = './%s/trainingShapes/trainingSet%s' % (dataset_name, image_id)
else:
    training_image_path = './%s/trainingShapes/trainingSet' % (dataset_name)

test_image_file = loadmat(test_image_path)
training_image_file = loadmat(training_image_path)

is_occluded_region_known = False
occluded_region = np.zeros(test_image_file['testImage'].shape)

if is_occluded_region_known:
    occluded_region[4:9, 4:9] = 1

# provide initial curve
plt.imshow(test_image_file['testImage'], cmap='gray')

size_i, size_j = training_image_file['sz_i'].item(), training_image_file['sz_j'].item()
psi = initial_level_set(5, size_i, size_j)
plt.contour(psi, levels=[1], colors='r')
plt.show()

training_matrix = training_image_file['AlignedShapeMatrix']
num_of_shapes_in_training_set = training_matrix.shape[1]
training_phi_matrix = np.zeros((size_i * size_j, num_of_shapes_in_training_set))

# construct level set representation of shapes in training set
for i in range(num_of_shapes_in_training_set):
    current_shape = np.double(training_matrix[:, i].reshape((size_i, size_j)) > 0)
    dummy = generate_level_set(-2 * current_shape + 1)
    dummy = dummy.T

    training_phi_matrix[:, i] = dummy[:].flatten()

# curve evolution with data term
num_of_iterations = 20
num_of_class_conf = {'aircraft': 1, 'MNIST': 10}
num_of_classes = num_of_class_conf[dataset_name]
num_of_shapes_in_each_class = 10

pose_for_each_class = np.zeros((4, num_of_classes))
pose_for_each_class[3, :] = 1
dt = 0.2  # gradient step size

for i in range(num_of_iterations):
    print('iters %d' % i)
    narrow_band = create_narrow_band(psi, 5)

    kappa = curvature(test_image_file['testImage'])
    psi, pose_for_each_class = evolve_with_data_term(test_image_file['testImage'].flatten(),
                                                     psi.flatten(),
                                                     narrow_band.flatten(),
                                                     training_matrix.flatten(),
                                                     training_phi_matrix.flatten(),
                                                     pose_for_each_class.flatten(),
                                                     num_of_classes, num_of_shapes_in_each_class,
                                                     dt,
                                                     i + 1, num_of_iterations, kappa.flatten(),
                                                     size_i, size_j)

    psi = psi.reshape((size_i, size_j))
    pose_for_each_class = pose_for_each_class.reshape((4, num_of_classes))

plt.imshow(test_image_file['testImage'], cmap='gray')
plt.contour(psi, levels=[0], colors='r')
plt.hold(True)
plt.show()

# MCMC shape sampling
num_of_samples = 500
num_of_sampling_iterations = 20
num_of_iteration_for_single_pertubation = num_of_class_conf[dataset_name]

gamma = 1
dt = 0.2
alpha = 5
beta = 1

for i in range(num_of_samples):

    current_curve = psi
    previous_selected_shape_id = 0
    accepted_count = 0
    pose = np.zeros(4)
    pose[3] = 360 / 360

    for j in range(num_of_sampling_iterations):
        psi_candidate = current_curve[:, :]

        mh_threshold = np.random.uniform()
        random_number_for_class_selection = np.random.uniform()

        random_number_array = np.random.uniform(size=gamma)
        p_forward = 0
        p_reverse = 0

        if j == 0:
            current_selected_class_id = 0
        current_selected_shape_id = np.zeros(gamma, np.int32)

        for k in range(num_of_iteration_for_single_pertubation):
            narrow_band = create_narrow_band(psi_candidate, 5)
            psi_candidate, p_forward, p_reverse, current_selected_class_id, current_selected_shape_id = mcmc_shape_sampling(
                test_image_file['testImage'].flatten(),
                psi_candidate.flatten(),
                narrow_band.flatten(),
                training_phi_matrix.flatten(),
                pose_for_each_class.flatten(),
                num_of_classes,
                num_of_shapes_in_each_class,
                dt, alpha, j,
                random_number_for_class_selection,
                gamma,
                random_number_array,
                p_forward, p_reverse,
                current_selected_class_id,
                current_selected_shape_id,
                previous_selected_shape_id,
                accepted_count, pose,
                k, beta, size_i, size_j)

            psi_candidate = psi_candidate.reshape((size_i, size_j))

            # plt.imshow(test_image_file['testImage'], cmap='gray')
            # plt.contour(narrow_band, levels=[0], colors='r')
            # plt.show()
        p_of_candidate = evaluate_energy_with_shape_prior(
            psi_candidate.flatten(), training_phi_matrix.flatten(),
            num_of_classes, num_of_shapes_in_each_class,
            pose, current_selected_class_id, size_i, size_j)

        p_of_current = evaluate_energy_with_shape_prior(
            current_curve.flatten(), training_phi_matrix.flatten(),
            num_of_classes, num_of_shapes_in_each_class,
            pose, current_selected_class_id, size_i, size_j)

        minus_log_of_data_candidate = evaluate_energy_with_data_term(
            test_image_file['testImage'], psi_candidate, occluded_region)
        minus_log_of_data_current = evaluate_energy_with_data_term(
            test_image_file['testImage'], current_curve, occluded_region)

        energy_candidate = alpha * -np.log(p_of_candidate)
        energy_current = alpha * -np.log(p_of_current)

        pi_of_candidate = np.exp(-energy_candidate)
        pi_of_current = np.exp(-energy_current)

        hasting_ratio = (pi_of_candidate * p_reverse) / (pi_of_current * p_forward)

        if mh_threshold < hasting_ratio or accepted_count == 0:
            current_curve = psi_candidate
            previous_selected_shape_id = current_selected_shape_id
            accepted_count += 1
        else:
            current_curve = current_curve
    plt.imshow(test_image_file['testImage'], cmap='gray')
    plt.contour(current_curve, levels=[0], colors='r')

    # plt.show()
    plt.savefig('./result/%03d_sample.png' % i)
    plt.close()
