import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

from computer_vision import calibration
from framework import utils
from framework import config
from content.experiments.amcl_simulation import AmclSimulationExperiment
from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir, selected_trunks_detection_experiments_and_repetitions
from content.data_pointers.lavi_april_18 import orchard_topology

ExperimentConfig = namedtuple('ExperimentConfig', ['odometry_noise_mu_x', 'odometry_noise_mu_y', 'odometry_noise_sigma_x', 'odometry_noise_sigma_y', 'scan_noise_sigma'])

############################################################################################################################
#                                                    CONFIGURATION AREA                                                    #
############################################################################################################################
repetitions = 2
result_samples_num = 20
experiment_configs_list = [ExperimentConfig(odometry_noise_mu_x=0.001, odometry_noise_mu_y=0,
                                            odometry_noise_sigma_x=0.01, odometry_noise_sigma_y=0.01, scan_noise_sigma=0.1)]
############################################################################################################################


def get_results_samples(experiment, namespace):
    joint_amcl_results_df = pd.DataFrame()
    for repetition_id in experiment.valid_repetitions:
        amcl_results_df = pd.read_csv(experiment.results[repetition_id]['%s_amcl_results_path' % namespace], index_col=0)
        joint_amcl_results_df = pd.concat([joint_amcl_results_df, amcl_results_df], axis=1)
    error_samples_df = pd.DataFrame()
    covariance_norm_samples_df = pd.DataFrame()
    delta_t = (joint_amcl_results_df.index[-1] - joint_amcl_results_df.index[0]) / result_samples_num
    search_timestamps = [joint_amcl_results_df.index[joint_amcl_results_df.index > delta_t * i][0] for i in
                         range(1, result_samples_num + 1)]
    for repetition_id in experiment.valid_repetitions:
        valid_repetition_indices = np.where(~joint_amcl_results_df['amcl_pose_error[%d]' % repetition_id].isnull())[0]
        this_repetition_errors = []
        this_repetition_covaraiance_norms = []
        for search_timestamp in search_timestamps:
            search_index = joint_amcl_results_df.index.get_loc(search_timestamp)

            def find_nearest(array, value):
                idx = (np.abs(array - value)).argmin()
                return array[idx]

            nearest_valid_index = find_nearest(valid_repetition_indices, search_index)
            this_repetition_errors.append(
                joint_amcl_results_df.iloc[nearest_valid_index]['amcl_pose_error[%d]' % repetition_id])
            this_repetition_covaraiance_norms.append(
                joint_amcl_results_df.iloc[nearest_valid_index]['amcl_covariance_norm[%d]' % repetition_id])
        error_samples_df = pd.concat([error_samples_df, pd.Series(this_repetition_errors)], axis=1)
        covariance_norm_samples_df = pd.concat(
            [covariance_norm_samples_df, pd.Series(this_repetition_covaraiance_norms, name=repetition_id)], axis=1)
    return error_samples_df, covariance_norm_samples_df


def plot_canopies_vs_trunks(plot_name, canopies_vector, trunks_vector, canopies_stds, trunks_stds):
    plt.figure()
    plt.errorbar(range(1, result_samples_num + 1), canopies_vector, yerr=canopies_stds, color='g')
    plt.errorbar(range(1, result_samples_num + 1), trunks_vector, yerr=trunks_stds, color='r')
    plt.xlim((0.8, result_samples_num + 0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(experiment.experiment_dir, '%s.png' % plot_name))


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_single_image')

    for trunks_detection_experiment_name, trunks_detection_repetition in selected_trunks_detection_experiments_and_repetitions:
        with open (os.path.join(trunks_detection_results_dir, trunks_detection_experiment_name, 'experiment_summary.json')) as f:
            trunks_detection_summary = json.load(f)

        optimized_sigma = trunks_detection_summary['results'][str(trunks_detection_repetition)]['optimized_sigma']
        optimized_grid_dim_x_in_pixels = trunks_detection_summary['results'][str(trunks_detection_repetition)]['optimized_grid_dim_x']
        optimized_grid_dim_y_in_pixels = trunks_detection_summary['results'][str(trunks_detection_repetition)]['optimized_grid_dim_y']
        mean_trunk_radius, std_trunk_radius = calibration.get_trunk_radius_in_meters(orchard_topology.measured_trunks_perimeters)
        meter_to_pixel_ratio = 38  # TODO: calculate this!!!!!!!!!!!
        mean_trunk_radius_in_pixels = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * meter_to_pixel_ratio))
        std_trunk_radius_in_pixels = std_trunk_radius * config.trunk_std_increasing_factor * meter_to_pixel_ratio
        image_path = trunks_detection_summary['data_sources']
        image_key = trunks_detection_summary['metadata']['image_key']
        semantic_trunks = trunks_detection_summary['results'][str(trunks_detection_repetition)]['semantic_trunks']
        experiment_config = experiment_configs_list[0] # TODO: change this!!!
        experiment = AmclSimulationExperiment(name='amcl_single_snapshot_on_%s' % image_key,
                                              data_sources={'localization_image_path': image_path, 'map_image_path': image_path, 'semantic_trunks': semantic_trunks},
                                              params={'odometry_source': 'synthetic',
                                                     'odometry_noise_mu_x': experiment_config.odometry_noise_mu_x,
                                                     'odometry_noise_mu_y': experiment_config.odometry_noise_mu_y,
                                                     'odometry_noise_sigma_x': experiment_config.odometry_noise_sigma_x,
                                                     'odometry_noise_sigma_y': experiment_config.odometry_noise_sigma_y,
                                                     'bounding_box_expand_ratio': config.bounding_box_expand_ratio,
                                                     'gaussian_scale_factor': config.cost_map_gaussians_scale_factor,
                                                     'min_angle': config.synthetic_scan_min_angle,
                                                     'max_angle': config.synthetic_scan_max_angle,
                                                     'samples_num': config.synthetic_scan_samples_num,
                                                     'min_distance': config.synthetic_scan_min_distance,
                                                     'max_distance': config.synthetic_scan_max_distance,
                                                     'resolution': config.top_view_resolution,
                                                     'r_primary_search_samples': config.synthetic_scan_r_primary_search_samples,
                                                     'r_secondary_search_step': config.synthetic_scan_r_secondary_search_step,
                                                     'mean_trunk_radius': mean_trunk_radius_in_pixels,
                                                     'std_trunk_radius': std_trunk_radius_in_pixels,
                                                     'scan_noise_sigma': experiment_config.scan_noise_sigma,
                                                      },
                                              metadata=trunks_detection_summary['metadata'],
                                              working_dir=execution_dir)
        experiment.run(repetitions, launch_rviz=True)

        # Graphs
        canopies_error_samples_df, canopies_covariance_norm_samples_df = get_results_samples(experiment, namespace='canopies')
        canopies_mean_errors = canopies_error_samples_df.mean(axis=1)
        canopies_std_errors = canopies_error_samples_df.std(axis=1)
        canopies_mean_covariance_norms = canopies_covariance_norm_samples_df.mean(axis=1)
        canopies_std_covariance_norms = canopies_covariance_norm_samples_df.std(axis=1)

        trunks_error_samples_df, trunks_covariance_norm_samples_df = get_results_samples(experiment, namespace='trunks')
        trunks_mean_errors = trunks_error_samples_df.mean(axis=1)
        trunks_std_errors = trunks_error_samples_df.std(axis=1)
        trunks_mean_covariance_norms = trunks_covariance_norm_samples_df.mean(axis=1)
        trunks_std_covariance_norms = trunks_covariance_norm_samples_df.std(axis=1)

        plot_canopies_vs_trunks('error', canopies_mean_errors, trunks_mean_errors, canopies_std_errors, trunks_std_errors)
        plot_canopies_vs_trunks('covariance_norm', canopies_mean_covariance_norms, trunks_mean_covariance_norms, canopies_std_covariance_norms, trunks_std_covariance_norms)