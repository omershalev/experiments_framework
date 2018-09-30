import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import combinations

from computer_vision import calibration
from framework import utils
from framework import config
from content.experiments.amcl_simulation import AmclSimulationExperiment
from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments_and_repetitions as selected_td_experiments_and_repetitions
from content.data_pointers.lavi_april_18 import orchard_topology

ExperimentConfig = namedtuple('ExperimentConfig', ['odometry_noise_mu_x', 'odometry_noise_mu_y', 'odometry_noise_sigma_x', 'odometry_noise_sigma_y',
                                                   'scan_noise_sigma', 'min_amcl_particles'])

############################################################################################################################
#                                                    CONFIGURATION AREA                                                    #
############################################################################################################################
repetitions = 2
result_samples_num = 20
two_snapshot = True
experiment_configs_list = [ExperimentConfig(odometry_noise_mu_x=0.001, odometry_noise_mu_y=0,
                                            odometry_noise_sigma_x=0.01, odometry_noise_sigma_y=0.01, scan_noise_sigma=0.1, min_amcl_particles=2000)]
# experiment_configs_list = [ExperimentConfig(odometry_noise_mu_x=0, odometry_noise_mu_y=0, odometry_noise_sigma_x=0, odometry_noise_sigma_y=0,
#                                             scan_noise_sigma=0, min_amcl_particles=3000)]
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


def plot_canopies_vs_trunks(plot_name, canopies_vector, trunks_vector, canopies_stds, trunks_stds, output_dir):
    plt.figure()
    plt.errorbar(range(1, result_samples_num + 1), canopies_vector, yerr=canopies_stds, color='g')
    plt.errorbar(range(1, result_samples_num + 1), trunks_vector, yerr=trunks_stds, color='r')
    plt.xlim((0.8, result_samples_num + 0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '%s.png' % plot_name))


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_simulation')

    for td_experiments_and_repetitions in combinations(selected_td_experiments_and_repetitions, r=2 if two_snapshot else 1):
        td_experiment_name_for_map, td_repetition_for_map = td_experiments_and_repetitions[0]
        if len(td_experiments_and_repetitions) == 2:
            td_experiment_name_for_localization, td_repetition_for_localization = td_experiments_and_repetitions[1]
        else:
            td_experiment_name_for_localization, td_repetition_for_localization = td_experiment_name_for_map, td_repetition_for_map
        with open (os.path.join(td_results_dir, td_experiment_name_for_map, 'experiment_summary.json')) as f:
            td_summary_for_map = json.load(f)
        with open (os.path.join(td_results_dir, td_experiment_name_for_localization, 'experiment_summary.json')) as f:
            td_summary_for_localization = json.load(f)


        optimized_sigma = td_summary_for_localization['results'][str(td_repetition_for_map)]['optimized_sigma']
        optimized_grid_dim_x_for_map = td_summary_for_map['results'][str(td_repetition_for_map)]['optimized_grid_dim_x']
        optimized_grid_dim_y_for_map = td_summary_for_map['results'][str(td_repetition_for_map)]['optimized_grid_dim_y']
        optimized_grid_dim_x_for_localization = td_summary_for_map['results'][str(td_repetition_for_localization)]['optimized_grid_dim_x']
        optimized_grid_dim_y_for_localization = td_summary_for_map['results'][str(td_repetition_for_localization)]['optimized_grid_dim_y']
        mean_trunk_radius, std_trunk_radius = calibration.calculate_trunk_radius_in_meters(orchard_topology.measured_trunks_perimeters)
        pixel_to_meter_ratio_for_map = calibration.calculate_pixel_to_meter(optimized_grid_dim_x_for_map, optimized_grid_dim_y_for_map,
                                                                            orchard_topology.measured_row_widths, orchard_topology.measured_intra_row_distances)
        pixel_to_meter_ratio_for_localization = calibration.calculate_pixel_to_meter(optimized_grid_dim_x_for_localization, optimized_grid_dim_y_for_localization,
                                                                                     orchard_topology.measured_row_widths, orchard_topology.measured_intra_row_distances)
        mean_trunk_radius_in_pixels_for_map = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * pixel_to_meter_ratio_for_map))
        mean_trunk_radius_in_pixels_for_localization = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * pixel_to_meter_ratio_for_localization))
        std_trunk_radius_in_pixels_for_map = std_trunk_radius * config.trunk_std_increasing_factor * pixel_to_meter_ratio_for_map
        std_trunk_radius_in_pixels_for_localization = std_trunk_radius * config.trunk_std_increasing_factor * pixel_to_meter_ratio_for_localization
        map_image_path = td_summary_for_map['data_sources']
        map_image_key = td_summary_for_map['metadata']['image_key']
        map_semantic_trunks = td_summary_for_map['results'][str(td_repetition_for_map)]['semantic_trunks']
        localization_image_path = td_summary_for_localization['data_sources']
        localization_image_key = td_summary_for_localization['metadata']['image_key']
        localization_semantic_trunks = td_summary_for_localization['results'][str(td_repetition_for_localization)]['semantic_trunks']
        for experiment_config in experiment_configs_list:
            for trajectory_name in orchard_topology.trajectories.keys():
                experiment = AmclSimulationExperiment(name='amcl_snapshots_for_%s_trajectory_on_%s' %
                                                           (trajectory_name, (map_image_key if not two_snapshot else '%s_and_%s' % (map_image_key, localization_image_key))),
                                                      data_sources={'map_image_path': map_image_path, 'localization_image_path': localization_image_path,
                                                                    'map_semantic_trunks': map_semantic_trunks, 'localization_semantic_trunks': localization_semantic_trunks,
                                                                    'trajectory_waypoints': orchard_topology.trajectories[trajectory_name]},
                                                      params={'odometry_noise_mu_x': experiment_config.odometry_noise_mu_x,
                                                              'odometry_noise_mu_y': experiment_config.odometry_noise_mu_y,
                                                              'odometry_noise_sigma_x': experiment_config.odometry_noise_sigma_x,
                                                              'odometry_noise_sigma_y': experiment_config.odometry_noise_sigma_y,
                                                              'bounding_box_expand_ratio': config.bounding_box_expand_ratio,
                                                              'cost_map_gaussian_scale_factor': config.cost_map_gaussians_scale_factor,
                                                              'cost_map_canopy_sigma': optimized_sigma,
                                                              'min_angle': config.synthetic_scan_min_angle,
                                                              'max_angle': config.synthetic_scan_max_angle,
                                                              'samples_num': config.synthetic_scan_samples_num,
                                                              'min_distance': config.synthetic_scan_min_distance,
                                                              'max_distance': config.synthetic_scan_max_distance,
                                                              'map_resolution': 1.0 / pixel_to_meter_ratio_for_map,
                                                              'localization_resolution': 1.0 / pixel_to_meter_ratio_for_localization,
                                                              'r_primary_search_samples': config.synthetic_scan_r_primary_search_samples,
                                                              'r_secondary_search_step': config.synthetic_scan_r_secondary_search_step,
                                                              'mean_trunk_radius_for_map': mean_trunk_radius_in_pixels_for_map,
                                                              'std_trunk_radius_for_map': std_trunk_radius_in_pixels_for_map,
                                                              'mean_trunk_radius_for_localization': mean_trunk_radius_in_pixels_for_localization,
                                                              'std_trunk_radius_for_localization': std_trunk_radius_in_pixels_for_localization,
                                                              'scan_noise_sigma': experiment_config.scan_noise_sigma,
                                                              'min_amcl_particles': experiment_config.min_amcl_particles,
                                                              'target_frequency': config.target_system_frequency
                                                              },
                                                      metadata={'map_image_key': map_image_key, 'localization_image_key': localization_image_key,
                                                                'map_altitude': td_summary_for_map['metadata']['altitude'],
                                                                'localization_altitude': td_summary_for_localization['metadata']['altitude']},
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

                plot_canopies_vs_trunks('error', canopies_mean_errors, trunks_mean_errors, canopies_std_errors, trunks_std_errors, experiment.experiment_dir)
                plot_canopies_vs_trunks('covariance_norm', canopies_mean_covariance_norms, trunks_mean_covariance_norms, canopies_std_covariance_norms,
                                        trunks_std_covariance_norms, experiment.experiment_dir)

                # TODO: thoroughly think about results visualization + how to save them to allow flexibility with future plotting...