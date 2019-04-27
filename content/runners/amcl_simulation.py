import os
import json
import numpy as np
from collections import namedtuple
from itertools import combinations

from computer_vision import calibration
from framework import utils
from framework import config
from content.experiments.amcl_simulation import AmclSimulationExperiment


ExperimentConfig = namedtuple('ExperimentConfig', ['odometry_noise_mu_x', 'odometry_noise_mu_y',
                                                   'odometry_noise_sigma_x', 'odometry_noise_sigma_y',
                                                   'scan_noise_sigma', 'min_amcl_particles'])


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
description = 'amcl'
repetitions = 10
different_localization_and_mapping_sources = True
# experiment_configs_list = [
#     ExperimentConfig(odometry_noise_mu_x=0.001,
#                      odometry_noise_mu_y=0,
#                      odometry_noise_sigma_x=0.01,
#                      odometry_noise_sigma_y=0,
#                      scan_noise_sigma=0,
#                      min_amcl_particles=1000),
#     ExperimentConfig(odometry_noise_mu_x=0.002,
#                      odometry_noise_mu_y=0,
#                      odometry_noise_sigma_x=0.01,
#                      odometry_noise_sigma_y=0,
#                      scan_noise_sigma=0,
#                      min_amcl_particles=1000),
#     ExperimentConfig(odometry_noise_mu_x=0.003,
#                      odometry_noise_mu_y=0,
#                      odometry_noise_sigma_x=0.01,
#                      odometry_noise_sigma_y=0,
#                      scan_noise_sigma=0,
#                      min_amcl_particles=1000),
#     ExperimentConfig(odometry_noise_mu_x=0.004,
#                      odometry_noise_mu_y=0,
#                      odometry_noise_sigma_x=0.01,
#                      odometry_noise_sigma_y=0,
#                      scan_noise_sigma=0,
#                      min_amcl_particles=1000),
#     ExperimentConfig(odometry_noise_mu_x=0.005,
#                      odometry_noise_mu_y=0,
#                      odometry_noise_sigma_x=0.01,
#                      odometry_noise_sigma_y=0,
#                      scan_noise_sigma=0,
#                      min_amcl_particles=1000),
#                            ]
experiment_configs_list = [ExperimentConfig(odometry_noise_mu_x=0,
                                            odometry_noise_mu_y=0,
                                            odometry_noise_sigma_x=0,
                                            odometry_noise_sigma_y=0,
                                            scan_noise_sigma=0,
                                            min_amcl_particles=1000)]
# selected_experiments = [
#     ('15-08-1', '15-53-1', 's_patrol'),  # so so
#     ('15-08-1', '15-53-1', 'wide_row'),  # so so
#     ('15-08-1', '19-04-1', 's_patrol'),  # so so
#     # ('15-53-1', '16-55-1', 'narrow_row'),  # so so
#     ('15-53-1', '16-55-1', 'u_turns'),
#     ('15-53-1', '16-55-1', 'wide_row'),  # so so
#     # ('15-53-1', '19-04-1', 'narrow_row'),  # here the canopies fail
#     ('15-53-1', '19-04-1', 'tasks_and_interrupts'),  # so so
#     ('15-53-1', '19-04-1', 'u_turns'),  # so so
#     ('16-55-1', '19-04-1', 'narrow_row'),
#     ('16-55-1', '19-04-1', 's_patrol'),  # so so
#     ('16-55-1', '19-04-1', 'wide_row'),  # so so
# ]
# selected_experiments = [('15-53-1', '16-55-1', 's_patrol')]
# selected_experiments = [('16-55-1', '19-04-1', 'tasks_and_interrupts')]
selected_experiments = [('15-08-1', '19-04-1', 's_patrol')]
# selected_experiments = None
first_sample_only = False
first_trajectory_only = False
setup = 'apr' # apr / nov1 / nov2
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_april_18.orchard_topology import measured_trunks_perimeters
    from content.data_pointers.lavi_april_18.orchard_topology import measured_row_widths
    from content.data_pointers.lavi_april_18.orchard_topology import measured_intra_row_distances
    from content.data_pointers.lavi_april_18.orchard_topology import trajectories
elif setup == 'nov1':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_november_18.dji import plot1_selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_measured_trunks_perimeters as measured_trunks_perimeters
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_measured_row_widths as measured_row_widths
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_measured_intra_row_distances as measured_intra_row_distances
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_trajectories as trajectories
elif setup == 'nov2':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_november_18.dji import plot2_selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_measured_trunks_perimeters as measured_trunks_perimeters
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_measured_row_widths as measured_row_widths
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_measured_intra_row_distances as measured_intra_row_distances
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_trajectories as trajectories


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder(description)

    for td_experiment in combinations(selected_td_experiments, r=2 if different_localization_and_mapping_sources else 1):
        td_experiment_name_for_map = td_experiment[0]
        if len(td_experiment) == 2:
            td_experiment_name_for_localization = td_experiment[1]
        else:
            td_experiment_name_for_localization = td_experiment_name_for_map
        with open (os.path.join(td_results_dir, td_experiment_name_for_map, 'experiment_summary.json')) as f:
            td_summary_for_map = json.load(f)
        with open (os.path.join(td_results_dir, td_experiment_name_for_localization, 'experiment_summary.json')) as f:
            td_summary_for_localization = json.load(f)

        optimized_sigma = td_summary_for_localization['results']['1']['optimized_sigma']
        optimized_grid_dim_x_for_map = td_summary_for_map['results']['1']['optimized_grid_dim_x']
        optimized_grid_dim_y_for_map = td_summary_for_map['results']['1']['optimized_grid_dim_y']
        optimized_grid_dim_x_for_localization = td_summary_for_map['results']['1']['optimized_grid_dim_x']
        optimized_grid_dim_y_for_localization = td_summary_for_map['results']['1']['optimized_grid_dim_y']
        mean_trunk_radius, std_trunk_radius = calibration.calculate_trunk_radius_in_meters(measured_trunks_perimeters)
        pixel_to_meter_ratio_for_map = calibration.calculate_pixel_to_meter(optimized_grid_dim_x_for_map, optimized_grid_dim_y_for_map,
                                                                            measured_row_widths, measured_intra_row_distances)
        pixel_to_meter_ratio_for_localization = calibration.calculate_pixel_to_meter(optimized_grid_dim_x_for_localization, optimized_grid_dim_y_for_localization,
                                                                                     measured_row_widths, measured_intra_row_distances)
        mean_trunk_radius_in_pixels_for_map = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * pixel_to_meter_ratio_for_map))
        mean_trunk_radius_in_pixels_for_localization = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * pixel_to_meter_ratio_for_localization))
        std_trunk_radius_in_pixels_for_map = std_trunk_radius * config.trunk_std_increasing_factor * pixel_to_meter_ratio_for_map
        std_trunk_radius_in_pixels_for_localization = std_trunk_radius * config.trunk_std_increasing_factor * pixel_to_meter_ratio_for_localization
        map_image_path = td_summary_for_map['data_sources']
        map_image_key = td_summary_for_map['metadata']['image_key']
        map_semantic_trunks = td_summary_for_map['results']['1']['semantic_trunks']
        if os.path.exists(os.path.join(td_results_dir, td_experiment_name_for_map, 'external_trunks.json')):
            with open(os.path.join(td_results_dir, td_experiment_name_for_map, 'external_trunks.json')) as f:
                map_external_trunks = json.load(f)
        else:
            map_external_trunks = []
        localization_image_path = td_summary_for_localization['data_sources']
        localization_image_key = td_summary_for_localization['metadata']['image_key']
        localization_semantic_trunks = td_summary_for_localization['results']['1']['semantic_trunks']
        if os.path.exists(os.path.join(td_results_dir, td_experiment_name_for_localization, 'external_trunks.json')):
            with open(os.path.join(td_results_dir, td_experiment_name_for_localization, 'external_trunks.json')) as f:
                localization_external_trunks = json.load(f)
        else:
            localization_external_trunks = []
        for experiment_config in experiment_configs_list:
            for trajectory_name in trajectories.keys():
                if selected_experiments is not None:
                    if (map_image_key, localization_image_key, trajectory_name) not in selected_experiments:
                        continue
                experiment = AmclSimulationExperiment(name='amcl_snapshots_for_%s_trajectory_on_%s' %
                                                           (trajectory_name, (map_image_key if not different_localization_and_mapping_sources else '%s_and_%s' % (map_image_key, localization_image_key))),
                                                      data_sources={'map_image_path': map_image_path, 'localization_image_path': localization_image_path,
                                                                    'map_semantic_trunks': map_semantic_trunks, 'localization_semantic_trunks': localization_semantic_trunks,
                                                                    'map_external_trunks': map_external_trunks, 'localization_external_trunks': localization_external_trunks,
                                                                    'trajectory_waypoints': trajectories[trajectory_name]},
                                                      params={'odometry_noise_mu_x': experiment_config.odometry_noise_mu_x,
                                                              'odometry_noise_mu_y': experiment_config.odometry_noise_mu_y,
                                                              'odometry_noise_sigma_x': experiment_config.odometry_noise_sigma_x,
                                                              'odometry_noise_sigma_y': experiment_config.odometry_noise_sigma_y,
                                                              'bounding_box_expand_ratio': config.bounding_box_expand_ratio,
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
                                                                'localization_altitude': td_summary_for_localization['metadata']['altitude'],
                                                                'trajectory_name': trajectory_name},
                                                      working_dir=execution_dir)
                experiment.run(repetitions, launch_rviz=True)
                if first_trajectory_only:
                    break
        if first_sample_only:
            break