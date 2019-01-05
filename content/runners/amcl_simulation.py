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

ExperimentConfig = namedtuple('ExperimentConfig', ['odometry_noise_mu_x', 'odometry_noise_mu_y',
                                                   'odometry_noise_sigma_x', 'odometry_noise_sigma_y',
                                                   'scan_noise_sigma', 'min_amcl_particles'])


def odometry_drift_x_configs_factory():
    return [ExperimentConfig(odometry_noise_mu_x=mu_x,
                             odometry_noise_mu_y=0,
                             odometry_noise_sigma_x=0,
                             odometry_noise_sigma_y=0,
                             scan_noise_sigma=0,
                             min_amcl_particles=2500) for mu_x in np.logspace(start=0, stop=1, num=10, base=0.001)]


def odometry_drift_xy_configs_factory():
    return [ExperimentConfig(odometry_noise_mu_x=mu,
                             odometry_noise_mu_y=mu,
                             odometry_noise_sigma_x=0,
                             odometry_noise_sigma_y=0,
                             scan_noise_sigma=0,
                             min_amcl_particles=2500) for mu in np.logspace(start=0, stop=1, num=10, base=0.001)]


def odometry_skid_x_configs_factory():
    return [ExperimentConfig(odometry_noise_mu_x=0,
                             odometry_noise_mu_y=0,
                             odometry_noise_sigma_x=sigma_x,
                             odometry_noise_sigma_y=0,
                             scan_noise_sigma=0,
                             min_amcl_particles=2500) for sigma_x in np.logspace(start=0, stop=1, num=10, base=0.01)]


def odometry_skid_xy_configs_factory():
    return [ExperimentConfig(odometry_noise_mu_x=0,
                             odometry_noise_mu_y=0,
                             odometry_noise_sigma_x=sigma,
                             odometry_noise_sigma_y=sigma,
                             scan_noise_sigma=0,
                             min_amcl_particles=2500) for sigma in np.logspace(start=0, stop=1, num=10, base=0.01)]


def scan_noise_configs_factory():
    return [ExperimentConfig(odometry_noise_mu_x=0,
                             odometry_noise_mu_y=0,
                             odometry_noise_sigma_x=0,
                             odometry_noise_sigma_y=0,
                             scan_noise_sigma=sigma,
                             min_amcl_particles=2500) for sigma in np.logspace(start=0, stop=1, num=10, base=0.01)]


def min_amcl_particles_configs_factory():
    return [ExperimentConfig(odometry_noise_mu_x=0,
                             odometry_noise_mu_y=0,
                             odometry_noise_sigma_x=0,
                             odometry_noise_sigma_y=0,
                             scan_noise_sigma=0,
                             min_amcl_particles=int(particles)) for particles in np.linspace(start=100, stop=5000, num=10)]

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
description = 'amcl_simulation_graphs_check'
repetitions = 3
two_snapshot = False
experiment_configs_list = [ExperimentConfig(odometry_noise_mu_x=0,
                                            odometry_noise_mu_y=0,
                                            odometry_noise_sigma_x=0,
                                            odometry_noise_sigma_y=0,
                                            scan_noise_sigma=0,
                                            min_amcl_particles=2500)]
only_one = True
setup = 'apr' # apr / nov1 / nov2 / nov3 / nov4
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_april_18 import orchard_topology
elif setup == 'nov1':
    raise NotImplementedError # TODO: implement
elif setup == 'nov2':
    raise NotImplementedError # TODO: implement
elif setup == 'nov3':
    raise NotImplementedError # TODO: implement
elif setup == 'nov4':
    raise NotImplementedError # TODO: implement


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder(description)

    for td_experiment in combinations(selected_td_experiments, r=2 if two_snapshot else 1):
        td_experiment_name_for_map = td_experiment[0]
        if len(td_experiment) == 2:
            td_experiment_name_for_localization = td_experiment[1]
        else:
            td_experiment_name_for_localization = td_experiment_name_for_map
        with open (os.path.join(td_results_dir, td_experiment_name_for_map, 'experiment_summary.json')) as f:
            td_summary_for_map = json.load(f)
        with open (os.path.join(td_results_dir, td_experiment_name_for_localization, 'experiment_summary.json')) as f:
            td_summary_for_localization = json.load(f)

        optimized_sigma = td_summary_for_localization['results']['0']['optimized_sigma']
        optimized_grid_dim_x_for_map = td_summary_for_map['results']['0']['optimized_grid_dim_x']
        optimized_grid_dim_y_for_map = td_summary_for_map['results']['0']['optimized_grid_dim_y']
        optimized_grid_dim_x_for_localization = td_summary_for_map['results']['0']['optimized_grid_dim_x']
        optimized_grid_dim_y_for_localization = td_summary_for_map['results']['0']['optimized_grid_dim_y']
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
        map_semantic_trunks = td_summary_for_map['results']['0']['semantic_trunks']
        localization_image_path = td_summary_for_localization['data_sources']
        localization_image_key = td_summary_for_localization['metadata']['image_key']
        localization_semantic_trunks = td_summary_for_localization['results']['0']['semantic_trunks']

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
                                                                'localization_altitude': td_summary_for_localization['metadata']['altitude'],
                                                                'trajectory_name': trajectory_name},
                                                      working_dir=execution_dir)
                experiment.run(repetitions, launch_rviz=True)

        if only_one:
            break