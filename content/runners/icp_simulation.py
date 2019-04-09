import os
import json
from collections import namedtuple
import numpy as np

from computer_vision import calibration
from framework import utils
from framework import config
from content.experiments.icp_simulation import IcpSimulationExperiment


ExperimentConfig = namedtuple('ExperimentConfig', ['scan_noise_sigma'])


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
description = 'apr_icp'
repetitions = 10
experiment_configs_list = [ExperimentConfig(scan_noise_sigma=0)]
first_sample_only = True
first_trajectory_only = True
# trajectories = {'fork':
#                 [('2/G', '3/G'), ('2/E', '3/E'), ('2/C', '3/C'),
#                 ('2/A', '3/A', 0, 150, 0, 150),
#                 ('4/A', '5/A', 0, 150, 0, 150),
#                 ('4/C', '5/C'), ('4/E', '5/E'), ('4/G', '5/G'),
#                 ('4/G', '5/G'), ('4/E', '5/E'), ('4/C', '5/C'),
#                 ('4/A', '5/A', 0, 150, 0, 150),
#                 ('6/A', '7/A', 0, 150, 0, 150),
#                 ('6/C', '7/C'), ('6/E', '7/E'), ('6/G', '7/G'),
#                 ('6/G', '7/G'), ('6/E', '7/E'), ('6/C', '7/C'),
#                 ('6/A', '7/A', 0, 150, 0, 150),
#                 ('8/A', '9/A', 0, 150, 0, 150),
#                 ('8/C', '9/C'), ('8/E', '9/E'), ('8/G', '9/G')]
#                 }
trajectories = {'fork':
                [('3/G', '4/G'), ('3/E', '4/E'), ('3/C', '4/C'),
                ('3/A', '4/A', 0, 150, 0, 150),
                ('4/A', '5/A', 0, 150, 0, 150),
                ('4/C', '5/C'), ('4/E', '5/E'), ('4/G', '5/G'),
                ('4/G', '5/G'), ('4/E', '5/E'), ('4/C', '5/C'),
                ('4/A', '5/A', 0, 150, 0, 150),
                ('6/A', '7/A', 0, 150, 0, 150),
                ('6/C', '7/C'), ('6/E', '7/E'), ('6/G', '7/G'),
                ('6/G', '7/G'), ('6/E', '7/E'), ('6/C', '7/C'),
                ('6/A', '7/A', 0, 150, 0, 150),
                ('7/A', '8/A', 0, 150, 0, 150),
                ('7/C', '8/C'), ('7/E', '8/E'), ('7/G', '8/G')]
                }
# trajectories = {'fork':
#                 [('3/G', '4/G'), ('3/E', '4/E'), ('3/C', '4/C'),
#                 ('3/A', '4/A', 0, 150, 0, 150),
#                 ('5/A', '6/A', 0, 150, 0, 150),
#                 ('5/C', '6/C'), ('5/E', '6/E'), ('5/G', '6/G'),
#                 ('5/G', '6/G'), ('5/E', '6/E'), ('5/C', '6/C'),
#                 ('5/A', '6/A', 0, 150, 0, 150),
#                 ('7/A', '8/A', 0, 150, 0, 150),
#                 ('7/C', '8/C'), ('7/E', '8/E'), ('7/G', '8/G'),
#                 ('7/G', '8/G'), ('7/E', '8/E'), ('7/C', '8/C'),
#                 ('7/A', '8/A', 0, 150, 0, 150),
#                 ('9/A', '10/A', 0, 150, 0, 150),
#                 ('9/C', '10/C'), ('9/E', '10/E'), ('9/G', '10/G')]
#                 }
setup = 'apr' # apr / nov1 / nov2
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_april_18.orchard_topology import measured_trunks_perimeters
    from content.data_pointers.lavi_april_18.orchard_topology import measured_row_widths
    from content.data_pointers.lavi_april_18.orchard_topology import measured_intra_row_distances
    if trajectories is None:
        from content.data_pointers.lavi_april_18.orchard_topology import trajectories
elif setup == 'nov1':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_november_18.dji import plot1_selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_measured_trunks_perimeters as measured_trunks_perimeters
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_measured_row_widths as measured_row_widths
    from content.data_pointers.lavi_november_18.orchard_topology import plot1_measured_intra_row_distances as measured_intra_row_distances
    if trajectories is None:
        from content.data_pointers.lavi_november_18.orchard_topology import plot1_trajectories as trajectories
elif setup == 'nov2':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_november_18.dji import plot2_selected_trunks_detection_experiments as selected_td_experiments
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_measured_trunks_perimeters as measured_trunks_perimeters
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_measured_row_widths as measured_row_widths
    from content.data_pointers.lavi_november_18.orchard_topology import plot2_measured_intra_row_distances as measured_intra_row_distances
    if trajectories is None:
        from content.data_pointers.lavi_november_18.orchard_topology import plot2_trajectories as trajectories


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder(description)

    for td_experiment in selected_td_experiments:
        with open (os.path.join(td_results_dir, td_experiment, 'experiment_summary.json')) as f:
            td_summary_for_localization = json.load(f)
        optimized_grid_dim_x_for_localization = td_summary_for_localization['results']['1']['optimized_grid_dim_x']
        optimized_grid_dim_y_for_localization = td_summary_for_localization['results']['1']['optimized_grid_dim_y']
        mean_trunk_radius, std_trunk_radius = calibration.calculate_trunk_radius_in_meters(measured_trunks_perimeters)
        pixel_to_meter_ratio_for_localization = calibration.calculate_pixel_to_meter(optimized_grid_dim_x_for_localization, optimized_grid_dim_y_for_localization,
                                                                                     measured_row_widths, measured_intra_row_distances)
        mean_trunk_radius_in_pixels_for_localization = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * pixel_to_meter_ratio_for_localization))
        std_trunk_radius_in_pixels_for_localization = std_trunk_radius * config.trunk_std_increasing_factor * pixel_to_meter_ratio_for_localization
        pixel_to_meter_ratio_for_localization = calibration.calculate_pixel_to_meter(optimized_grid_dim_x_for_localization, optimized_grid_dim_y_for_localization,
                                                                                     measured_row_widths, measured_intra_row_distances)
        localization_image_path = td_summary_for_localization['data_sources']
        localization_image_key = td_summary_for_localization['metadata']['image_key']
        localization_semantic_trunks = td_summary_for_localization['results']['1']['semantic_trunks']
        if os.path.exists(os.path.join(td_results_dir, td_experiment, 'external_trunks.json')):
            with open(os.path.join(td_results_dir, td_experiment, 'external_trunks.json')) as f:
                localization_external_trunks = json.load(f)
        else:
            localization_external_trunks = []

        for experiment_config in experiment_configs_list:
            for trajectory_name in trajectories.keys():
                experiment = IcpSimulationExperiment(name='icp_snapshots_for_%s_trajectory_on_%s' % (trajectory_name, localization_image_key),
                                                      data_sources={'localization_image_path': localization_image_path,
                                                                    'localization_semantic_trunks': localization_semantic_trunks,
                                                                    'trajectory_waypoints': trajectories[trajectory_name],
                                                                    'localization_external_trunks': localization_external_trunks},
                                                      params={'bounding_box_expand_ratio': config.bounding_box_expand_ratio,
                                                              'mean_trunk_radius_for_localization': mean_trunk_radius_in_pixels_for_localization,
                                                              'std_trunk_radius_for_localization': std_trunk_radius_in_pixels_for_localization,
                                                              'min_angle': config.synthetic_scan_min_angle,
                                                              'max_angle': config.synthetic_scan_max_angle,
                                                              'samples_num': config.synthetic_scan_samples_num,
                                                              'min_distance': config.synthetic_scan_min_distance,
                                                              'max_distance': config.synthetic_scan_max_distance,
                                                              'localization_resolution': 1.0 / pixel_to_meter_ratio_for_localization,
                                                              'r_primary_search_samples': config.synthetic_scan_r_primary_search_samples,
                                                              'r_secondary_search_step': config.synthetic_scan_r_secondary_search_step,
                                                              'scan_noise_sigma': experiment_config.scan_noise_sigma,
                                                              'target_frequency': config.target_system_frequency},
                                                      metadata={'localization_image_key': localization_image_key,
                                                                'localization_altitude': td_summary_for_localization['metadata']['altitude'],
                                                                'trajectory_name': trajectory_name},
                                                      working_dir=execution_dir)
                experiment.run(repetitions, launch_rviz=True)
                if first_trajectory_only:
                    break
        if first_sample_only:
            break