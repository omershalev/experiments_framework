import os
import json
import cv2
import numpy as np

from computer_vision import calibration
from framework import utils
from framework import cv_utils
from framework import config
from content.data_pointers.lavi_april_18 import orchard_topology
from content.experiments.path_planning import PathPlanningExperiment
from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir, selected_trunks_detection_experiments_and_repetitions

num_of_trajectories = 20

# TODO: play with hyper paremeters (sigma, radius, etc.) and show their influence on the chosen trajectories - those are parameters configurable by the farmer!!!

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('path_planning')
    trunks_detection_experiment_name, trunks_detection_repetition = selected_trunks_detection_experiments_and_repetitions[0]
    with open(os.path.join(trunks_detection_results_dir, trunks_detection_experiment_name, 'experiment_summary.json')) as f:
        trunks_detection_summary = json.load(f)
    image_path = trunks_detection_summary['data_sources']
    image_key = trunks_detection_summary['metadata']['image_key']
    optimized_sigma = trunks_detection_summary['results'][str(trunks_detection_repetition)]['optimized_sigma']
    optimized_grid_dim_x_in_pixels = trunks_detection_summary['results'][str(trunks_detection_repetition)]['optimized_grid_dim_x']
    optimized_grid_dim_y_in_pixels = trunks_detection_summary['results'][str(trunks_detection_repetition)]['optimized_grid_dim_y']
    meter_to_pixel_ratio = 38 # TODO: calculate this!!!!!!!!!!!
    mean_trunk_radius, _ = calibration.calculate_trunk_radius_in_meters(orchard_topology.measured_trunks_perimeters)
    # TODO: consider using trunk std radius for generating the map!!!!!
    mean_trunk_radius_in_pixels = int(np.round(mean_trunk_radius * config.trunk_dilation_ratio * meter_to_pixel_ratio))
    semantic_trunks = trunks_detection_summary['results'][str(trunks_detection_repetition)]['semantic_trunks']
    trunk_points_list = semantic_trunks.values()
    image = cv2.imread(image_path)
    upper_left, lower_right = cv_utils.get_bounding_box(image, trunk_points_list, expand_ratio=config.bounding_box_expand_ratio)
    orchard_pattern = orchard_topology.plot_pattern
    row_labels = np.unique([label.split('/')[0] for label in semantic_trunks.keys()])
    tree_labels = np.unique([label.split('/')[1] for label in semantic_trunks.keys()])
    i = 0
    while i < num_of_trajectories:
        start_label = '%s/%s' % (np.random.choice(row_labels), np.random.choice(tree_labels))
        goal_label = '%s/%s' % (np.random.choice(row_labels), np.random.choice(tree_labels))
        if start_label == goal_label:
            continue
        if start_label not in semantic_trunks.keys() or goal_label not in semantic_trunks.keys():
            continue
        i += 1
        start = tuple(np.array(semantic_trunks[start_label]) + np.array([mean_trunk_radius_in_pixels + 5, 0]))
        goal = tuple(np.array(semantic_trunks[goal_label]) + np.array([mean_trunk_radius_in_pixels + 5, 0]))
        waypoints = [start, goal]
        experiment = PathPlanningExperiment(name='path_planning_on_%s_from_%s_to_%s' % (image_key, start_label.replace('/', ''), goal_label.replace('/', '')), # TODO: remove hardcoded
                                            data_sources={'map_image_path': image_path, 'trunk_points_list': trunk_points_list,
                                                          'map_upper_left': upper_left, 'map_lower_right': lower_right, 'waypoints': waypoints},
                                            params={'trunk_radius': mean_trunk_radius_in_pixels, 'gaussian_scale_factor': config.cost_map_gaussians_scale_factor,
                                                    'canopy_sigma': optimized_sigma, 'bounding_box_expand_ratio': config.bounding_box_expand_ratio},
                                            working_dir=execution_dir, metadata=trunks_detection_summary['metadata'])
        experiment.run(repetitions=1)