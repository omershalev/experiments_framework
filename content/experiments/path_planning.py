import numpy as np
import cv2
import os

from computer_vision import maps_generation
from computer_vision.astar_path_planner import AstarPathPlanner
from framework.experiment import Experiment
from framework import cv_utils


class PathPlanningExperiment(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):

        image = cv2.imread(self.data_sources['image_path'])
        trunk_points_list = self.data_sources['trunk_points_list']
        waypoints = self.data_sources['waypoints']
        trunk_radius = self.params['trunk_radius']
        canopy_sigma = self.params['canopy_sigma']
        gaussian_scale_factor = self.params['gaussian_scale_factor']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']

        # Crop the image around the trunks
        upper_left, lower_right = cv_utils.get_bounding_box(image, trunk_points_list, expand_ratio=bounding_box_expand_ratio)
        cropped_image = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        trunk_points_list = (np.array(trunk_points_list) - np.array(upper_left)).tolist()
        waypoints = (np.array(waypoints) - np.array(upper_left)).tolist()

        # Get cost map
        cost_map = maps_generation.generate_cost_map(cropped_image, trunk_points_list, canopy_sigma, gaussian_scale_factor,
                                                                 gaussian_square_size_to_sigma_ratio=3,
                                                                 gaussian_circle_radius_to_sigma_ratio=2.5,
                                                                 trunk_radius=trunk_radius)
        cv2.imwrite(os.path.join(self.repetition_dir, 'cost_map.jpg'), 255.0 * cost_map)

        # Plan a path
        path_planner = AstarPathPlanner(cost_map)
        trajctory = []
        for section_start, section_end in zip(waypoints[:-1], waypoints[1:]):
            trajctory += list(path_planner.astar(tuple(section_start), tuple(section_end)))
        self.results[self.repetition_id]['trajectory'] = trajctory
        trajectory_image = cv2.cvtColor(np.uint8(255.0 * cost_map), cv2.COLOR_GRAY2BGR)
        trajectory_image = cv_utils.draw_points_on_image(trajectory_image, trajctory, color=(255, 255, 0), radius=5)
        cv2.imwrite(os.path.join(self.repetition_dir, 'trajectory.jpg'), trajectory_image)
