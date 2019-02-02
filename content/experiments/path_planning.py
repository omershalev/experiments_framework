import numpy as np
import cv2
import os

from computer_vision import maps_generation
from computer_vision import segmentation
from computer_vision.astar_path_planner import AstarPathPlanner
from framework.experiment import Experiment
from framework import cv_utils


class PathPlanningExperiment(Experiment):

    def clean_env(self):
        pass

    def task(self, **kwargs):

        image = cv2.imread(self.data_sources['map_image_path'])
        waypoints = self.data_sources['waypoints']
        upper_left = self.data_sources['map_upper_left']
        lower_right = self.data_sources['map_lower_right']

        # Crop the image
        cropped_image = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        waypoints = (np.array(waypoints) - np.array(upper_left)).tolist()

        # Get cost map
        cost_map = maps_generation.generate_cost_map(cropped_image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'cost_map.jpg'), 255.0 * cost_map)

        # Plan a path
        path_planner = AstarPathPlanner(cost_map)
        trajectory = []
        for section_start, section_end in zip(waypoints[:-1], waypoints[1:]):
            trajectory += list(path_planner.astar(tuple(section_start), tuple(section_end)))

        # Save results
        self.results[self.repetition_id]['trajectory'] = trajectory

        trajectory_on_cost_map_image = cv2.cvtColor(np.uint8(255.0 * cost_map), cv2.COLOR_GRAY2BGR)
        trajectory_on_cost_map_image = cv_utils.draw_points_on_image(trajectory_on_cost_map_image, trajectory, color=(0, 255, 255), radius=5)
        cv2.imwrite(os.path.join(self.repetition_dir, 'trajectory_on_cost_map.jpg'), trajectory_on_cost_map_image)
        self.results[self.repetition_id]['trajectory_on_cost_map_path'] = os.path.join(self.repetition_dir, 'trajectory_on_cost_map.jpg')

        _, trajectory_on_mask_image = segmentation.extract_canopy_contours(cropped_image)
        trajectory_on_mask_image = cv2.cvtColor(trajectory_on_mask_image, cv2.COLOR_GRAY2BGR)
        trajectory_on_mask_image = cv_utils.draw_points_on_image(trajectory_on_mask_image, trajectory, color=(0, 255, 255), radius=5)
        cv2.imwrite(os.path.join(self.repetition_dir, 'trajectory_on_mask.jpg'), trajectory_on_mask_image)
        self.results[self.repetition_id]['trajectory_on_mask_path'] = os.path.join(self.repetition_dir, 'trajectory_on_mask.jpg')

        trajectory_on_image = cv_utils.draw_points_on_image(cropped_image, trajectory, color=(0, 255, 255), radius=5)
        cv2.imwrite(os.path.join(self.repetition_dir, 'trajectory_on_image.jpg'), trajectory_on_image)
        self.results[self.repetition_id]['trajectory_on_image_path'] = os.path.join(self.repetition_dir, 'trajectory_on_image.jpg')
