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

        image = cv2.imread(self.data_sources['map_image_path'])
        trunk_points_list = self.data_sources['trunk_points_list']
        waypoints = self.data_sources['waypoints']
        upper_left = self.data_sources['map_upper_left']
        lower_right = self.data_sources['map_lower_right']

        # Crop the image
        cropped_image = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        trunk_points_list = (np.array(trunk_points_list) - np.array(upper_left)).tolist()
        waypoints = (np.array(waypoints) - np.array(upper_left)).tolist()

        # Get cost map
        cost_map = maps_generation.generate_cost_map(cropped_image)
        cv2.imwrite(os.path.join(self.repetition_dir, 'cost_map.jpg'), 255.0 * cost_map)

        # Plan a path
        path_planner = AstarPathPlanner(cost_map)
        trajctory = []
        for section_start, section_end in zip(waypoints[:-1], waypoints[1:]):
            trajctory += list(path_planner.astar(tuple(section_start), tuple(section_end)))
        self.results[self.repetition_id]['trajectory'] = trajctory
        trajectory_image = cv2.cvtColor(np.uint8(255.0 * cost_map), cv2.COLOR_GRAY2BGR)
        trajectory_image = cv_utils.draw_points_on_image(trajectory_image, trajctory, color=(255, 0, 255), radius=5)
        semantic_trunks = self.params['semantic_trunks']
        trajectory_image = cv_utils.draw_points_on_image(trajectory_image,
                                                         [np.array(semantic_trunks[trunk_label]) - np.array(upper_left) for trunk_label in semantic_trunks.keys()],
                                                         color=(0, 200, 0))
        for trunk_label in semantic_trunks.keys():
            cv2.putText(trajectory_image, trunk_label, tuple(np.array(semantic_trunks[trunk_label]) - np.array(upper_left) + np.array([-40, -70])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 0), thickness=8, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.repetition_dir, 'trajectory.jpg'), trajectory_image)
