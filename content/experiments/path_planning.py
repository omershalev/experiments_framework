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

        trunks = self.data_sources['trunk_points_list']
        trunk_radius = self.data_sources['trunk_radius']
        image = cv2.imread(self.data_sources['image_path'])
        canopy_sigma = self.data_sources['canopy_sigma']

        # Get cost map
        cost_map, map_origin = maps_generation.generate_cost_map(image, trunks, canopy_sigma, gaussian_scale_factor=0.7, # TODO: change hard coded values
                                                                 gaussian_square_size_to_sigma_ratio=3, gaussian_circle_radius_to_sigma_ratio=3, trunk_radius=trunk_radius)
        cv2.imwrite(os.path.join(self.repetition_dir, 'cost_map.jpg'), 255.0 * cost_map)

        # Plan a path
        path_planner = AstarPathPlanner(cost_map)
        start = (self.data_sources['start'][0] - map_origin[0], self.data_sources['start'][1] - map_origin[1])
        goal = (self.data_sources['goal'][0] - map_origin[0], self.data_sources['goal'][1] - map_origin[1])
        trajctory = path_planner.astar(start, goal)
        trajectory_image = cv_utils.draw_points_on_image(255.0 * cost_map, trajctory, color=255)
        cv2.imwrite(os.path.join(self.repetition_dir, 'trajectory.jpg'), trajectory_image)
