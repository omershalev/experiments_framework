import os
import json
import cv2
import numpy as np

from framework import utils
from framework import cv_utils
from framework import config
from content.experiments.path_planning import PathPlanningExperiment



#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
td_baseline_experiment_name = 'trunks_detection_on_apr_15-08-1'
td_obstacle_1_experiment_name = 'trunks_detection_on_apr_15-19-1'
td_obstacle_2_experiment_name = 'trunks_detection_on_apr_15-17-1'
obstacle_1_location = (2743, 1320)
obstacle_2_location = (2181, 1770)
trajectory_waypoints = [('8/A', '9/A'),
                        ('9/C', '10/C'),
                        ('9/E', '10/E'),
                        ('8/I', '9/I'),
                        ('7/C', '8/C'),
                        ('7/A', '8/A'),
                        ('2/A', '3/A'),
                        ('2/F', '3/F'),
                        ('4/F', '5/F'),
                        ('6/A', '7/A')]
obstacle_size = [80, 80]
#################################################################################################


def get_trajectory(td_summary, image_path, start_waypoint_idx, end_waypoint_idx=None):
    semantic_trunks = td_summary['results']['1']['semantic_trunks']
    trunk_points_list = semantic_trunks.values()
    image = cv2.imread(image_path)
    upper_left, lower_right = cv_utils.get_bounding_box(image, trunk_points_list, expand_ratio=config.bounding_box_expand_ratio)
    waypoints_coordinates = []
    for waypoint in trajectory_waypoints[start_waypoint_idx:end_waypoint_idx]:
        point1 = semantic_trunks[waypoint[0]]
        point2 = semantic_trunks[waypoint[1]]
        waypoints_coordinates.append(((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2))
    experiment = PathPlanningExperiment(name='path_planning',
                                        data_sources={'map_image_path': image_path,
                                                      'map_upper_left': upper_left, 'map_lower_right': lower_right,
                                                      'waypoints': waypoints_coordinates},
                                        working_dir=execution_dir, metadata=td_summary['metadata'])
    experiment.run(repetitions=1)
    trajectory = experiment.results[1]['trajectory']
    return trajectory, waypoints_coordinates


def plot_trajectory(td_summary, image_path, trajectory, waypoints_coordinates, start_waypoint_idx, output_file_path):
    semantic_trunks = td_summary['results']['1']['semantic_trunks']
    trunk_points_list = semantic_trunks.values()
    image = cv2.imread(image_path)
    upper_left, lower_right = cv_utils.get_bounding_box(image, trunk_points_list, expand_ratio=config.bounding_box_expand_ratio)
    cropped_image = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
    trajectory_image = cv_utils.draw_points_on_image(cropped_image, trajectory, color=(0, 255, 255), radius=5)
    trajectory_image = cv_utils.draw_points_on_image(trajectory_image,
                                                     [tuple(np.array(coordinates) - np.array(upper_left)) for coordinates in waypoints_coordinates],
                                                     color=(0, 255, 255), radius=30)
    label_idx = start_waypoint_idx
    for coordinates in waypoints_coordinates:
        trajectory_image = cv_utils.put_shaded_text_on_image(trajectory_image, label=chr(label_idx + 65),
                                                             location=tuple(np.array(coordinates) - np.array(upper_left)),
                                                             color=(0, 255, 255), offset=(-40, -70))
        label_idx += 1
    cv2.imwrite(os.path.join(output_file_path), trajectory_image)


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('trajectory_update')

    with open(os.path.join(td_results_dir, td_baseline_experiment_name, 'experiment_summary.json')) as f:
        td_baseline_summary = json.load(f)
    with open(os.path.join(td_results_dir, td_obstacle_1_experiment_name, 'experiment_summary.json')) as f:
        td_obstacle_1_summary = json.load(f)
    with open(os.path.join(td_results_dir, td_obstacle_2_experiment_name, 'experiment_summary.json')) as f:
        td_obstacle_2_summary = json.load(f)

    baseline_image_path = td_baseline_summary['data_sources']
    obstacle_1_image_path = td_obstacle_1_summary['data_sources']
    obstacle_2_image_path = td_obstacle_2_summary['data_sources']

    trajectory, waypoints_coordinates = get_trajectory(td_baseline_summary, baseline_image_path, start_waypoint_idx=0)
    plot_trajectory(td_baseline_summary, baseline_image_path, trajectory, waypoints_coordinates, start_waypoint_idx=0,
                    output_file_path=os.path.join(execution_dir, 'no_obstacle_trajectory.jpg'))

    obstacle_1_image = cv2.imread(obstacle_1_image_path)
    cv2.rectangle(obstacle_1_image, tuple(np.array(obstacle_1_location) - np.array(obstacle_size)),
                  tuple(np.array(obstacle_1_location) + np.array(obstacle_size)), color=(0, 200, 0), thickness=-1)
    cv2.imwrite(os.path.join(execution_dir, 'synthetic_green_obstacle_1_image.jpg'), obstacle_1_image)
    trajectory, waypoints_coordinates = get_trajectory(td_obstacle_1_summary, os.path.join(execution_dir, 'synthetic_green_obstacle_1_image.jpg'), start_waypoint_idx=2)
    plot_trajectory(td_obstacle_1_summary, obstacle_1_image_path, trajectory, waypoints_coordinates, start_waypoint_idx=2,
                    output_file_path=os.path.join(execution_dir, 'obstacle_1_trajectory.jpg'))

    obstacle_2_image = cv2.imread(obstacle_2_image_path)
    cv2.rectangle(obstacle_2_image, tuple(np.array(obstacle_2_location) - np.array(obstacle_size)),
                  tuple(np.array(obstacle_2_location) + np.array(obstacle_size)), color=(0, 200, 0), thickness=-1)
    cv2.imwrite(os.path.join(execution_dir, 'synthetic_green_obstacle_2_image.jpg'), obstacle_2_image)
    trajectory, waypoints_coordinates = get_trajectory(td_obstacle_2_summary, os.path.join(execution_dir, 'synthetic_green_obstacle_2_image.jpg'), start_waypoint_idx=5)
    plot_trajectory(td_obstacle_2_summary, obstacle_2_image_path, trajectory, waypoints_coordinates, start_waypoint_idx=5,
                    output_file_path=os.path.join(execution_dir, 'obstacle_2_trajectory.jpg'))
