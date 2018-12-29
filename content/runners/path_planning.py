import os
import json
import cv2
import numpy as np

from computer_vision import calibration
from framework import utils
from framework import cv_utils
from framework import config
from content.experiments.path_planning import PathPlanningExperiment


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
num_of_trajectories = 20
description = 'path_planning'
two_snapshot = False
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
    trunks_detection_experiment_name = selected_td_experiments[0]
    with open(os.path.join(td_results_dir, trunks_detection_experiment_name, 'experiment_summary.json')) as f:
        trunks_detection_summary = json.load(f)
    image_path = trunks_detection_summary['data_sources']
    image_key = trunks_detection_summary['metadata']['image_key']
    semantic_trunks = trunks_detection_summary['results']['0']['semantic_trunks']
    trunk_points_list = semantic_trunks.values()
    image = cv2.imread(image_path)
    upper_left, lower_right = cv_utils.get_bounding_box(image, trunk_points_list, expand_ratio=config.bounding_box_expand_ratio)
    orchard_pattern = orchard_topology.plot_pattern
    row_labels = np.unique([label.split('/')[0] for label in semantic_trunks.keys()])
    tree_labels = np.unique([label.split('/')[1] for label in semantic_trunks.keys()])
    i = 0
    while i < num_of_trajectories:
        start_label_1 = semantic_trunks.keys()[np.random.randint(low=0, high=len(semantic_trunks.keys()))]
        start_label_2 = '/'.join([str(int(start_label_1.split('/')[0]) + 1), start_label_1.split('/')[1]])
        if start_label_2 not in semantic_trunks.keys():
            continue
        goal_label_1 = semantic_trunks.keys()[np.random.randint(low=0, high=len(semantic_trunks.keys()))]
        goal_label_2 = '/'.join([str(int(goal_label_1.split('/')[0]) + 1), goal_label_1.split('/')[1]])
        if goal_label_2 not in semantic_trunks.keys():
            continue
        if start_label_1 == goal_label_1:
            continue
        start = ((semantic_trunks[start_label_1][0] + semantic_trunks[start_label_2][0]) / 2,
                 (semantic_trunks[start_label_1][1] + semantic_trunks[start_label_2][1]) / 2)
        goal = ((semantic_trunks[goal_label_1][0] + semantic_trunks[goal_label_2][0]) / 2,
                (semantic_trunks[goal_label_1][1] + semantic_trunks[goal_label_2][1]) / 2)
        waypoints = [start, goal]
        experiment = PathPlanningExperiment(name='path_planning_on_%s_from_%s-%s_to_%s-%s' % (image_key,
                                                                                              start_label_1.replace('/', ''),
                                                                                              start_label_2.replace('/', ''),
                                                                                              goal_label_1.replace('/', ''),
                                                                                              goal_label_2.replace('/', '')),
                                            data_sources={'map_image_path': image_path, 'trunk_points_list': trunk_points_list,
                                                          'map_upper_left': upper_left, 'map_lower_right': lower_right, 'waypoints': waypoints},
                                            params={'semantic_trunks': {label: semantic_trunks[label] for label in [start_label_1, start_label_2, goal_label_1, goal_label_2]}},
                                            working_dir=execution_dir, metadata=trunks_detection_summary['metadata'])
        experiment.run(repetitions=1)
        i += 1