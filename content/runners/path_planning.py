import os
import json
import numpy as np
import random
from framework import utils
from content.data_pointers.lavi_april_18 import orchard_topology
from content.experiments.path_planning import PathPlanningExperiment
from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir, selected_trunks_detection_experiments_and_repetitions

num_of_trajectories = 20
trunk_radius = 30 # TODO: change according to measurements

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('path_planning')
    trunks_detection_experiment_name, trunks_detection_repetition = selected_trunks_detection_experiments_and_repetitions[0]
    with open(os.path.join(trunks_detection_results_dir, trunks_detection_experiment_name, 'experiment_summary.json')) as f:
        trunks_detection_summary = json.load(f)
    image_path = trunks_detection_summary['data_sources']
    image_key = trunks_detection_summary['metadata']['image_key']
    trunk_points_list = trunks_detection_summary['results'][str(trunks_detection_repetition)]['pattern_points']
    orchard_pattern = orchard_topology.orchard_pattern
    i = 0

    while i < num_of_trajectories:
        start_index = random.randint(0, len(trunk_points_list) - 1)
        goal_index = random.randint(0, len(trunk_points_list) - 1)
        if start_index == goal_index:
            continue
        i += 1
        start = (int(np.round(trunk_points_list[start_index][0] + trunk_radius + 5)), int(np.round(trunk_points_list[start_index][1])))
        goal = (int(np.round(trunk_points_list[goal_index][0] + trunk_radius + 5)), int(np.round(trunk_points_list[goal_index][1])))
        experiment = PathPlanningExperiment(name='path_planning_on_%s_from_%s_to_%s' % (image_key, start, goal),
                                            data_sources={'image_path': image_path, 'trunk_points_list': trunk_points_list, 'trunk_radius': trunk_radius, # TODO: change 30
                                                          'canopy_sigma': 150, 'start': start, 'goal': goal}, # TODO: change 100
                                            working_dir=execution_dir, metadata={'image_key': image_key, 'altitude': 80})
        experiment.run(repetitions=1)