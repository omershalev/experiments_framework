import os
import json
from collections import OrderedDict

from framework import utils
from computer_vision import calibration
from content.experiments.global_ekf_updates import GlobalEkfUpdatesExperiment

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################


#################################################################################################

from content.data_pointers.lavi_november_18.jackal import jackal_18
from content.data_pointers.lavi_november_18.dji import plot1_snapshots_80_meters_ugv_poses_path, trunks_detection_results_dir
from content.data_pointers.lavi_november_18 import orchard_topology


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('global_ekf_updates')
    with open(plot1_snapshots_80_meters_ugv_poses_path) as f:
        ugv_poses = json.load(f, object_pairs_hook=OrderedDict)
    for relevant_update_index, update_image_key in enumerate(ugv_poses.keys()[1:]):
        with open(os.path.join(trunks_detection_results_dir, 'trunks_detection_on_nov1_%s' % update_image_key, 'experiment_summary.json')) as f:
            td_experiment_summary = json.load(f)
        pixel_to_meter = calibration.calculate_pixel_to_meter(td_experiment_summary['results']['1']['optimized_grid_dim_x'],
                                                              td_experiment_summary['results']['1']['optimized_grid_dim_y'],
                                                              orchard_topology.plot1_measured_row_widths,
                                                              orchard_topology.plot1_measured_intra_row_distances)
        resolution = 1.0 / pixel_to_meter
        experiment = GlobalEkfUpdatesExperiment(name='global_ekf_update_%d' % (relevant_update_index + 1),
                                                data_sources={'jackal_bag_path': jackal_18['10-25'].path,
                                                              'ugv_poses_path': plot1_snapshots_80_meters_ugv_poses_path,
                                                              'relevant_update_index': (relevant_update_index + 1)},
                                                params={'bag_start_time': 85, 'resolution': resolution},
                                                working_dir=execution_dir)
        experiment.run(repetitions=1)