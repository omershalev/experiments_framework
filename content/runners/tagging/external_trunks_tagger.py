import os
import cv2
import json

from framework import utils
from framework import viz_utils
from framework import cv_utils


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
setup = 'apr' # apr / nov1 / nov2
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments
elif setup == 'nov1':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir
    from content.data_pointers.lavi_november_18.dji import plot1_selected_trunks_detection_experiments as selected_trunks_detection_experiments
elif setup == 'nov2':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir
    from content.data_pointers.lavi_november_18.dji import plot2_selected_trunks_detection_experiments as selected_trunks_detection_experiments


if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('external_trunks_tagging')
    for experiment_name in selected_trunks_detection_experiments:
        with open(os.path.join(trunks_detection_results_dir, experiment_name, 'experiment_summary.json')) as f:
            experiment_summary = json.load(f)
        image = cv2.imread(experiment_summary['data_sources'])
        external_trunk_poses = cv_utils.sample_pixel_coordinates(image, multiple=True)
        image_with_points = cv_utils.draw_points_on_image(image, external_trunk_poses, color=(255, 255, 255))
        viz_utils.show_image('external_trunks', image_with_points)
        with open(os.path.join(trunks_detection_results_dir, experiment_name, 'external_trunks.json'), 'w') as f:
            json.dump(external_trunk_poses, f, indent=4)
