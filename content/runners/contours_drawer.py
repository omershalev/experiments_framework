import os
import cv2
import json

from computer_vision import segmentation
from framework import utils

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
source_image_index = 2
setup = 'nov1' # apr / nov1
#################################################################################################

if setup == 'apr':
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments as selected_td_experiments
elif setup == 'nov1':
    from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as td_results_dir
    from content.data_pointers.lavi_november_18.dji import selected_trunks_detection_experiments as selected_td_experiments
else:
    raise NotImplementedError

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('canopy_contours_drawer')
    with open(os.path.join(td_results_dir, selected_td_experiments[source_image_index], 'experiment_summary.json')) as f:
        td_summary = json.load(f)
    image = cv2.imread(td_summary['data_sources'])
    contours, canopies_mask = segmentation.extract_canopy_contours(image)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, contourIdx=-1, color=(0, 255, 0), thickness=5)
    canopies_mask_with_contours = cv2.cvtColor(canopies_mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canopies_mask_with_contours, contours, contourIdx=-1, color=(0, 255, 0), thickness=5)

    cv2.imwrite(os.path.join(execution_dir, 'image.jpg'), image)
    cv2.imwrite(os.path.join(execution_dir, 'canopies_mask.jpg'), canopies_mask)
    cv2.imwrite(os.path.join(execution_dir, 'image_with_contours.jpg'), image_with_contours)
    cv2.imwrite(os.path.join(execution_dir, 'canopies_mask_with_contours.jpg'), canopies_mask_with_contours)