import os
import json
import cv2
import numpy as np

from framework import utils
from framework import cv_utils
from computer_vision import segmentation


#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
apr_selected_td_experiments = {'noon': 'trunks_detection_on_apr_15-08-1', 'late_noon': 'trunks_detection_on_apr_15-53-1',
                  'afternoon': 'trunks_detection_on_apr_16-55-1', 'late_afternoon': 'trunks_detection_on_apr_19-04-1'}
nov_selected_td_experiment = 'trunks_detection_on_nov1_10-12-1'
#################################################################################################

from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as apr_td_results_dir
from content.data_pointers.lavi_november_18.dji import trunks_detection_results_dir as nov_td_results_dir

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('contours_comparison')

    with open(os.path.join(apr_td_results_dir, apr_selected_td_experiments['noon'], 'experiment_summary.json')) as f:
        apr_noon_td_summary = json.load(f)
    with open(os.path.join(apr_td_results_dir, apr_selected_td_experiments['late_noon'], 'experiment_summary.json')) as f:
        apr_late_noon_td_summary = json.load(f)
    with open(os.path.join(apr_td_results_dir, apr_selected_td_experiments['afternoon'], 'experiment_summary.json')) as f:
        apr_afternoon_td_summary = json.load(f)
    with open(os.path.join(apr_td_results_dir, apr_selected_td_experiments['late_afternoon'], 'experiment_summary.json')) as f:
        apr_late_afternoon_td_summary = json.load(f)
    with open(os.path.join(nov_td_results_dir, nov_selected_td_experiment, 'experiment_summary.json')) as f:
        nov_td_summary = json.load(f)

    apr_noon_image = cv2.imread(apr_noon_td_summary['data_sources'])
    apr_late_noon_image = cv2.imread(apr_late_noon_td_summary['data_sources'])
    apr_afternoon_image = cv2.imread(apr_afternoon_td_summary['data_sources'])
    apr_late_afternoon_image = cv2.imread(apr_late_afternoon_td_summary['data_sources'])
    nov_image = cv2.imread(nov_td_summary['data_sources'])

    apr_noon_trunks = apr_noon_td_summary['results']['1']['semantic_trunks']
    apr_late_noon_trunks = apr_late_noon_td_summary['results']['1']['semantic_trunks']
    apr_afternoon_trunks = apr_afternoon_td_summary['results']['1']['semantic_trunks']
    apr_late_afternoon_trunks = apr_late_afternoon_td_summary['results']['1']['semantic_trunks']
    nov_trunks = nov_td_summary['results']['1']['semantic_trunks']

    apr_late_noon_image, _ = cv_utils.warp_image(apr_late_noon_image, apr_late_noon_trunks.values(), apr_noon_trunks.values(), method='affine')
    apr_afternoon_image, _ = cv_utils.warp_image(apr_afternoon_image, apr_afternoon_trunks.values(), apr_noon_trunks.values(), method='affine')
    apr_late_afternoon_image, _ = cv_utils.warp_image(apr_late_afternoon_image, apr_late_afternoon_trunks.values(), apr_noon_trunks.values(), method='affine')
    nov_image, _ = cv_utils.warp_image(nov_image, nov_trunks.values(), apr_noon_trunks.values(), method='affine')
    cv2.imwrite(os.path.join(execution_dir, 'apr_noon.jpg'), apr_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'apr_late_noon.jpg'), apr_late_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'apr_afternoon.jpg'), apr_afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'apr_late_afternoon.jpg'), apr_late_afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'nov.jpg'), nov_image)

    apr_noon_contours, _ = segmentation.extract_canopy_contours(apr_noon_image)
    apr_late_noon_contours, _ = segmentation.extract_canopy_contours(apr_late_noon_image)
    apr_afternoon_contours, _ = segmentation.extract_canopy_contours(apr_afternoon_image)
    apr_late_afternoon_contours, _ = segmentation.extract_canopy_contours(apr_late_afternoon_image)
    nov_contours, _ = segmentation.extract_canopy_contours(nov_image)

    cv2.drawContours(apr_noon_image, apr_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(apr_late_noon_image, apr_late_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(apr_afternoon_image, apr_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(apr_late_afternoon_image, apr_late_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(nov_image, nov_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.imwrite(os.path.join(execution_dir, 'apr_noon_contours.jpg'), apr_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'apr_late_noon_contours.jpg'), apr_late_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'apr_afternoon_contours.jpg'), apr_afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'apr_late_afternoon_contours.jpg'), apr_late_afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'nov_contours.jpg'), nov_image)

    cropped_apr_noon_image = cv_utils.center_crop(apr_noon_image, x_ratio=0.4, y_ratio=0.4)
    cropped_apr_late_noon_image = cv_utils.center_crop(apr_late_noon_image, x_ratio=0.4, y_ratio=0.4)
    cropped_apr_afternoon_image = cv_utils.center_crop(apr_afternoon_image, x_ratio=0.4, y_ratio=0.4)
    cropped_apr_late_afternoon_image = cv_utils.center_crop(apr_late_afternoon_image, x_ratio=0.4, y_ratio=0.4)
    cropped_nov_image = cv_utils.center_crop(nov_image, x_ratio=0.4, y_ratio=0.4)
    cv2.imwrite(os.path.join(execution_dir, 'cropped_apr_noon.jpg'), cropped_apr_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'cropped_apr_late_noon.jpg'), cropped_apr_late_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'cropped_apr_afternoon.jpg'), cropped_apr_afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'cropped_apr_late_afternoon.jpg'), cropped_apr_late_afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'cropped_nov.jpg'), cropped_nov_image)

    image_shape = apr_noon_image.shape
    all_contours_image = np.zeros(image_shape)
    cv2.drawContours(all_contours_image, apr_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(all_contours_image, apr_late_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(all_contours_image, apr_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(all_contours_image, apr_late_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    apr_all_contours_image = all_contours_image[600:-100, 300:-400]
    cv2.imwrite(os.path.join(execution_dir, 'apr_all_contours.jpg'), apr_all_contours_image)
    cv2.drawContours(all_contours_image, nov_contours, contourIdx=-1, color=(255, 0, 255), thickness=3)
    apr_nov_all_contours_image = all_contours_image[600:-100, 300:-400]
    cv2.imwrite(os.path.join(execution_dir, 'apr_nov_all_contours.jpg'), apr_nov_all_contours_image)