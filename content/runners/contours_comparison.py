import os
import json
import cv2
import numpy as np

from framework import utils
from framework import cv_utils
from framework import config
from computer_vision import segmentation

config.base_raw_data_path = os.path.join(config.root_dir_path, 'resources/lavi_apr_18/raw')
config.markers_locations_path = os.path.join(config.root_dir_path, 'resources/lavi_apr_18/markers_locations')
from content.data_pointers.lavi_april_18.dji import snapshots_80_meters as snapshots_pointers
from content.data_pointers.lavi_april_18.dji import snapshots_80_meters_markers_locations_json_path as markers_pointers

image_keys = {'noon': '15-08-1', 'late_noon': '15-53-1', 'afternoon': '16-55-1', 'late_afternoon': '19-04-1'}

if __name__ == '__main__':

    execution_dir = utils.create_new_execution_folder('image_slice_comparison')

    with open(markers_pointers) as f:
        markers = json.load(f)

    noon_image = cv2.imread(snapshots_pointers[image_keys['noon']].path)
    late_noon_image = cv2.imread(snapshots_pointers[image_keys['late_noon']].path)
    afternoon_image = cv2.imread(snapshots_pointers[image_keys['afternoon']].path)
    late_afternoon_image = cv2.imread(snapshots_pointers[image_keys['late_afternoon']].path)

    noon_markers = markers[image_keys['noon']]
    late_noon_markers = markers[image_keys['late_noon']]
    afternoon_markers = markers[image_keys['afternoon']]
    late_afternoon_markers = markers[image_keys['late_afternoon']]

    late_noon_image, _ = cv_utils.warp_image(late_noon_image, late_noon_markers, noon_markers, method='affine')
    afternoon_image, _ = cv_utils.warp_image(afternoon_image, afternoon_markers, noon_markers, method='affine')
    late_afternoon_image, _ = cv_utils.warp_image(late_afternoon_image, late_afternoon_markers, noon_markers, method='affine')
    # noon_image = cv_utils.center_crop(noon_image, x_ratio=0.4, y_ratio=0.4)
    # late_noon_image = cv_utils.center_crop(late_noon_image, x_ratio=0.4, y_ratio=0.4)
    # afternoon_image = cv_utils.center_crop(afternoon_image, x_ratio=0.4, y_ratio=0.4)
    # late_afternoon_image = cv_utils.center_crop(late_afternoon_image, x_ratio=0.4, y_ratio=0.4)
    cv2.imwrite(os.path.join(execution_dir, 'noon_aligned.jpg'), noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'late_noon_aligned.jpg'), late_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'afternoon_aligned.jpg'), afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'late_afternoon_aligned.jpg'), late_afternoon_image)

    noon_contours, _ = segmentation.extract_canopy_contours(noon_image)
    late_noon_contours, _ = segmentation.extract_canopy_contours(late_noon_image)
    afternoon_contours, _ = segmentation.extract_canopy_contours(afternoon_image)
    late_afternoon_contours, _ = segmentation.extract_canopy_contours(late_afternoon_image)

    cv2.drawContours(noon_image, noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(late_noon_image, late_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(afternoon_image, afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(late_afternoon_image, late_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.imwrite(os.path.join(execution_dir, 'noon_contours.jpg'), noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'late_noon_contours.jpg'), late_noon_image)
    cv2.imwrite(os.path.join(execution_dir, 'afternoon_contours.jpg'), afternoon_image)
    cv2.imwrite(os.path.join(execution_dir, 'late_afternoon_contours.jpg'), late_afternoon_image)

    image_shape = noon_image.shape
    noon_contours_only_image = cv2.drawContours(np.zeros(image_shape), noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    late_noon_contours_only_image = cv2.drawContours(np.zeros(image_shape), late_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    afternoon_contours_only_image = cv2.drawContours(np.zeros(image_shape), afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    late_afternoon_contours_only_image = cv2.drawContours(np.zeros(image_shape), late_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.imwrite(os.path.join(execution_dir, 'noon_contours_only.jpg'), noon_contours_only_image)
    cv2.imwrite(os.path.join(execution_dir, 'late_noon_contours_only.jpg'), late_noon_contours_only_image)
    cv2.imwrite(os.path.join(execution_dir, 'afternoon_contours_only.jpg'), afternoon_contours_only_image)
    cv2.imwrite(os.path.join(execution_dir, 'late_afternoon_contours_only.jpg'), late_afternoon_contours_only_image)

    all_contours = np.zeros(image_shape)
    cv2.drawContours(all_contours, noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(all_contours, late_noon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(all_contours, afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.drawContours(all_contours, late_afternoon_contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    cv2.imwrite(os.path.join(execution_dir, 'all_contours.jpg'), all_contours)