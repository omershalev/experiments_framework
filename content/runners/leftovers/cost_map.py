import json
import cv2
import numpy as np

from framework import viz_utils
from framework import cv_utils
from content.data_pointers.lavi_april_18 import dji
from computer_vision import segmentation
from computer_vision import trunks_detection
from skimage.measure import compare_ssim

viz_mode = True

image_key = '15-08-1'

metadata_path = r'/home/omer/Downloads/experiment_metadata_baseline.json'


if __name__ == '__main__':

    with open(metadata_path) as f:
        metadata = json.load(f)
    trunks = metadata['results']['1']['trunk_points_list']
    optimized_sigma = metadata['results']['1']['optimized_sigma']


    image = cv2.imread(dji.snapshots_80_meters[image_key].path)
    if viz_mode:
        viz_utils.show_image('image', image)

    trunks = [(int(elem[0]), int(elem[1])) for elem in trunks]
    upper_left, lower_right = cv_utils.get_bounding_box(image, trunks, expand_ratio=0.1)
    cropped_image = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
    trunks = np.array(trunks) - np.array(upper_left)

    if viz_mode:
        viz_utils.show_image('cropped image', cropped_image)

    gaussians = trunks_detection.get_gaussians_grid_image(trunks, optimized_sigma, cropped_image.shape[1], cropped_image.shape[0], scale_factor=0.7,
                                                          square_size_to_sigma_ratio=3, circle_radius_to_sigma_ratio=3)
    if viz_mode:
        viz_utils.show_image('gaussians', gaussians)

    contours, contours_mask = segmentation.extract_canopy_contours(cropped_image)
    cost_map = cv2.bitwise_and(gaussians, gaussians, mask=contours_mask)
    cost_map = cv_utils.draw_points_on_image(cost_map, trunks, color=255, radius=20)
    # cost_map = cv2.drawContours(cost_map, contours, contourIdx=-1, color=255, thickness=3)
    if viz_mode:
        viz_utils.show_image('cost_map', cost_map)
