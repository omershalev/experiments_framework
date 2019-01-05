import json
import cv2
import numpy as np

from framework import viz_utils
from framework import cv_utils
from content.data_pointers.lavi_april_18 import dji
from computer_vision import segmentation
from skimage.measure import compare_ssim

############################################################################################################################
#                                                    CONFIGURATION AREA                                                    #
############################################################################################################################
viz_mode = True
use_ground_truth = True

baseline_image_key = '15-08-1'
obstacle_in_3_4_image_key = '15-17-1'
obstacle_in_4_5_image_key = '15-18-3'
obstacle_in_5_6_image_key = '15-19-1'
############################################################################################################################

metadata_baseline_path = r'/home/omer/Downloads/experiment_metadata_baseline.json'
metadata_obstacle_in_3_4_path = r'/home/omer/Downloads/experiment_metadata_obstacle_in_3_4.json'
metadata_obstacle_in_4_5_path = r'/home/omer/Downloads/experiment_metadata_obstacle_in_4_5.json'
metadata_obstacle_in_5_6_path = r'/home/omer/Downloads/experiment_metadata_obstacle_in_5_6.json'


if __name__ == '__main__':

    baseline_image = cv2.imread(dji.snapshots_80_meters[baseline_image_key].path)
    obstacle_in_3_4_image = cv2.imread(dji.snapshots_80_meters[obstacle_in_3_4_image_key].path)
    obstacle_in_4_5_image = cv2.imread(dji.snapshots_80_meters[obstacle_in_4_5_image_key].path)
    obstacle_in_5_6_image = cv2.imread(dji.snapshots_80_meters[obstacle_in_5_6_image_key].path)


    baseline_gray_image = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)
    obstacle_in_3_4_gray_image = cv2.cvtColor(obstacle_in_3_4_image, cv2.COLOR_BGR2GRAY)
    obstacle_in_4_5_gray_image = cv2.cvtColor(obstacle_in_4_5_image, cv2.COLOR_BGR2GRAY)
    obstacle_in_5_6_gray_image = cv2.cvtColor(obstacle_in_5_6_image, cv2.COLOR_BGR2GRAY)

    if viz_mode:
        viz_utils.show_image('baseline', baseline_gray_image)
        viz_utils.show_image('obstacle_in_3_4', obstacle_in_3_4_gray_image)
        viz_utils.show_image('obstacle_in_4_5', obstacle_in_4_5_gray_image)
        viz_utils.show_image('obstacle_in_5_6', obstacle_in_5_6_gray_image)

    with open(metadata_baseline_path) as f:
        metadata_baseline = json.load(f)
    with open(metadata_obstacle_in_3_4_path) as f:
        metadata_obstacle_in_3_4 = json.load(f)
    with open(metadata_obstacle_in_4_5_path) as f:
        metadata_obstacle_in_4_5 = json.load(f)
    with open(metadata_obstacle_in_5_6_path) as f:
        metadata_obstacle_in_5_6 = json.load(f)

    points_baseline = metadata_baseline['results']['1']['pattern_points']
    points_obstacle_in_3_4 = metadata_obstacle_in_3_4['results']['1']['pattern_points']
    points_obstacle_in_4_5 = metadata_obstacle_in_4_5['results']['1']['pattern_points']
    points_obstacle_in_5_6 = metadata_obstacle_in_5_6['results']['1']['pattern_points']

    with open(dji.snapshots_80_meters_markers_locations_json_path) as f:
        markers_locations = json.load(f)

    markers_basline = markers_locations[baseline_image_key]
    markers_obstacle_in_3_4 = markers_locations[obstacle_in_3_4_image_key]
    markers_obstacle_in_4_5 = markers_locations[obstacle_in_4_5_image_key]
    markers_obstacle_in_5_6 = markers_locations[obstacle_in_5_6_image_key]



    # TODO: try once more the enabling of my estimated transform (instead of the ground truth)
    if use_ground_truth:
        warped_obstacle_in_3_4_image, _ = cv_utils.warp_image(image=obstacle_in_3_4_image, points_in_image=markers_obstacle_in_3_4, points_in_baseline=markers_basline, method='rigid')
        warped_obstacle_in_4_5_image, _ = cv_utils.warp_image(image=obstacle_in_4_5_image, points_in_image=markers_obstacle_in_4_5, points_in_baseline=markers_basline, method='rigid')
        warped_obstacle_in_5_6_image, _ = cv_utils.warp_image(image=obstacle_in_5_6_image, points_in_image=markers_obstacle_in_5_6, points_in_baseline=markers_basline, method='rigid')
        warped_obstacle_in_3_4_gray_image, _ = cv_utils.warp_image(image=obstacle_in_3_4_gray_image, points_in_image=markers_obstacle_in_3_4, points_in_baseline=markers_basline, method='rigid')
        warped_obstacle_in_4_5_gray_image, _ = cv_utils.warp_image(image=obstacle_in_4_5_gray_image, points_in_image=markers_obstacle_in_4_5, points_in_baseline=markers_basline, method='rigid')
        warped_obstacle_in_5_6_gray_image, _ = cv_utils.warp_image(image=obstacle_in_5_6_gray_image, points_in_image=markers_obstacle_in_5_6, points_in_baseline=markers_basline, method='rigid')
    else:
        warped_obstacle_in_3_4_gray_image, _ = cv_utils.warp_image(image=obstacle_in_3_4_gray_image, points_in_image=points_obstacle_in_3_4, points_in_baseline=points_baseline, method='rigid')
        warped_obstacle_in_4_5_gray_image, _ = cv_utils.warp_image(image=obstacle_in_4_5_gray_image, points_in_image=points_obstacle_in_4_5, points_in_baseline=points_baseline, method='rigid')
        warped_obstacle_in_5_6_gray_image, _ = cv_utils.warp_image(image=obstacle_in_5_6_gray_image, points_in_image=points_obstacle_in_5_6, points_in_baseline=points_baseline, method='rigid')

    # _, baseline_to_obstacle_in_3_4_diff = compare_ssim(baseline_image, warped_obstacle_in_3_4_image, full=True)
    # baseline_to_obstacle_in_3_4_diff = (baseline_to_obstacle_in_3_4_diff * 255).astype('uint8')
    # _, baseline_to_obstacle_in_4_5_diff = compare_ssim(baseline_image, warped_obstacle_in_4_5_image, full=True)
    # baseline_to_obstacle_in_4_5_diff = (baseline_to_obstacle_in_4_5_diff * 255).astype('uint8')
    # _, baseline_to_obstacle_in_5_6_diff = compare_ssim(baseline_image, warped_obstacle_in_5_6_image, full=True)
    # baseline_to_obstacle_in_5_6_diff = (baseline_to_obstacle_in_5_6_diff * 255).astype('uint8')


    _, baseline_contours_mask = segmentation.extract_canopy_contours(baseline_image, margin_width=15, margin_color=255)
    baseline_contours_mask = cv2.dilate(baseline_contours_mask, kernel=np.ones((20, 20), np.uint8), iterations=1)
    baseline_contours_mask = 1 - (1 / 255.0) * baseline_contours_mask
    _, obstacle_in_3_4_contours_mask = segmentation.extract_canopy_contours(warped_obstacle_in_3_4_image, margin_width=15, margin_color=255)
    obstacle_in_3_4_contours_mask = cv2.dilate(obstacle_in_3_4_contours_mask, kernel=np.ones((20, 20), np.uint8), iterations=1)
    obstacle_in_3_4_contours_mask = 1 - (1 / 255.0) * obstacle_in_3_4_contours_mask
    _, obstacle_in_4_5_contours_mask = segmentation.extract_canopy_contours(warped_obstacle_in_4_5_image, margin_width=15, margin_color=255)
    obstacle_in_4_5_contours_mask = cv2.dilate(obstacle_in_4_5_contours_mask, kernel=np.ones((20, 20), np.uint8), iterations=1)
    obstacle_in_4_5_contours_mask = 1 - (1 / 255.0) * obstacle_in_4_5_contours_mask
    _, obstacle_in_5_6_contours_mask = segmentation.extract_canopy_contours(warped_obstacle_in_5_6_image, margin_width=15, margin_color=255)
    obstacle_in_5_6_contours_mask = cv2.dilate(obstacle_in_5_6_contours_mask, kernel=np.ones((20, 20), np.uint8), iterations=1)
    obstacle_in_5_6_contours_mask = 1 - (1 / 255.0) * obstacle_in_5_6_contours_mask

    baseline_to_obstacle_in_3_4_diff = cv2.subtract(baseline_gray_image, warped_obstacle_in_3_4_gray_image)
    baseline_to_obstacle_in_4_5_diff = cv2.subtract(baseline_gray_image, warped_obstacle_in_4_5_gray_image)
    baseline_to_obstacle_in_5_6_diff = cv2.subtract(baseline_gray_image, warped_obstacle_in_5_6_gray_image)

    baseline_to_obstacle_in_3_4_diff = np.multiply(baseline_to_obstacle_in_3_4_diff, obstacle_in_3_4_contours_mask)
    baseline_to_obstacle_in_4_5_diff = np.multiply(baseline_to_obstacle_in_4_5_diff, obstacle_in_4_5_contours_mask)
    baseline_to_obstacle_in_5_6_diff = np.multiply(baseline_to_obstacle_in_5_6_diff, obstacle_in_5_6_contours_mask)
    baseline_to_obstacle_in_3_4_diff = np.multiply(baseline_to_obstacle_in_3_4_diff, baseline_contours_mask)
    baseline_to_obstacle_in_4_5_diff = np.multiply(baseline_to_obstacle_in_4_5_diff, baseline_contours_mask)
    baseline_to_obstacle_in_5_6_diff = np.multiply(baseline_to_obstacle_in_5_6_diff, baseline_contours_mask)

    baseline_to_obstacle_in_5_6_diff = cv2.fastNlMeansDenoising(baseline_to_obstacle_in_5_6_diff.astype(np.uint8), None, 4, 7, 21)

    _, obstacle_contours, hierarchy = cv2.findContours(baseline_to_obstacle_in_5_6_diff.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda contour: cv2.contourArea(contour) > 500, obstacle_contours)
    obstacles_mask = cv2.drawContours(np.zeros((baseline_to_obstacle_in_5_6_diff.shape[0], baseline_to_obstacle_in_5_6_diff.shape[1]), np.uint8), contours, contourIdx=-1, color=255, thickness=-1)


    # cv2.imwrite(r'/home/omer/temp/warping/contours_mask.jpg', baseline_contours_mask * 255.0)
    cv2.imwrite(r'/home/omer/temp/warping/obstacles_mask.jpg', obstacles_mask)
    cv2.imwrite(r'/home/omer/temp/warping/diff_3_4.jpg', baseline_to_obstacle_in_3_4_diff)
    cv2.imwrite(r'/home/omer/temp/warping/diff_4_5.jpg', baseline_to_obstacle_in_4_5_diff)
    cv2.imwrite(r'/home/omer/temp/warping/diff_5_6.jpg', baseline_to_obstacle_in_5_6_diff)
    cv2.imwrite(r'/home/omer/temp/warping/baseline.jpg', baseline_gray_image)
    cv2.imwrite(r'/home/omer/temp/warping/warped_3_4.jpg', warped_obstacle_in_3_4_gray_image)
    cv2.imwrite(r'/home/omer/temp/warping/warped_4_5.jpg', warped_obstacle_in_4_5_gray_image)
    cv2.imwrite(r'/home/omer/temp/warping/warped_5_6.jpg', warped_obstacle_in_5_6_gray_image)