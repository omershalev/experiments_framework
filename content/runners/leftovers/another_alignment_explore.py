import json
import cv2

from framework import viz_utils
from framework import cv_utils
from content.data_pointers.lavi_april_18 import dji

viz_mode = False

if __name__ == '__main__':
    with open(r'/home/omer/Downloads/experiment_metadata1.json') as f:
        metadata1 = json.load(f)
    with open(r'/home/omer/Downloads/experiment_metadata2.json') as f:
        metadata2 = json.load(f)
    with open(dji.snapshots_60_meters_markers_locations_json_path) as f:
        markers_locations = json.load(f)

    image_key1 = '19-03-3'
    image_key2 = '16-54-1'

    image1 = cv2.imread(metadata1['data_sources'])
    image2 = cv2.imread(metadata2['data_sources'])
    points1 = metadata1['results']['1']['pattern_points']
    points2 = metadata2['results']['1']['pattern_points']
    markers1 = markers_locations[image_key1]
    markers2 = markers_locations[image_key2]

    print 'groud truth:'
    warped_image_gt2 = cv_utils.warp_image(image=image2, points_in_image=markers2, points_in_baseline=markers1)
    print cv_utils.calculate_image_diff(image1, warped_image_gt2, method='ssim')
    print cv_utils.calculate_image_diff(image1, warped_image_gt2, method='mse')
    print ('')
    warped_image_gt1 = cv_utils.warp_image(image=image1, points_in_image=markers1, points_in_baseline=markers2)
    print cv_utils.calculate_image_diff(image2, warped_image_gt1, method='ssim')
    print cv_utils.calculate_image_diff(image2, warped_image_gt1, method='mse')


    if viz_mode:
        viz_utils.show_image('image1', image1)
        viz_utils.show_image('image2', image2)

    # TODO: consider playing with the crop_ratio for calculate_image_diff
    print '\nmine:'
    warped_image2 = cv_utils.warp_image(image=image2, points_in_image=points2, points_in_baseline=points1)
    if viz_mode:
        viz_utils.show_image('warped image', warped_image2)
    print cv_utils.calculate_image_diff(image1, warped_image2, method='ssim')
    print cv_utils.calculate_image_diff(image1, warped_image2, method='mse')
    print ('')
    warped_image1 = cv_utils.warp_image(image=image1, points_in_image=points1, points_in_baseline=points2)
    if viz_mode:
        viz_utils.show_image('warped image', warped_image1)
    print cv_utils.calculate_image_diff(image2, warped_image1, method='ssim')
    print cv_utils.calculate_image_diff(image2, warped_image1, method='mse')
