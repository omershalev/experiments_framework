import os
import cv2

from computer_vision.contours_scan_cython import contours_scan
from computer_vision import maps_generation
from framework import utils
from framework import cv_utils
from framework import config

#################################################################################################
#                                             CONFIG                                            #
#################################################################################################
from content.data_pointers.lavi_april_18.dji import snapshots_80_meters as snapshots
image_key = '15-08-1'
roi_expansion = 2.8
#################################################################################################

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('synthetic_scan_drawer')
    image = cv2.imread(snapshots[image_key].path)
    map_image = maps_generation.generate_canopies_map(image)
    dummy_resolution = 1
    center_x, center_y = cv_utils.sample_pixel_coordinates(map_image)
    scan_ranges = contours_scan.generate(map_image,
                                         center_x=center_x,
                                         center_y=center_y,
                                         min_angle=config.synthetic_scan_min_angle,
                                         max_angle=config.synthetic_scan_max_angle,
                                         samples_num=config.synthetic_scan_samples_num,
                                         min_distance=config.synthetic_scan_min_distance,
                                         max_distance=config.synthetic_scan_max_distance,
                                         resolution=dummy_resolution,
                                         r_primary_search_samples=config.synthetic_scan_r_primary_search_samples,
                                         r_secondary_search_step=config.synthetic_scan_r_secondary_search_step)
    trunks_scan_points_list = cv_utils.get_coordinates_list_from_scan_ranges(scan_ranges, center_x, center_y,
                                                                             config.synthetic_scan_min_angle,
                                                                             config.synthetic_scan_max_angle,
                                                                             dummy_resolution)
    viz_image = image.copy()
    viz_image = cv_utils.draw_points_on_image(viz_image, trunks_scan_points_list, color=(0, 0, 255), radius=3)
    cv2.circle(viz_image, (center_x, center_y), radius=8, color=(255, 0, 255), thickness=-1)
    cv2.circle(viz_image, (center_x, center_y), radius=config.synthetic_scan_max_distance, color=(120, 0, 0), thickness=2)
    viz_image, _, _ = cv_utils.crop_region(viz_image, center_x, center_y, config.synthetic_scan_max_distance * roi_expansion, config.synthetic_scan_max_distance * roi_expansion)
    cv2.imwrite(os.path.join(execution_dir, 'synthetic_scan.jpg'), viz_image)