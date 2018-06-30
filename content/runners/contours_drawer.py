import cv2
import os

import air_ground_orchard_navigation.computer_vision.segmentation as canopy_contours
import experiments_framework.framework.viz_utils as viz_utils
import experiments_framework.framework.utils as utils
import experiments_framework.content.data_pointers.lavi_april_18.dji as dji_data

image_paths_list = [descriptor.path for descriptor in dji_data.snapshots_60_meters.values() + dji_data.snapshots_80_meters.values()]
show = True

if __name__ == '__main__':
    samples = []
    execution_dir = utils.create_new_execution_folder('canopy_contours_images')
    idx = 0
    for image_path in image_paths_list:
        image = cv2.imread(image_path)
        contours, _ = canopy_contours.extract_canopy_contours(image)
        cv2.drawContours(image, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv2.imwrite(os.path.join(execution_dir, '%s.jpg' % idx), image)
        idx += 1
        if show:
            viz_utils.show_image('image', image)