import os
import cv2
import numpy as np

from computer_vision import segmentation
import framework.viz_utils as viz_utils
import framework.utils as utils
import content.data_pointers.lavi_april_18.dji as dji_data

image_paths_list = [descriptor.path for descriptor in dji_data.snapshots_60_meters.values() + dji_data.snapshots_80_meters.values()]
show = True
image_path = r'/home/omer/orchards_ws/data/lavi_apr_18/panorama/DJI_0178_afternoon_good_stitch_full_movie.jpg' # ignored if None

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('canopy_contours_images')
    if image_path is not None:
        image = cv2.imread(image_path)
        contours, _ = segmentation.extract_canopy_contours(image)
        cv2.drawContours(image, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        cv2.imwrite(os.path.join(execution_dir, 'image.jpg'), image)
        if show:
            viz_utils.show_image('image', image)
    else:
        idx = 0
        for image_path in image_paths_list:
            image = cv2.imread(image_path)
            contours, _ = segmentation.extract_canopy_contours(image)
            cv2.drawContours(image, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
            cv2.imwrite(os.path.join(execution_dir, '%s.jpg' % idx), image)
            idx += 1

            if show:
                viz_utils.show_image('image', image)
