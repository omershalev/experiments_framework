import cv2
import numpy as np

from air_ground_orchard_navigation.computer_vision import segmentation, contours_scan2
# import sys
# sys.path.append(r'/home/omer/orchards_ws/astar/air_ground_orchard_navigation/computer_vision/astar/air_ground_orchard_navigation/computer_vision')
# import contours_scan
from air_ground_orchard_navigation.computer_vision.contours_scan_cython import contours_scan
from framework import config
from content.data_pointers.lavi_april_18 import panorama
from framework import viz_utils
from framework import cv_utils


if __name__ == '__main__':
    image_path = panorama.full_orchard['dji_afternoon'].path
    image = cv2.imread(image_path)
    map_image = segmentation.extract_canopies_map(image)

    x, y = cv_utils.sample_pixel_coordinates(map_image)
    import datetime
    start1 = datetime.datetime.now()
    range1, coordinates_list = contours_scan2.generate(map_image, center_x=x, center_y=y, min_angle=0, max_angle=2 * np.pi, samples_num=360, min_distance=3, max_distance=300, resolution=config.top_view_resolution) # TODO: resolution
    end1 = datetime.datetime.now()
    print 'without: ' + str((end1 - start1).microseconds)
    start2 = datetime.datetime.now()
    range2 = contours_scan.generate(map_image, center_x=x, center_y=y, min_angle=0, max_angle=2 * np.pi, samples_num=360, min_distance=3, max_distance=300, resolution=config.top_view_resolution) # TODO: resolution
    end2 = datetime.datetime.now()
    print 'with: ' + str((end2 - start2).microseconds)
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
    for (scan_x,scan_y) in coordinates_list:
        map_image = cv2.circle(map_image, (scan_x, scan_y), radius=10, color=(255,0,255), thickness=2)
    viz_utils.show_image('scan', map_image)