import cv2
import numpy as np
from itertools import product
import rospy
import torch

from experiments_framework.framework import ros_utils
from air_ground_orchard_navigation.computer_vision import segmentation, contours_scan
from experiments_framework.content.data_pointers.lavi_april_18 import panorama
from experiments_framework.framework import cv_utils
from experiments_framework.framework import viz_utils
from experiments_framework.framework import ros_utils
from experiments_framework.framework import config

if __name__ == '__main__':
    image_path = panorama.full_orchard['dji_afternoon'].path
    image = cv2.imread(image_path)
    map_image = segmentation.extract_canopies_map(image)
    i = 0
    for x, y in product(xrange(map_image.shape[1]), xrange(map_image.shape[0])):
        if map_image[(y, x)] == 0:
            print (x, y)
            map_image = cv2.circle(map_image, (x, y), radius=10, color=(255, 255, 255), thickness=2)
            range, coordinates_list = contours_scan.generate(map_image, center_x=x, center_y=y, min_angle=0, max_angle=2*np.pi, samples_num=360, min_distance=3, max_distance=300, resolution=0.0125) # TODO: resolution
    viz_utils.show_image('points', map_image)