import cv2
import numpy as np
# import rospy
# import numba

from experiments_framework.framework import ros_utils
from air_ground_orchard_navigation.computer_vision import segmentation, contours_scan
from experiments_framework.content.data_pointers.lavi_april_18 import panorama
from experiments_framework.framework import viz_utils
from experiments_framework.framework import cv_utils
# from experiments_framework.framework import ros_utils
from experiments_framework.framework import config

import datetime

# def test(image):
#     start = datetime.datetime.now()
#     for theta in np.linspace(start=0,stop=np.pi, num=720):
#         for r in np.linspace(start=2, stop=300):
#             if image[(int(500+r*np.cos(theta)), int(500+r*np.sin(theta)))] == 1:
#                 pass
#     end = datetime.datetime.now()
#     delta = (end-start).microseconds
#     print (delta)


if __name__ == '__main__':
    image_path = panorama.full_orchard['dji_afternoon'].path
    image = cv2.imread(image_path)
    map_image = segmentation.extract_canopies_map(image)

    # test(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    # ros_utils.save_image_to_map(map_image, resolution=0.0125, map_name='temp', dir_name=r'/home/omer/Downloads')
    # cv_utils.mark_trajectory_on_image(map_image)

    # ros_utils.start_master(use_sim_time=False)
    # rospy.init_node('betach_at_omeret')
    # import rospy.rostime as rostime
    # import time
    # for _ in range(5):
    #     time.sleep(1)
    #     now = rostime.Time.now()
    #     print type(now)
    # ros_utils.kill_master()


    x, y = cv_utils.sample_pixel_coordinates(map_image)
    range, coordinates_list = contours_scan.generate(map_image, center_x=x, center_y=y, min_angle=0, max_angle=2*np.pi, samples_num=360, min_distance=3, max_distance=300, resolution=0.0125) # TODO: resolution
    map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
    for (scan_x,scan_y) in coordinates_list:
        map_image = cv2.circle(map_image, (scan_x, scan_y), radius=10, color=(255,0,255), thickness=2)
    viz_utils.show_image('scan', map_image)