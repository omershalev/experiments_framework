#!/usr/bin/env python

import cv2
import numpy as np
import datetime
import time
import rospy
from computer_vision import segmentation
from computer_vision import maps_generation
from computer_vision.contours_scan_cython import contours_scan
from framework import viz_utils
from framework import cv_utils
# from experiments_framework.framework import ros_utils

from sensor_msgs.msg import LaserScan


rospy.init_node('video_games')
pub = rospy.Publisher('scan', LaserScan, queue_size=1) # TODO: queue_size
prev_scan_time = None

cap = cv2.VideoCapture(r'/home/omer/orchards_ws/data/lavi_apr_18/raw/dji/DJI_0167.MP4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")
n = 0
mean_ = None


while (cap.isOpened()):

    ts = time.time()

    ret, frame = cap.read()
    print ('FPS = ' + str(cap.get(cv2.CAP_PROP_POS_MSEC)))
    # contours, _ = maps_generation.extract_canopy_contours(frame)
    # cv2.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)

    vehicle_x, vehicle_y = segmentation.extract_vehicle(frame)


    # frame = cv_utils.crop_region(frame, vehicle_x, vehicle_y, 310, 310)
    # cv2.circle(frame, (155, 155), radius=3, color=(255, 0, 255), thickness=-1) # TODO: 155 is incorrect!!!!!
    # cv2.circle(frame, (vehicle_x, vehicle_y), radius=3, color=(255, 0, 255), thickness=-1)

    ## map_image = maps_generation.extract_canopies_map(frame)
    ## from experiments_framework.framework import ros_utils
    ## ros_utils.save_image_to_map(map_image, resolution=0.0125, map_name='purple_map', dir_name=r'/home/omer/Desktop')

    map_image = maps_generation.generate_canopies_map(frame)
    # map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Desktop/purple_map.pgm'), cv2.COLOR_BGR2GRAY)
    # map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Downloads/dji_15-53-1_map.pgm'), cv2.COLOR_BGR2GRAY)

    # vehicle_x = 1000
    # vehicle_y = 1000


    # _, scan_coordinates_list = contours_scan2.generate(map_image, vehicle_x, vehicle_y, 0, 2 * np.pi, 360, 3, 300, 0.0125)
    scan_ranges = contours_scan.generate(map_image,
                                         center_x=vehicle_x,
                                         center_y=vehicle_y,
                                         min_angle=0,
                                         max_angle=2 * np.pi,
                                         samples_num=360,
                                         min_distance=3,
                                         max_distance=300,
                                         resolution=0.0125,
                                         r_primary_search_samples=25,
                                         r_secondary_search_step=3)  # TODO: fine tune parameters!
    curr_scan_time = datetime.datetime.now()
    if prev_scan_time is None:
        prev_scan_time = datetime.datetime.now()
        continue
    laser_scan = LaserScan()
    laser_scan.header.stamp = rospy.rostime.Time.now()
    laser_scan.header.frame_id = 'contours_scan_link'
    laser_scan.angle_min = 0
    laser_scan.angle_max = 2 * np.pi
    laser_scan.angle_increment = (2 * np.pi - 0) / 360
    laser_scan.scan_time = (curr_scan_time - prev_scan_time).seconds
    laser_scan.range_min = 3 * 0.0125
    laser_scan.range_max = 300 * 0.0125
    laser_scan.ranges = scan_ranges
    pub.publish(laser_scan)
    prev_scan_time = curr_scan_time

    scan_coordinates_list = cv_utils.get_coordinates_list_from_scan_ranges(scan_ranges, vehicle_x, vehicle_y, 0, 2 * np.pi, 0.0125) # TODO: incorrect!!!!!!!!
    for scan_coordinate in scan_coordinates_list:
        cv2.circle(frame, (scan_coordinate[0], scan_coordinate[1]), radius=3, color=(0, 0, 255), thickness=-1)
    if ret == True:

        viz_utils.show_image('video', frame, wait_key=False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

    te = time.time()
    if n == 0:
        mean_ = te-ts
    else:
        mean_ = float(mean_) * (n - 1) / n + (te-ts) / n
    n += 1
    print (mean_)

cap.release()
cv2.destroyAllWindows()
