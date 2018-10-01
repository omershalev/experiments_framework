import os
import cv2
import rospy
import rosbag

from computer_vision import segmentation
from computer_vision import maps_generation
from computer_vision import object_tracking
from computer_vision.contours_scan_cython import contours_scan
from framework import utils
from framework import cv_utils
from framework import config
from cv_bridge import CvBridge

from sensor_msgs.msg import LaserScan

video_path = r'/home/omer/orchards_ws/resources/lavi_apr_18/raw/dji/DJI_0167.MP4'

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_video')
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        raise Exception('Error opening video stream')
    min_angle = config.synthetic_scan_min_angle
    max_angle = config.synthetic_scan_max_angle
    samples_num = config.synthetic_scan_samples_num
    min_distance = config.synthetic_scan_min_distance
    max_distance = config.synthetic_scan_max_distance
    resolution = config.top_view_resolution
    r_primary_search_samples = config.synthetic_scan_r_primary_search_samples
    r_secondary_search_step = config.synthetic_scan_r_secondary_search_step
    frame_id = 'canopies/contours_scan_link'
    frame_idx = 1
    bridge = CvBridge()

    bag = rosbag.Bag(os.path.join(execution_dir, 'scan.bag'), 'w') # TODO: change to experiment_dir
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        vehicle_pose = segmentation.extract_vehicle(frame)
        if frame_idx == 1:
            tracker = object_tracking.PointMassTracker(frequency=config.target_system_frequency,
                                                       transition_variances=(1e-5, 1e-5, 1e-3, 1e-3),
                                                       observation_variances=(1e-9, 1e-9),
                                                       init_pose=(vehicle_pose[0], vehicle_pose[1]),
                                                       init_variances=(1e-3, 1e-3, 1e-3, 1e-3),
                                                       change_thresholds=(20, 20))
            vehicle_x, vehicle_y = vehicle_pose[0], vehicle_pose[1]
        else:
            filtered_pose = tracker.update_and_get_estimation(vehicle_pose)
            # vehicle_x, vehicle_y = filtered_pose[0], filtered_pose[1]
            if vehicle_pose is None: # TODO: temp
                continue # TODO: temp
            vehicle_x, vehicle_y = vehicle_pose[0], vehicle_pose[1] # TODO: temp
        contours_image = maps_generation.generate_canopies_map(frame)
        scan_ranges = contours_scan.generate(contours_image,
                                             center_x=vehicle_x,
                                             center_y=vehicle_y,
                                             min_angle=min_angle,
                                             max_angle=max_angle,
                                             samples_num=samples_num,
                                             min_distance=min_distance,
                                             max_distance=max_distance,
                                             resolution=resolution,
                                             r_primary_search_samples=r_primary_search_samples,
                                             r_secondary_search_step=r_secondary_search_step)

        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3
        ros_timestamp = rospy.rostime.Time.from_sec(video_timestamp)
        print video_timestamp
        # if video_timestamp > 60:
        #     break

        # scan_points_list = cv_utils.get_coordinates_list_from_scan_ranges(scan_ranges, vehicle_x, vehicle_y, min_angle, max_angle, resolution)
        # visualization_image = cv_utils.draw_points_on_image(frame, scan_points_list, color=(0, 0, 255), radius=5)
        # visualization_image = cv2.circle(visualization_image, center=(int(vehicle_x), int(vehicle_y)), radius=25, color=(255, 0, 255), thickness=5)
        # image_message = bridge.cv2_to_imgmsg(visualization_image, encoding='bgr8')
        # image_message.header.seq = frame_idx
        # image_message.header.stamp = ros_timestamp
        # image_message.header.frame_id = frame_id

        scan_message = LaserScan()
        scan_message.header.seq = frame_idx
        scan_message.header.stamp = ros_timestamp
        scan_message.header.frame_id = frame_id
        scan_message.angle_min = min_angle
        scan_message.angle_max = max_angle
        scan_message.angle_increment = (max_angle - min_angle) / samples_num
        scan_message.scan_time = 1.0 / config.target_system_frequency # TODO: think
        scan_message.range_min = min_distance * resolution
        scan_message.range_max = max_distance * resolution
        scan_message.ranges = scan_ranges
        frame_idx += 1
        bag.write(topic='/canopies/scan', msg=scan_message, t=ros_timestamp)
        # bag.write(topic='/scan_visualization', msg=image_message, t=ros_timestamp)
    bag.close()
    cap.release()