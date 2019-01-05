import os
import pickle
import cv2
import numpy as np
import rospy
import rosbag
import datetime
from collections import OrderedDict
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped

from content.data_pointers.lavi_april_18.jackal import jackal_18 as ugv_data
from content.data_pointers.lavi_april_18 import orchard_topology
from framework import config
from framework import cv_utils
from computer_vision import segmentation
from computer_vision import trunks_detection
from computer_vision import maps_generation
from computer_vision import calibration
from computer_vision.contours_scan_cython import contours_scan


min_angle = config.synthetic_scan_min_angle
max_angle = config.synthetic_scan_max_angle
samples_num = config.synthetic_scan_samples_num
min_distance = config.synthetic_scan_min_distance
max_distance = config.synthetic_scan_max_distance
r_primary_search_samples = config.synthetic_scan_r_primary_search_samples
r_secondary_search_step = config.synthetic_scan_r_secondary_search_step
frame_id = 'canopies/contours_scan_link'

sync_delta = 120


def generate_odometry_pickle(ugv_bag_path, output_pickle_path):
    ugv_bag = rosbag.Bag(ugv_bag_path)
    transforms = OrderedDict()
    first_timestamp = ugv_bag.get_start_time()
    experiment_start_time = rospy.Time(first_timestamp + sync_delta)
    for _, message, timestamp in ugv_bag.read_messages(topics='/tf', start_time=experiment_start_time):
        for transform in message.transforms:
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_link':
                new_stamp = transform.header.stamp.to_sec() - experiment_start_time.to_sec()
                if new_stamp < 0:
                    continue
                transforms[new_stamp] = (transform.transform.translation.x, transform.transform.translation.y)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(transforms, f)


def generate_scans_pickle(video_path, output_pickle_path, resolution):
    scans_and_ugv_poses = OrderedDict()
    prev_scan_time = None
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        raise Exception('Error opening video stream')
    frame_idx = 0
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while cap.isOpened():
        is_success, frame = cap.read()
        frame_idx += 1
        if frame_idx >= frames_count:
            break
        if not is_success:
            print 'skipped!!!!!!'
            continue
        if prev_scan_time is None:
            prev_scan_time = datetime.datetime.now()
            continue
        curr_scan_time = datetime.datetime.now()
        vehicle_pose = segmentation.extract_vehicle(frame)
        if vehicle_pose is None:
            continue
        else:
            contours_image = maps_generation.generate_canopies_map(frame)
            scan_ranges = contours_scan.generate(contours_image,
                                                 center_x=vehicle_pose[0],
                                                 center_y=vehicle_pose[1],
                                                 min_angle=min_angle,
                                                 max_angle=max_angle,
                                                 samples_num=samples_num,
                                                 min_distance=min_distance,
                                                 max_distance=max_distance,
                                                 resolution=resolution,
                                                 r_primary_search_samples=r_primary_search_samples,
                                                 r_secondary_search_step=r_secondary_search_step)
            video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3

            laser_scan = LaserScan()
            laser_scan.header.stamp = rospy.rostime.Time.from_sec(video_timestamp)
            laser_scan.header.frame_id = frame_id
            laser_scan.header.seq = frame_idx
            laser_scan.angle_min = min_angle
            laser_scan.angle_max = max_angle
            laser_scan.angle_increment = (max_angle - min_angle) / samples_num
            laser_scan.scan_time = (curr_scan_time - prev_scan_time).seconds
            laser_scan.range_min = min_distance * resolution
            laser_scan.range_max = max_distance * resolution
            laser_scan.ranges = np.asanyarray(scan_ranges)
            prev_scan_time = curr_scan_time

            vehicle_pose_point = PointStamped()
            vehicle_pose_point.point.x = vehicle_pose[0]
            vehicle_pose_point.point.y = vehicle_pose[1]
            vehicle_pose_point.header.stamp = rospy.rostime.Time.from_sec(video_timestamp)
            vehicle_pose_point.header.seq = frame_idx

            scans_and_ugv_poses[video_timestamp] = (laser_scan, vehicle_pose_point)
            print video_timestamp

    cap.release()
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(scans_and_ugv_poses, f)


def get_image_from_video(video_path, time_in_msec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_msec)
    is_success, frame = cap.read()
    if is_success:
        return frame
    return None


def mark_dimensions(video_path, output_pickle_path):
    points_sets = []
    for t in np.linspace(start=20, stop=600, num=10):
        image = get_image_from_video(video_path, time_in_msec=t * 1e3)
        points_sets.append(cv_utils.sample_pixel_coordinates(image, multiple=True))
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(points_sets, f)


def sample_video(video_path, output_path, start_time, stop_time, samples):
    for t in np.linspace(start_time, stop_time, samples):
        image = get_image_from_video(video_path, time_in_msec=t * 1e3)
        cv2.imwrite(os.path.join(output_path, '%d.jpg' % int(t)), image)

def estimate_resolution(images_dir):
    grid_dim_x_values = []
    grid_dim_y_values = []
    for image_filename in os.listdir(images_dir):
        image = cv2.imread(os.path.join(images_dir, image_filename))
        orientation = trunks_detection.estimate_rows_orientation(image)
        _, rotated_centroids, _, _ = trunks_detection.find_tree_centroids(image, correction_angle=orientation * (-1))
        grid_dim_x, grid_dim_y = trunks_detection.estimate_grid_dimensions(rotated_centroids)
        grid_dim_x_values.append(grid_dim_x)
        grid_dim_y_values.append(grid_dim_y)
    return 1.0 / calibration.calculate_pixel_to_meter(np.mean(grid_dim_x_values), np.mean(grid_dim_y_values),
                                                      orchard_topology.measured_row_widths,
                                                      orchard_topology.measured_intra_row_distances)


# def estimate_resolution(points_pickle_path):
#     with open(points_pickle_path) as f:
#         points_sets = pickle.load(f)
#     dim_x = []
#     dim_y = []
#     for set in points_sets:
#         if len(set) == 0:
#             continue
#         dim_x.append(abs(set[1][0] - set[2][0]))
#         dim_y.append(abs(set[0][1] - set[1][1]))
#         # dim_x.append(abs(set[1][1] - set[2][1]))
#         # dim_y.append(abs(set[0][0] - set[1][0]))
#     mean_dim_x = np.mean(dim_x)
#     mean_dim_y = np.mean(dim_y)
#     return 1.0 / calibration.calculate_pixel_to_meter(mean_dim_x, mean_dim_y, orchard_topology.measured_row_widths, orchard_topology.measured_intra_row_distances)

if __name__ == '__main__':
    ugv_bag_path = ugv_data['17-45-36'].path
    video_path = r'/home/omer/orchards_ws/resources/lavi_apr_18/raw/dji/DJI_0167.MP4'
    odometry_pickle_path = r'/home/omer/Downloads/ugv_odometry.pkl'
    scans_and_ugv_poses_pickle_path = r'/home/omer/Downloads/scans.pkl'
    points_pickle_path = r'/home/omer/Downloads/points.pkl'
    video_samples_path = r'/home/omer/Downloads/video_samples'

    # generate_odometry_pickle(ugv_bag_path ,odometry_pickle_path)
    # sample_video(video_path, video_samples_path, start_time=20, stop_time=600, samples=50)
    resolution = estimate_resolution(video_samples_path)
    generate_scans_pickle(video_path, scans_and_ugv_poses_pickle_path, resolution)

'''
The Jackal begins movement on second 123 in the bag file
The DJI begins movement on second 3 in the video
==> 120 delay 
'''

# mark_dimensions(video_path, points_pickle_path)
# print estimate_resolution(points_pickle_path)
