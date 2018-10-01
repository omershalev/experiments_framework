import pickle
import cv2
import numpy as np
import rospy
import rosbag
from collections import OrderedDict

from content.data_pointers.lavi_april_18.jackal import jackal_18 as ugv_data
from content.data_pointers.lavi_april_18 import orchard_topology
from framework import config
from framework import cv_utils

from computer_vision import segmentation
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
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        raise Exception('Error opening video stream')
    while cap.isOpened():
        is_success, frame = cap.read()
        if not is_success:
            continue
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
            scans_and_ugv_poses[video_timestamp] = (scan_ranges, vehicle_pose)
    cap.release()
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(scans_and_ugv_poses, f)


def get_image_at_specific_time(video_path, time_in_msec):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_msec)
    is_success, frame = cap.read()
    if is_success:
        return frame
    return None


def mark_dimensions(video_path, output_pickle_path):
    points_sets = []
    for t in np.linspace(start=20, stop=600, num=10):
        image = get_image_at_specific_time(video_path, time_in_msec=t * 1e3)
        points_sets.append(cv_utils.sample_pixel_coordinates(image, multiple=True))
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(points_sets, f)


def estimate_resolution(points_pickle_path):
    with open(points_pickle_path) as f:
        points_sets = pickle.load(f)
    dim_x = []
    dim_y = []
    for set in points_sets:
        if len(set) == 0:
            continue
        dim_x.append(abs(set[1][0] - set[2][0]))
        dim_y.append(abs(set[0][1] - set[1][1]))
        # dim_x.append(abs(set[1][1] - set[2][1]))
        # dim_y.append(abs(set[0][0] - set[1][0]))
    mean_dim_x = np.mean(dim_x)
    mean_dim_y = np.mean(dim_y)
    return 1.0 / calibration.calculate_pixel_to_meter(mean_dim_x, mean_dim_y, orchard_topology.measured_row_widths, orchard_topology.measured_intra_row_distances)

if __name__ == '__main__':
    ugv_bag_path = ugv_data['17-45-36'].path
    video_path = r'/home/omer/orchards_ws/resources/lavi_apr_18/raw/dji/DJI_0167.MP4'
    odometry_pickle_path = r'/home/omer/Downloads/ugv_odometry.pkl'
    scans_and_ugv_poses_pickle_path = r'/home/omer/Downloads/scans.pkl'
    points_pickle_path = r'/home/omer/Downloads/points.pkl'

    # generate_odometry_pickle(ugv_bag_path ,odometry_pickle_path)
    # mark_dimensions(video_path, points_pickle_path)
    resolution = estimate_resolution(points_pickle_path)
    generate_scans_pickle(video_path, scans_and_ugv_poses_pickle_path, resolution)

'''
The Jackal begins movement on second 123 in the bag file
The DJI begins movement on second 3 in the video
==> 120 delay 
'''