import os
import cv2
import numpy as np
import pickle

from framework import cv_utils
from computer_vision import trunks_detection
from computer_vision import calibration
from content.data_pointers.lavi_april_18 import base_resources_path, base_raw_data_path
from content.data_pointers.lavi_april_18 import orchard_topology


video_path = os.path.join(base_raw_data_path, 'dji', 'DJI_0167.MP4')
pickle_path = os.path.join(base_resources_path, 'amcl_video', 'scans_and_poses.pkl')


def estimate_resolution(images_dir):
    grid_dim_x_values = []
    grid_dim_y_values = []
    for image_filename in os.listdir(images_dir):
        image = cv2.imread(os.path.join(images_dir, image_filename))
        orientation, _, _ = trunks_detection.estimate_rows_orientation(image)
        try:
            _, rotated_centroids, _, _, _ = trunks_detection.find_tree_centroids(image, correction_angle=orientation * (-1))
            grid_dim_x, grid_dim_y = trunks_detection.estimate_grid_dimensions(rotated_centroids)
        except Exception, e:
            grid_dim_x, grid_dim_y = np.nan, np.nan
        grid_dim_x_values.append(grid_dim_x)
        grid_dim_y_values.append(grid_dim_y)
    return 1.0 / calibration.calculate_pixel_to_meter(np.nanmean(grid_dim_x_values), np.nanmean(grid_dim_y_values),
                                                      orchard_topology.measured_row_widths,
                                                      orchard_topology.measured_intra_row_distances)


if __name__ == '__main__':
    video_samples_path = os.path.join(base_resources_path, 'amcl_video', 'video_samples')
    # resolution = estimate_resolution(video_samples_path)
    resolution = 0.0156892495353
    with open(pickle_path, 'rb') as f:
        scans_and_ugv_poses = pickle.load(f)
    cap = cv2.VideoCapture(video_path)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter()
    out.open(r'/home/omer/Documents/tracking_output_avc1.mp4', fourcc, 30.0, sz, True)
    if not cap.isOpened():
        raise Exception('Error opening video stream')
    frame_idx = 0
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while cap.isOpened():
        print (frame_idx)
        is_success, frame = cap.read()
        frame_idx += 1
        if frame_idx >= frames_count:
            break
        if not is_success:
            continue
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3
        if video_timestamp not in scans_and_ugv_poses.keys():
            continue
        laser_scan, vehicle_pose_point = scans_and_ugv_poses[video_timestamp]
        x = vehicle_pose_point.point.x
        y = vehicle_pose_point.point.y
        cv2.circle(frame, (x, y), radius=19, color=(255, 0, 255), thickness=4)
        coordinates_list = cv_utils.get_coordinates_list_from_scan_ranges(laser_scan.ranges, x, y, min_angle=-np.pi, max_angle=np.pi, resolution=resolution)
        frame = cv_utils.draw_points_on_image(frame, coordinates_list, color=(0, 0, 255), radius=4)
        out.write(frame)
        window_name = 'vid'
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
        # if frame_idx > 3000:
        #     break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
