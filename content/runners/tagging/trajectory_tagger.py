import os
import cv2
import json

from framework import cv_utils
from framework import utils
from framework import ros_utils
from computer_vision import segmentation
from content.data_pointers.lavi_april_18 import dji
from framework import config
from framework import logger

_logger = logger.get_logger()

image_path = None # ignored if None
# relevant_keys = ['15-20-1', '16-54-1', '19-03-1', '15-08-1', '15-53-1', '16-55-1', '19-04-1'] # if None, take all
relevant_keys = ['15-53-1'] # if None, take all
snapshots_groups = [dji.snapshots_60_meters, dji.snapshots_80_meters]
markers_locations_json_paths = [dji.snapshots_60_meters_markers_locations_json_path, dji.snapshots_80_meters_markers_locations_json_path]
fork_shaped = False
S_shaped = False
random_shaped = True


if __name__ == '__main__':

    if image_path is not None:
        raise NotImplemented
    group_idx = 1
    for snapshots_group, markers_locations_json_path in zip(snapshots_groups, markers_locations_json_paths):
        execution_dir = utils.create_new_execution_folder('trajectory_tagging_%d' % group_idx)
        group_idx += 1
        with open(markers_locations_json_path) as f:
            markers_locations = json.load(f)
        for key, data_descriptor in snapshots_group.items():
            if relevant_keys is not None:
                if key not in relevant_keys:
                    continue
            points = markers_locations[key]
            image = cv2.imread(data_descriptor.path)
            map_image = segmentation.extract_canopies_map(image)
            (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = cv_utils.get_bounding_box(map_image, points, expand_ratio=config.markers_bounding_box_expand_ratio)
            map_image = map_image[upper_left_y:lower_right_y, upper_left_x:lower_right_x]

            if fork_shaped:
                _logger.info('Mark fork-shaped trajectory')
                pose_time_tuples_list = cv_utils.mark_trajectory_on_image(map_image)
                ros_utils.trajectory_to_bag(pose_time_tuples_list, bag_path=os.path.join(execution_dir, '%s_fork_trajectory.bag' % key))
                ros_utils.downsample_bag(input_bag_path=os.path.join(execution_dir, '%s_fork_trajectory.bag' % key),
                                         topic='ugv_pose',
                                         target_frequency=config.synthetic_scan_target_frequency)

            if S_shaped:
                _logger.info('Mark S-shaped trajectory')
                pose_time_tuples_list = cv_utils.mark_trajectory_on_image(map_image)
                ros_utils.trajectory_to_bag(pose_time_tuples_list, bag_path=os.path.join(execution_dir, '%s_S_trajectory.bag' % key))
                ros_utils.downsample_bag(input_bag_path=os.path.join(execution_dir, '%s_S_trajectory.bag' % key),
                                         topic='ugv_pose',
                                         target_frequency=config.synthetic_scan_target_frequency)

            if random_shaped:
                _logger.info('Mark random-shaped trajectory')
                pose_time_tuples_list = cv_utils.mark_trajectory_on_image(map_image)
                ros_utils.trajectory_to_bag(pose_time_tuples_list, bag_path=os.path.join(execution_dir, '%s_random_trajectory.bag' % key))
                ros_utils.downsample_bag(input_bag_path=os.path.join(execution_dir, '%s_random_trajectory.bag' % key),
                                         topic='ugv_pose',
                                         target_frequency=config.synthetic_scan_target_frequency)
                print pose_time_tuples_list
                cv2.imwrite(r'/home/omer/Documents/another2.png', map_image)
            break