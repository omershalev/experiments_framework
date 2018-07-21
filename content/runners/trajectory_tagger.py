import cv2
import json

from experiments_framework.framework import cv_utils
from experiments_framework.framework import viz_utils
from experiments_framework.framework import ros_utils
from air_ground_orchard_navigation.computer_vision import segmentation
from experiments_framework.content.data_pointers.lavi_april_18 import dji

if __name__ == '__main__':
    # key = '19-03-5'
    # data_descriptor = dji.snapshots_60_meters[key]
    # image = cv2.imread(data_descriptor.path)
    # map_image = segmentation.extract_canopies_map(image)
    # pose_time_tuples_list = cv_utils.mark_trajectory_on_image(map_image)
    # ros_utils.trajectory_to_bag(pose_time_tuples_list, bag_path=r'/home/omer/Downloads/%s_simple_trajectory_4.bag' % key)


    key = '19-03-5'
    data_descriptor = dji.snapshots_60_meters[key]
    import json
    with open(dji.snapshots_60_meters_markers_locations_json) as f:
        markers_locations = json.load(f)
    points = markers_locations[key]

    image = cv2.imread(data_descriptor.path)
    map_image = segmentation.extract_canopies_map(image)
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = cv_utils.get_bounding_box(map_image, points, expand_ratio=0.1)
    map_image = map_image[upper_left_y:lower_right_y, upper_left_x:lower_right_x]
    pose_time_tuples_list = cv_utils.mark_trajectory_on_image(map_image)
    ros_utils.trajectory_to_bag(pose_time_tuples_list, bag_path=r'/home/omer/Downloads/%s_simple_trajectory_12.bag' % key)