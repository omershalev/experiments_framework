import cv2
import json

from experiments_framework.framework import ros_utils
from experiments_framework.framework import cv_utils
from experiments_framework.framework import utils
from air_ground_orchard_navigation.computer_vision import segmentation
from experiments_framework.framework import config
from experiments_framework.content.data_pointers.lavi_april_18 import dji


image_path = None # ignored if None
relevant_keys = None # if None, take all
snapshots_groups = [dji.snapshots_60_meters, dji.snapshots_80_meters]
markers_locations_json_paths = [dji.snapshots_60_meters_markers_locations_json_path, dji.snapshots_80_meters_markers_locations_json_path]


if __name__ == '__main__':

    if image_path is not None:
        raise NotImplemented
    group_idx = 1
    for snapshots_group, markers_locations_json_path in zip(snapshots_groups, markers_locations_json_paths):
        execution_dir = utils.create_new_execution_folder('map_generation_%d' % group_idx)
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
            ros_utils.save_image_to_map(map_image, resolution=config.top_view_resolution, map_name='%s_map' % key, dir_name=execution_dir)
