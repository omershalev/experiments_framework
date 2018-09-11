import os
import cv2
import json

from experiments_framework.framework import cv_utils
from experiments_framework.framework import utils
from experiments_framework.framework import viz_utils
from air_ground_orchard_navigation.computer_vision import segmentation
from experiments_framework.content.data_pointers.lavi_april_18 import dji
from experiments_framework.framework import config
from experiments_framework.framework import logger

_logger = logger.get_logger()

image_path = None # ignored if None
relevant_keys = None # if None, take all
snapshots_groups = [dji.snapshots_60_meters, dji.snapshots_80_meters]
markers_locations_json_paths = [dji.snapshots_60_meters_markers_locations_json_path, dji.snapshots_80_meters_markers_locations_json_path]
fork_shaped = True
S_shaped = True
random_shaped = True


if __name__ == '__main__':

    if image_path is not None:
        raise NotImplemented
    group_idx = 1
    for snapshots_group, markers_locations_json_path in zip(snapshots_groups, markers_locations_json_paths):
        execution_dir = utils.create_new_execution_folder('trunks_tagging_%d' % group_idx)
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
            trunk_poses = cv_utils.sample_pixel_coordinates(map_image, multiple=True)
            for pose in trunk_poses:
                cv2.circle(map_image, pose, radius=15, color=0, thickness=-1)
            viz_utils.show_image('trunk_locations', map_image)
            with open(os.path.join(execution_dir, '%s_trunks.json' % key), 'w') as f:
                json.dump(trunk_poses, f, indent=4)
