import cv2
import json

from framework import cv_utils
from framework import viz_utils
from computer_vision import maps_generation
from content.data_pointers.lavi_april_18 import dji

if __name__ == '__main__':
    with open(dji.snapshots_60_meters_markers_locations_json_path) as f:
        markers_locations = json.load(f)
    for key, data_descriptor in dji.snapshots_60_meters.items():
        image = cv2.imread(data_descriptor.path)
        map_image = maps_generation.generate_canopies_map(image)
        map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
        map_image = cv_utils.mark_rectangle_on_image(map_image, markers_locations[key])
        cv_utils.mark_bounding_box(map_image, markers_locations[key], expand_ratio=0.1)
        viz_utils.show_image(key, map_image)