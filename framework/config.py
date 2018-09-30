import os
import numpy as np

############################################# PATHS #############################################
root_dir_path = r'/home/omer/orchards_ws'
base_raw_data_path = os.path.join(root_dir_path, 'resources/lavi_apr_18/raw')
panorama_path = os.path.join(root_dir_path, 'resources/lavi_apr_18/panorama')
markers_locations_path = os.path.join(root_dir_path, 'resources/lavi_apr_18/markers_locations')
base_results_path = os.path.join(root_dir_path, 'results')
temp_output_path = r'/home/omer/temp'

output_to_console = True

screen_resolution = (1920, 1080)

##################################### BEST KNOWN PARAMETERS #####################################
bounding_box_expand_ratio = 0.15 # TODO: this was changed from 0.15 - is that okay?
target_system_frequency = 30 # TODO: consider 25...
top_view_resolution = 0.0125 # TODO: eventually, this should be removed
synthetic_scan_min_angle = -np.pi
synthetic_scan_max_angle = np.pi
synthetic_scan_samples_num = 360
synthetic_scan_min_distance = 3 # TODO: this is in pixels - problematic!
synthetic_scan_max_distance = 300 # TODO: this is in pixels - problematic!; consider taking 350 pixels instead!!!!!
# synthetic_scan_r_primary_search_samples = 50
# synthetic_scan_r_secondary_search_step = 2
synthetic_scan_r_primary_search_samples = 35
synthetic_scan_r_secondary_search_step = 2
trunk_dilation_ratio = 1
trunk_std_increasing_factor = 10 # TODO: think
cost_map_gaussians_scale_factor = 0.9
scale_factor = 2.2 # TODO: choose better name + use for error calculation!!!