import os
import numpy as np

#################################################################################################
#                                              ENV                                              #
#################################################################################################
root_dir_path = r'/home/omer/orchards_ws'
base_results_path = os.path.join(root_dir_path, 'results')
temp_output_path = r'/home/omer/temp'
output_to_console = True
screen_resolution = (1920, 1080)


#################################################################################################
#                                             PARAMS                                            #
#################################################################################################
bounding_box_expand_ratio = 0.15
target_system_frequency = 30
top_view_resolution = 0.0125 # TODO: eventually, this should be removed
synthetic_scan_min_angle = -np.pi
synthetic_scan_max_angle = np.pi
synthetic_scan_samples_num = 360
synthetic_scan_min_distance = 3
synthetic_scan_max_distance = 250
synthetic_scan_r_primary_search_samples = 50
synthetic_scan_r_secondary_search_step = 2
trunk_dilation_ratio = 1
trunk_std_increasing_factor = 2 # TODO: think