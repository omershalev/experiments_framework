import time
import json
import os
import cv2

from framework.experiment import Experiment
from framework import ros_utils
from framework import cv_utils
from framework import utils
from framework import config
from computer_vision import maps_generation

class AmclVideoExperiment(Experiment):

    def _launch_base_to_scan_static_tf(self, namespace):
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'ns': namespace, 'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

    def _launch_map_server(self, map_yaml_path, namespace):
        ros_utils.launch(package='localization',
                         launch_file='map.launch',
                         argv={'ns': namespace, 'map_yaml_path': map_yaml_path})
        ros_utils.wait_for_rosout_message(node_name='%s/map_server' % namespace,
                                          desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                          is_regex=True)

    def _launch_amcl(self, namespace):
        ros_utils.launch(package='localization', launch_file='amcl.launch', argv={'ns': namespace})
        ros_utils.wait_for_rosout_message(node_name='%s/amcl' % namespace, desired_message='Done initializing likelihood field model.')

    def _launch_icp(self, namespace):
        ros_utils.launch(package='localization', launch_file='icp.launch', argv={'ns': namespace})

    def clean_env(self):
        utils.kill_process('amcl')
        ros_utils.kill_master()

    def task(self, **kwargs):

        launch_rviz = kwargs.get('launch_rviz', False)

        origin_map_image_path = self.data_sources['map_image_path']
        map_semantic_trunks = self.data_sources['map_semantic_trunks']
        scan_bag_path = self.data_sources['scan_bag_path']
        bounding_box_expand_ratio = self.params['bounding_box_expand_ratio']


        # Generate canopies and trunk map images
        map_image = cv2.imread(origin_map_image_path)
        canopies_map_image = maps_generation.generate_canopies_map(map_image)
        upper_left, lower_right = cv_utils.get_bounding_box(canopies_map_image, map_semantic_trunks.values(), expand_ratio=bounding_box_expand_ratio)
        canopies_map_image = canopies_map_image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

        canopies_map_yaml_path, _ = ros_utils.save_image_to_map(canopies_map_image, resolution=self.params['resolution'], map_name='canopies_map', dir_name=self.repetition_dir)
        cv2.imwrite(os.path.join(self.repetition_dir, 'map_image.jpg'), map_image)

        # Start ROS and RVIZ
        ros_utils.start_master()
        if launch_rviz:
            ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'src/experiments_framework/framework/amcl.rviz'))

        # Launch localization stack for canopies and trunks
        self._launch_base_to_scan_static_tf(namespace='canopies')
        self._launch_map_server(canopies_map_yaml_path, namespace='canopies')
        self._launch_amcl(namespace='canopies')
        self._launch_icp(namespace='canopies')

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(scan_bag_path) # TODO: need to read from data_sources
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        # Kill RVIZ
        if launch_rviz:
            ros_utils.kill_rviz()

if __name__ == '__main__':
    from content.data_pointers.lavi_april_18.dji import selected_trunks_detection_experiments_and_repetitions
    from content.data_pointers.lavi_april_18.dji import trunks_detection_results_dir as td_results_dir
    td_exp_name, td_exp_rep = selected_trunks_detection_experiments_and_repetitions[0]
    with open(os.path.join(td_results_dir, td_exp_name, 'experiment_summary.json')) as f:
        td_summary = json.load(f)
    image_path = td_summary['data_sources']
    map_semantic_trunks = td_summary['results'][str(td_exp_rep)]['semantic_trunks']
    experiment = AmclVideoExperiment(name='amcl_video',
                                     data_sources={'map_image_path': image_path, 'scan_bag_path': r'/home/omer/temp/20180929-223317_amcl_video_keep/scan.bag',
                                                   'map_semantic_trunks': map_semantic_trunks},
                                     params={'resolution': 0.0225, 'bounding_box_expand_ratio': config.bounding_box_expand_ratio},
                                     working_dir=r'/home/omer/temp')
    experiment.run(repetitions=1, launch_rviz=True)