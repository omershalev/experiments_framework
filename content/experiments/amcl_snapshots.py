import time
import os

from experiments_framework.framework.experiment import Experiment
from experiments_framework.framework import ros_utils
from experiments_framework.framework import utils
from experiments_framework.framework import config

class AmclSnapshotsExperiment(Experiment):

    def clean_env(self):
        utils.kill_process('amcl')
        ros_utils.kill_master()


    def task(self, **kwargs):

        launch_rviz = kwargs.get('launch_rviz', False)

        ros_utils.start_master()
        if launch_rviz:
            ros_utils.launch_rviz(os.path.join(config.root_dir_path, 'src/experiments_framework/framework/amcl.rviz'))

        # Launch base_link to contours_scan_link static TF
        ros_utils.launch(package='localization',
                         launch_file='static_identity_tf.launch',
                         argv={'frame_id': 'base_link', 'child_frame_id': 'contours_scan_link'})

        # Launch synthetic scan generator
        ros_utils.launch(package='localization',
                         launch_file='synthetic_scan_generator.launch',
                         argv={'localization_image_path': self.data_sources['localization_image_path', 'resolution': config.top_view_resolution]})

        # Launch map server
        ros_utils.launch(package='localization',
                         launch_file='map.launch',
                         argv={'map_yaml_path': self.data_sources['map_yaml_path']})

        # Wait for map server to load
        ros_utils.wait_for_rosout_message(node_name='map_server',
                                          desired_message=r'Read a \d+ X \d+ map @ \d\.?\d+? m/cell',
                                          is_regex=True)

        # Launch AMCL
        ros_utils.launch(package='localization', launch_file='amcl.launch')

        # Wait for AMCL to load
        ros_utils.wait_for_rosout_message(node_name='amcl', desired_message='Done initializing likelihood field model.')

        # Launch ICP
        ros_utils.launch(package='localization', launch_file='icp.launch')

        # Start recording output bag
        self.results['output_bag'] = os.path.join(self.repetition_dir, self.name)
        ros_utils.start_recording_bag(os.path.join(self.repetition_dir, self.name, 'output'), ['/amcl_pose', '/particlecloud', '/scanmatcher_pose', '/vehicle_pose'])

        # Start input bag and wait
        _, bag_duration = ros_utils.play_bag(self.data_sources['trajectory_bag_path'])
        time.sleep(bag_duration)

        # Stop recording output bag
        ros_utils.stop_recording_bags()

        if launch_rviz:
            ros_utils.kill_rviz()


if __name__ == '__main__':
    experiment = AmclSnapshotsExperiment(name='amcl_exploration',
                                         data_sources={'trajectory_bag_path': r'/home/omer/Downloads/15-53-1_random_trajectory.bag',
                                              'localization_image_path': r'/home/omer/Downloads/dji_15-53-1_map.pgm',
                                              'map_yaml_path': r'/home/omer/Downloads/dji_15-53-1_map.yaml'},
                                         working_dir=r'/home/omer/Downloads')
    experiment.run(repetitions=1, launch_rviz=True)