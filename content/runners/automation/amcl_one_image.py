from experiments_framework.framework.ros_utils import kill_master
from experiments_framework.framework import utils
from experiments_framework.content.experiments.amcl_snapshots import AmclSnapshotsExperiment

from experiments_framework.content.data_pointers.lavi_april_18 import dji

if __name__ == '__main__':
    execution_dir = utils.create_new_execution_folder('amcl_one_image')
    for image_path in 'TBD':
        for bag_name, bag_descriptor in 'TBD':
            experiment = AmclSnapshotsExperiment(name='amcl_exploration',
                                                 data_sources={'trajectory_bag_path': r'/home/omer/Downloads/15-53-1_simple_trajectory_1.bag',
                                                      'localization_image_path': r'/home/omer/Downloads/dji_15-53-1_map.pgm',
                                                      'map_yaml_path': r'/home/omer/Downloads/dji_15-08-1_map.yaml'},
                                                 working_dir=r'/home/omer/Downloads')
            experiment.run(repetitions=1, launch_rviz=False)

