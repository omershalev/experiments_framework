import time
import datetime

from framework.experiment import Experiment
import framework.ros_utils as ros_utils
import framework.utils as utils

class GmappingExperiment(Experiment):

    def clean_env(self):
        utils.kill_process('slam_gmapping')
        ros_utils.kill_master()


    def task(self, **kwargs):
        ros_utils.start_master(use_sim_time=True)
        ros_utils.launch(package='jackal_navigation', launch_file='gmapping.launch')
        _, bag_duration = ros_utils.play_bag(self.data_sources, use_clock=True)
        start_time = datetime.datetime.now()
        while True:
            if (datetime.datetime.now() - start_time).seconds > (bag_duration + 1):
                break
            if kwargs.get('periodic_map_saving'):
                ros_utils.save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S'), dir_name=self.repetition_dir)
            time.sleep(3)
        ros_utils.save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S-final'), dir_name=self.repetition_dir) # TODO: verify that this works

if __name__ == '__main__':

    import experiments_framework.content.data_pointers.lavi_april_18.jackal as jackal_data

    execution_dir = utils.create_new_execution_folder('gmapping_jackal_18')
    for bag_name, bag_descriptor in jackal_data.forks: # TODO: check why this doesn't run for 18-24-26
        experiment = GmappingExperiment(name='gmapping on %s' % bag_name, data_sources=bag_descriptor.path, working_dir=execution_dir)
        experiment.run(repetitions=1, periodic_map_saving=False)