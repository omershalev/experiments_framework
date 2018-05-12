import time
import datetime

from experiments_framework.framework.experiment import Experiment
import experiments_framework.framework.ros_utils as ros_utils
import experiments_framework.framework.utils as utils
import experiments_framework.content.jackal_data as jackal_data

class GmappingExperiment(Experiment):

    def cleanup(self):
        ros_utils.kill_all()

    def task(self):
        ros_utils.start_master(use_sim_time=True)
        ros_utils.launch('jackal_navigation', 'gmapping.launch')
        _, bag_duration = ros_utils.play_bag(self.data_sources, use_clock=True)
        start_time = datetime.datetime.now()
        while True:
            if (datetime.datetime.now() - start_time).seconds > (bag_duration + 1):
                break
            ros_utils.save_map(map_name=datetime.datetime.now().strftime('map_%H_%M_%S'), dir_name=self.repetition_dir)
            time.sleep(3)


if __name__ == '__main__':

    # _, bag_duration = utils.ros_bag_play(r'/home/omer/orchards_ws/data/jackal_18/2018-04-24-18-30-40_2.bag', use_clock=True)
    # p2 = subprocess.Popen(['rosbag', 'play', '/home/omer/orchards_ws/data/jackal_18/2018-04-04-17-26-09_0.bag'])


    execution_dir = utils.create_new_execution_folder('gmapping_jackal_18')
    for bag_file in jackal_data.jackal_18:
        experiment = GmappingExperiment(name='gmapping', data_sources=bag_file, working_dir=execution_dir)
        experiment.run(1)