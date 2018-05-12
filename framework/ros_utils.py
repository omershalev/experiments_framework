import subprocess
import time
import rosbag

import utils


def start_master(use_sim_time=True):
    ros_proc = utils.new_process(['roscore'], output_to_console=False)
    time.sleep(2)
    if use_sim_time:
        param_set_proc = utils.new_process(['rosparam', 'set', 'use_sim_time', 'true'])
        param_set_proc.communicate()
    return ros_proc


def kill_all():
    for proc_name in ['roscore', 'rosmaster', 'rosout', 'slam_gmapping']:
        proc = utils.new_process(['killall', '-9', proc_name])
        proc.communicate()


def launch(package, launch_file):
    launch_proc = utils.new_process(['roslaunch', package, launch_file])
    return launch_proc


def play_bag(bag_file, use_clock=True):
    if type(bag_file) == tuple:
        bags = [rosbag.Bag(single_bag_file) for single_bag_file in bag_file]
        duration_in_seconds = sum([bag.get_end_time() - bag.get_start_time() for bag in bags])
        path_for_command_line = ' '.join(bag_file)
    else:
        bag = rosbag.Bag(bag_file)
        duration_in_seconds = bag.get_end_time() - bag.get_start_time()
        path_for_command_line = bag_file
    if use_clock:
        bag_proc = utils.new_process(['rosbag', 'play', path_for_command_line, '--clock'])
    else:
        bag_proc = utils.new_process(['rosbag', 'play', path_for_command_line])
    return bag_proc, duration_in_seconds


def save_map(map_name, dir_name):
    save_map_proc = subprocess.Popen(['rosrun', 'map_server', 'map_saver', '-f', map_name], cwd=dir_name)
    time.sleep(1)
    save_map_proc.kill()