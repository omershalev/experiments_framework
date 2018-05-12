import os
import subprocess
from threading import Timer
import time
import datetime
import rosbag


def _run_timeout_protected_process(command, timeout_in_seconds):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = Timer(timeout_in_seconds, lambda proc: proc.kill(), [proc])
    try:
        timer.start()
        while proc.poll() is None:
            pass
        output = proc.stdout.read() if proc.stdout is not None else None
        error = proc.stderr.read() if proc.stderr is not None else None
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, ' '.join(command), output)
    finally:
        timer.cancel()
    return output, error

def _new_process(command, cwd=None, output_to_console=False):
    if output_to_console:
        proc = subprocess.Popen(command, cwd=cwd)
    else:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    return proc

def ros_start(use_sim_time=True):
    ros_proc = _new_process(['roscore'], output_to_console=False)
    time.sleep(2)
    if use_sim_time:
        param_set_proc = _new_process(['rosparam', 'set', 'use_sim_time', 'true'], output_to_console=False)
        param_set_proc.communicate()
    return ros_proc

def ros_kill_all():
    for proc_name in ['roscore', 'rosmaster', 'rosout', 'slam_gmapping']:
        proc = _new_process(['killall', '-9', proc_name])
        proc.communicate()


def ros_launch(package, launch_file):
    launch_proc = _new_process(['roslaunch', package, launch_file], output_to_console=True)
    return launch_proc

def ros_bag_play(bag_file, use_clock=True):
    if type(bag_file) == tuple:
        bags = [rosbag.Bag(single_bag_file) for single_bag_file in bag_file]
        duration_in_seconds = sum([bag.get_end_time() - bag.get_start_time() for bag in bags])
        path_for_command_line = ' '.join(bag_file)
    else:
        bag = rosbag.Bag(bag_file)
        duration_in_seconds = bag.get_end_time() - bag.get_start_time()
        path_for_command_line = bag_file
    if use_clock:
        bag_proc = _new_process(['rosbag', 'play', path_for_command_line, '--clock'], output_to_console=True)
    else:
        bag_proc = _new_process(['rosbag', 'play', path_for_command_line], output_to_console=True)
    return bag_proc, duration_in_seconds

def ros_save_map(map_name, dir_name):
    save_map_proc = subprocess.Popen(['rosrun', 'map_server', 'map_saver', '-f', map_name], cwd=dir_name)
    time.sleep(1)
    save_map_proc.kill()