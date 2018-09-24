import numpy as np
import pandas as pd
import yaml
import os
import time
import re
import cv2
from collections import OrderedDict
from multiprocessing import Process
import rospy
import rosbag
import rosnode
from scipy.signal import resample
from geometry_msgs.msg import Pose2D
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import utils
import logger

DEBUG_ROSOUT = False

_logger = logger.get_logger()


def start_master(use_sim_time=True):
    _logger.info('Launching ROS master')
    ros_proc = utils.new_process(['roscore'], output_to_console=False)
    time.sleep(2)
    if use_sim_time:
        param_set_proc = utils.new_process(['rosparam', 'set', 'use_sim_time', 'true'])
        param_set_proc.communicate()
    return ros_proc


def kill_master():
    _logger.info('Killing all live ROS master processes')
    for proc_name in ['roscore', 'rosmaster', 'rosout']:
        utils.kill_process(proc_name)
    roslaunch_kill_proc = utils.new_process(['killall', 'roslaunch', '-2'])
    roslaunch_kill_proc.communicate()


def launch_rviz(config_path=None):
    command = ['rviz']
    if config_path is not None:
        command += ['-d', config_path]
    utils.new_process(command)


def kill_rviz():
    utils.kill_process('rviz')


def launch(**kwargs):
    if kwargs.has_key('direct_path'):
        command = ['roslaunch', kwargs.get('direct_path')]
    else:
        command = ['roslaunch', kwargs.get('package'), kwargs.get('launch_file')]
    if kwargs.has_key('argv'):
        command += ['%s:=%s' % (arg, value) for arg, value in kwargs.get('argv').items()]
    launch_proc = utils.new_process(command)
    return launch_proc


def run_node(package, node, namespace=None, argc=None, argv=None):
    if namespace is not None:
        command = ['export', 'ROS_NAMESPACE=%s' % namespace, '&&', 'rosrun', package, node]
    else:
        command = ['rosrun', package, node]
    if argc is not None:
        command += argc
    elif argv is not None:
        command += ['%s:=%s' % (arg, value) for arg, value in argv.items()]
    node_proc = utils.new_process(command)
    return node_proc


def play_bag(bag_path, use_clock=True):
    _logger.info('Starting bag playing')
    if type(bag_path) == tuple:
        bags = [rosbag.Bag(single_bag_path) for single_bag_path in bag_path]
        duration_in_seconds = sum([bag.get_end_time() - bag.get_start_time() for bag in bags])
        path_for_command_line = ' '.join(bag_path)
    else:
        bag = rosbag.Bag(bag_path)
        duration_in_seconds = bag.get_end_time() - bag.get_start_time()
        path_for_command_line = bag_path
    if use_clock:
        play_proc = utils.new_process(['rosbag', 'play', path_for_command_line, '--clock'], output_to_console=True)
    else:
        play_proc = utils.new_process(['rosbag', 'play', path_for_command_line], output_to_console=True)
    return play_proc, duration_in_seconds


def start_recording_bag(bag_path, topics=None):
    _logger.info('Starting bag recording')
    if topics is None:
        topics = ['-a']
    record_proc = utils.new_process(['rosbag', 'record'] + topics + ['-O', '%s.bag' % bag_path])
    return record_proc


def stop_recording_bags():
    _logger.info('Stopping bag recording')
    nodes_list = rosnode.get_node_names()
    for node_name in nodes_list:
        if node_name.find('record') != -1:
            rosnode.kill_nodes([node_name])


def save_map(map_name, dir_name):
    save_map_proc = utils.new_process(['rosrun', 'map_server', 'map_saver', '-f', map_name], cwd=dir_name)
    time.sleep(1)
    save_map_proc.kill()


def bag_to_dataframe(bag_path, topic, fields):
    data = {}
    timestamps = []
    for field in fields:
        data[field] = np.array([])
    if type(bag_path) is not tuple:
        bag_path = (bag_path,)
    for single_bag_path in bag_path:
        single_bag = rosbag.Bag(single_bag_path)
        for _, message, timestamp in single_bag.read_messages(topics=topic):
            for field in fields:
                data[field] = np.append(data[field], utils.rgetattr(message, field))
            timestamps.append(timestamp.to_sec())
    df = pd.concat([pd.Series(data[field], index=timestamps, name=field) for field in fields], axis=1)
    return df


def save_image_to_map(image, resolution, map_name, dir_name):
    cv2.imwrite(os.path.join(dir_name, map_name + '.pgm'), image)
    yaml_content = {'image' : map_name + '.pgm',
                    'resolution' : resolution,
                    'origin' : [0.0, 0.0, 0.0],
                    'negate' : 1,
                    'occupied_thresh' : 0.9,
                    'free_thresh' : 0.1}
    with open(os.path.join(dir_name, map_name + '.yaml'), 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file)
    return os.path.join(dir_name, map_name + '.yaml'), os.path.join(dir_name, map_name + '.pgm')


def video_to_bag(video_path, topic, frame_id, bag_path):
    bag = rosbag.Bag(bag_path, 'w')
    cap = cv2.VideoCapture(video_path)
    cv_bridge = CvBridge()
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3
        if not ret:
            break
        stamp = rospy.rostime.Time.from_sec(video_timestamp)
        message = cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header.seq = frame_idx
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        frame_idx += 1
        bag.write(topic, message, stamp)
    cap.release()
    bag.close()


def play_video_to_topic(video_path, topic, frame_id):
    class _VideoPlayer(object):
        def __init__(self, video_path, topic, ros_frame_id):
            rospy.init_node('video_player')
            pub = rospy.Publisher(topic, Image, queue_size=1)
            cap = cv2.VideoCapture(video_path)
            cv_bridge = CvBridge()
            frame_idx = 0
            prev_publish_time = None
            prev_video_timestamp = None
            while cap.isOpened():
                ret, frame = cap.read()
                video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3
                if not ret:
                    break
                stamp = rospy.rostime.Time.from_sec(video_timestamp)
                message = cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                message.header.seq = frame_idx
                message.header.stamp = stamp
                message.header.frame_id = ros_frame_id
                if frame_idx != 0:
                    delay = max(0, (video_timestamp - prev_video_timestamp) - (time.time() - prev_publish_time) - 5e-2)
                    time.sleep(delay)
                pub.publish(message)
                prev_publish_time = time.time()
                prev_video_timestamp = video_timestamp
                frame_idx += 1
            cap.release()

    p = Process(target=_VideoPlayer, args=(video_path, topic, frame_id))
    p.start()
    p.join()


def trajectory_to_bag(pose_time_tuples_list, bag_path, topic='ugv_pose'):
    bag = rosbag.Bag(bag_path, 'w')
    for pose_time in pose_time_tuples_list:
        # TODO: go over legacy code and see where IMAGE_HEIGHT is subtracted
        x = pose_time[0]
        y = pose_time[1]
        t = pose_time[2]
        ros_time = rospy.Time.from_sec(t)
        pose_2d_message = Pose2D(x, y, 0)
        bag.write(topic, pose_2d_message, ros_time)
    bag.close()


def wait_for_rosout_message(node_name, desired_message, is_regex=False):
    class _RosoutMessageWaiter(object):
        def __init__(self, node_name, desired_message, is_regex):
            self.wait = True
            self.node_name = node_name
            self.desired_message = desired_message
            self.is_regex = is_regex
            rospy.init_node('wait_for_rosout_message')
            rospy.Subscriber('/rosout', Log, self.rosout_callback)
            while self.wait:
                time.sleep(1)
        def rosout_callback(self, message):
            if DEBUG_ROSOUT:
                print('Received a new message on rosout: %s' % message.msg)
            if message.name == self.node_name or message.name[1:] == self.node_name:
                if self.is_regex:
                    if re.match(self.desired_message, message.msg) is not None:
                        self.wait = False
                else:
                    if message.msg == self.desired_message:
                        self.wait = False

    p = Process(target=_RosoutMessageWaiter, args=(node_name, desired_message, is_regex))
    p.start()
    p.join()


def downsample_bag(input_bag_path, topic, target_frequency, output_bag_path=None):
    input_bag = rosbag.Bag(input_bag_path)
    duration_in_seconds = input_bag.get_end_time() - input_bag.get_start_time()
    time_to_message = OrderedDict()
    for _, message, t in input_bag.read_messages(topics=topic):
        time_to_message[t.to_sec()] = message
    input_bag.close()
    samples_num = int(float(duration_in_seconds) * target_frequency)
    resampled_times = resample(np.array(time_to_message.keys()), num=samples_num)
    sampled_times = map(lambda sampled_time: np.array(time_to_message.keys()).flat[np.abs(np.array(time_to_message.keys()) - sampled_time).argmin()], resampled_times)
    if output_bag_path is None:
        output_bag_path = input_bag_path
    output_bag = rosbag.Bag(output_bag_path, 'w')
    for sampled_time in sampled_times:
        output_bag.write(topic, time_to_message[sampled_time], rospy.Time.from_sec(sampled_time))
    output_bag.close()