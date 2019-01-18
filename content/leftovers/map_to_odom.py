#!/usr/bin/env python

import rospy
import tf.transformations
from geometry_msgs.msg import Pose2D

class TfPub(object):
    def __init__(self):
        rospy.init_node('map_to_odom_tf_publisher')
        rospy.Subscriber('/ugv_pose', Pose2D, self.pose_callback)
        self.resolution = 0.0125
        self.init_pose = None

    def pose_callback(self, this_pose):
        if self.init_pose is None:
            self.init_pose = this_pose
        br = tf.TransformBroadcaster() # TODO: initialize in __init__? what's the frequency of this?
        br.sendTransform((self.init_pose.x * self.resolution, (1899-self.init_pose.y) * self.resolution, 0),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         'odom',
                         'map')

if __name__ == '__main__':
    TfPub()
    rospy.spin()