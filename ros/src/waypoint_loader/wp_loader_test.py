#!/usr/bin/env python

import sys
import math

from geometry_msgs.msg import Quaternion

from styx_msgs.msg import Lane, Waypoint

import tf
import rospy

def callback(data, result, sub):
    sub.unregister()
    rospy.loginfo('Received {0} waypoints'.format(len(data.waypoints)))
    assert(all((abs(wp.pose.pose.position.z)<1e-5) for wp in data.waypoints))
    rospy.loginfo('Verified that all z-coordinates for positions are zero')
    assert(all((abs(wp.pose.pose.orientation.x)<1e-5) for wp in data.waypoints))
    assert(all((abs(wp.pose.pose.orientation.y)<1e-5) for wp in data.waypoints))
    rospy.loginfo('Verified that orientation x and y axis are all zero')
    assert(all((abs(wp.twist.twist.linear.y)<1e-5) for wp in data.waypoints))
    assert(all((abs(wp.twist.twist.linear.z)<1e-5) for wp in data.waypoints))
    rospy.loginfo('Verified that only non-zero linear twist is x-axis')
    assert(all((abs(wp.twist.twist.angular.x)<1e-5) for wp in data.waypoints))
    assert(all((abs(wp.twist.twist.angular.y)<1e-5) for wp in data.waypoints))
    assert(all((abs(wp.twist.twist.angular.z)<1e-5) for wp in data.waypoints))
    rospy.loginfo('Verified that all angular twists are zero')
    rospy.loginfo(data.waypoints[0])
    result += [
            {'position': (wp.pose.pose.position.x, wp.pose.pose.position.y), \
                    'orientation_e': tf.transformations.euler_from_quaternion(\
                    [0,0,wp.pose.pose.orientation.z,wp.pose.pose.orientation.w])[2]/math.pi*180, \
                    'twist_linear': wp.twist.twist.linear.x}
            for wp in data.waypoints]
    rospy.signal_shutdown("Work finished.")

def listener(result):
    rospy.init_node('wp_loader_test')
    sub = rospy.Subscriber('base_waypoints', Lane, lambda data: callback(data, result, sub))
    rospy.spin()

if __name__ == '__main__':
    l = []
    listener(l)
    print('\n'.join(str((i,x)) for i,x in enumerate(l[2000:2010])))

