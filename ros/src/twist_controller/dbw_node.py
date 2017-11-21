#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Lane, CTE
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # steering
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        # throttle
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        # brake
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.controller = Controller(self.vehicle_mass, self.fuel_capacity, self.brake_deadband, self.decel_limit,
                                     self.accel_limit, self.wheel_radius, self.wheel_base, self.steer_ratio,
                                     self.max_lat_accel, self.max_steer_angle)

        rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb)

        self.cte = 0.
        rospy.Subscriber('/cte', CTE, self.cte_cb)
        self.velocity = 0.
        rospy.Subscriber('/current_velocity', TwistStamped, self.vehicle_velocity_cb)

        # steering
        rospy.Subscriber('/actual/steering_cmd', SteeringCmd, self.actual_steer_cb)
        # throttle
        rospy.Subscriber('/actual/throttle_cmd', ThrottleCmd, self.actual_throttle_cb)
        # brake
        rospy.Subscriber('/actual/brake_cmd', BrakeCmd, self.actual_brake_cb)

        # Drive-by-Wire enabled notification
        self.dbw_enabled = False
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        self.target_speed = 0.

        self.loop()

    def waypoints_cb(self, msg):
        if len(msg.waypoints):
            self.target_speed = msg.waypoints[0].twist.twist.linear.x

    def vehicle_velocity_cb(self, msg):
        self.velocity = msg.twist.linear.x

    def cte_cb(self, msg):
        self.cte = msg.cte

    def loop(self):
        """
        Main loop that periodically controls the vehicle
        """
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            if self.dbw_enabled:
              throttle, brake, steering = self.controller.control(\
                      rospy.get_time(), self.target_speed, self.velocity, self.cte) 
              rospy.logdebug("tgt v {:.2f} velocity {:.2f} cte {:.2f} thr {:.2f} brk {:.2f} str {:.2f}".format(\
                      self.target_speed, \
                      self.velocity, self.cte, throttle, brake, steering))
              self.publish(throttle, brake, -steering)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        """
        Publishes throttle, brake and steering values to car
        :param throttle: throttle value
        :param brake: brake value
        :param steer: steering value
        """
        # publish throttle value to /vehicle/throttle_cmd
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        # publish steering value to /vehicle/steering_cmd
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        # publish brake value to /vehicle/brake_cmd
        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def dbw_enabled_cb(self, msg):
        """
        Callback for /vehicle/dbw_enabled topic subscriber
        :param msg: message from dbw_enabled topic
        """
        self.dbw_enabled = msg.data
        if self.dbw_enabled:
            self.controller.reset(rospy.get_time(), self.cte)

    def actual_steer_cb(self, msg):
        """
        Callback for /actual/steering topic subscriber
        :param msg: message from dbw_enabled topic
        """
        if self.dbw_enabled and self.steer is not None:
            self.steer_data.append({'actual': msg.steering_wheel_angle_cmd,
                                    'proposed': self.steer})
            self.steer = None

    def actual_throttle_cb(self, msg):
        """
        Callback for /actual/ topic subscriber
        :param msg: message from dbw_enabled topic
        """
        if self.dbw_enabled and self.throttle is not None:
            self.throttle_data.append({'actual': msg.pedal_cmd,
                                       'proposed': self.throttle})
            self.throttle = None

    def actual_brake_cb(self, msg):
        """
        Callback for /actual/ topic subscriber
        :param msg: message from dbw_enabled topic
        """
        if self.dbw_enabled and self.brake is not None:
            self.brake_data.append({'actual': msg.pedal_cmd,
                                    'proposed': self.brake})
            self.brake = None


if __name__ == '__main__':
    DBWNode()
