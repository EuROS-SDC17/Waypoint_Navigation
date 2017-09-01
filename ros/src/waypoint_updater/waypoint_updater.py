#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Point
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
DEFAULT_VELOCITY = 10 # default velocity for 1st phase waypoint updater

debug=True

class WaypointUpdater(object):
    """
    Responsible for updating vehicle waypoints
    """
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Waypoint, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.position = None
        self.orientation = None
        self.base_waypoints = None
        self.traffic_lights = None
        self.obstacles = None

        rospy.spin()

    def pose_cb(self, msg):
        """
        Callback for receiving position and orientation of the vehicle
        :param msg: geometry_msgs/Pose message
        """
        if debug:
            print "received pose: ", msg.pose.position, msg.pose.orientation
        self.position = msg.pose.position
        self.orientation = msg.pose.orientation
        final_waypoints = self.prepare_waypoints()
        if debug:
            print "prepared waypoints: ", final_waypoints
        if not final_waypoints:
           return
        msg = self.make_waypoints_message(msg.header.frame_id, final_waypoints)
        if debug:
            print "new waypoint update message: ", msg
        self.final_waypoints_pub.publish(msg)


    def waypoints_cb(self, msg):
        """
        Callback for receiving all base waypoints of a track
        :param msg: styx_msgs/Lane message
        """
        if debug:
            print "received waypoints: ", len(msg.waypoints)
        if self.base_waypoints is None:
            self.base_waypoints = msg.waypoints

    def traffic_cb(self, msg):
        """
        Callback for receiving traffic lights
        :param msg:
        """
        # TODO: Callback for /traffic_waypoint message. Implement
        if debug:
            print "received traffic light: ", msg
        self.traffic_lights = msg

    def obstacle_cb(self, msg):
        """
        Callback for receiving obstacle positions
        :param msg:
        """
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        if debug:
            print "received obstacle: ", msg
        self.obstacles = msg

    def get_waypoint_velocity(self, waypoint):
        """
        Gets linear velocity at a given waypoint
        :param waypoint: waypoint where we want to get linear speed
        :return: linear speed at a given waypoint
        """
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        """

        :param waypoints:
        :param waypoint:
        :param velocity:
        :return:
        """
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        """
        Computes cumulative distance between two waypoints
        :param waypoints:
        :param wp1:
        :param wp2:
        :return:
        """
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def make_vector(self, a, b):
        """
        Makes a vector pointing from a to b
        :param a: first point
        :param b: second point
        :return: vector pointing from a to b
        """
        direction = Point()
        direction = direction
        direction.x = b.x - a.x
        direction.y = b.y - a.y
        direction.z = b.z - a.z
        return direction

    def find_nearest_waypoint_index_ahead(self):
        """
        Finds an index of a nearest waypoint laying ahead of vehicle
        :return: index of a nearest waypoint laying ahead of vehicle
        """
        if self.position is None or self.base_waypoints is None:
            return -1
        min_distance = 1E6
        min_index = -1
        index = -1
        for waypoint in self.base_waypoints:
            wp = waypoint.pose.pose.position
            index += 1
            # get direction from vehicle to waypoint
            if debug:
                print "orientation=", self.orientation
            direction = self.make_vector(self.position, wp)
            if debug:
                print "direction=", direction
            # only waypoints ahead are relevant
            if not self.is_matching_orientation(self.orientation, direction):
                continue;
            # is it the nearest waypoint so far?
            distance = self.distance(self.position, wp)
            if distance < min_distance:
                min_distance = distance
                min_index = index
        return min_index

    def is_matching_orientation(self, a, b):
        """
        Scalar product test if vectors point have common angle inside [-90, 90]
        :param a: first orientation
        :param b: second orientation
        :return: true if orientations point to the same half-space
        """
        return a.x * b.x + a.y * b.y + a.z * b.z > 0;

    def distance(self, a, b):
        """

        :param a:
        :param b:
        :return:
        """
        return math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z))

    def prepare_waypoints(self):
        """
        Prepares a list of nearest LOOKAHEAD_WPS waypoints laying ahead of vehicle
        Assumptions:
          - waypoints always cover a non-intersecting loop
          - waypoints are connected as index increases, no jumps/holes in sequence
          - waypoints are ordered in a single direction
        :return: LOOKAHEAD_WPS number of waypoints laying ahead of the vehicle, starting with the nearest
        """
        i = self.find_nearest_waypoint_index_ahead()
        if i == -1:
            return []
        # now decide which way to go
        prev_i = i - 1
        next_i = i + 1
        if prev_i < 0:
            prev_i = len(self.base_waypoints) - 1
        if next_i >= len(self.base_waypoints):
            next_i = 0
        p_wb = self.base_waypoints[prev_i].pose.pose.position
        n_wb = self.base_waypoints[next_i].pose.pose.position
        prev_direction = self.make_vector(self.position, p_wb)
        next_direction = self.make_vector(self.position, n_wb)
        scan_direction = 1 # default direction is towards next waypoint in sequence
        if not self.is_matching_orientation(self.orientation, next_direction):
            # if orientation with next waypoint doesn't match, we need to scan backwards
            scan_direction = -1;

        j = i + scan_direction
        result = []
        for count in range(0, LOOKAHEAD_WPS):
            if scan_direction < 0 and j < 0:
                # wrap to last index
                j = len(self.base_waypoints) - 1
            elif scan_direction > 0 and j >= len(self.base_waypoints):
                # wrap to zero
                j = 0;
            self.set_waypoint_velocity(self.base_waypoints, j, DEFAULT_VELOCITY)
            waypoint = self.base_waypoints[j]
            result.append(waypoint)
        return result

    def make_waypoints_message(self, frame_id, waypoints):
        """
        Prepares a styx_msgs/Lane message containing waypoints
        :param frame_id: frame ID
        :param waypoints: List of waypoints
        :return: styx_msgs/Lane message containing waypoints
        """
        msg = Lane()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.waypoints = waypoints
        return msg

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
