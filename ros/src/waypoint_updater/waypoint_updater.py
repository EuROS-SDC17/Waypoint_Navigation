#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Point
from styx_msgs.msg import Lane, Waypoint, CTE

import tf

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

LOOKAHEAD_WPS = 10 # Number of waypoints we will publish. You can change this number
DEFAULT_VELOCITY = 10 # default velocity for 1st phase waypoint updater

class WaypointUpdater(object):
    """
    Responsible for updating vehicle waypoints
    """
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        rospy.Subscriber('/traffic_waypoint', Waypoint, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.cte_pub = rospy.Publisher('cte', CTE, queue_size=1)

        self.search_range = int(rospy.get_param('~search_range'))
        self.nearest_waypoint_display_interval = int(rospy.get_param('~nearest_waypoint_info_interval', 1))
        self.nearest_waypoint_display_count = 0

        self.previous_closest_wp_index = None
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
        rospy.logdebug("received pose: {0}".format(msg))
        self.position = msg.pose.position

        orient = msg.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])[2]
        self.orientation = Point(math.cos(yaw), math.sin(yaw), 0.)

        final_waypoints, cte = self.prepare_waypoints()
        rospy.logdebug("prepared waypoints: {0}".format(final_waypoints))

        if not final_waypoints:
           return

        final_wp_msg = self.make_waypoints_message(msg.header.frame_id, final_waypoints)
        cte_msg = self.make_cte_message(msg.header.frame_id, cte)

        self.final_waypoints_pub.publish(final_wp_msg)
        self.cte_pub.publish(cte_msg)


    def waypoints_cb(self, msg):
        """
        Callback for receiving all base waypoints of a track
        :param msg: styx_msgs/Lane message
        """
        rospy.logdebug("received waypoints: {0}".format(len(msg.waypoints)))

        if self.base_waypoints is None:
            self.base_waypoints = msg.waypoints

    def traffic_cb(self, msg):
        """
        Callback for receiving traffic lights
        :param msg:
        """
        rospy.logdebug("received traffic light: {0}".format(msg))

        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_lights = msg

    def obstacle_cb(self, msg):
        """
        Callback for receiving obstacle positions
        :param msg:
        """
        rospy.logdebug("received obstacle: {0}".format(msg))

        # TODO: Callback for /obstacle_waypoint message. We will implement it later
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
        Sets the velocity that the vehicle should be driving at at the given waypoint
        :param waypoints: list of all waypoints
        :param waypoint: waypoint where we want to set linear speed
        :param velocity: velocity value
        :return:
        """
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        """
        Computes cumulative distance between two waypoints
        :param waypoints: list of all waypoints
        :param wp1: waypoint 1
        :param wp2: waypoint 2
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
        rospy.logdebug("find nearest waypoint for position: {0}".format(self.position))

        min_distance = 1E6
        min_index = -1

        if self.previous_closest_wp_index is None:
            candidate_index = None
            for index,waypoint in enumerate(self.base_waypoints):
                wp = waypoint.pose.pose.position
                distance = self.distance(self.position, wp)
                if distance < min_distance:
                    min_distance = distance
                    candidate_index = index
            # NOTE below is a somewhat approximate but faster version
            # will fail if there are different sections of the track
            # that are very close. If we don't do this very often
            # we could afford to do a full scan
            # dist_decreased = False
            # prev_dist = None
            # candidate_index = None
            # for index,waypoint in enumerate(self.base_waypoints):
            #     wp = waypoint.pose.pose.position
            #     distance = self.distance(self.position, wp)
            #     if (prev_dist is not None) and (distance>prev_dist) and (dist_decreased):
            #         break
            #     candidate_index = index
            #     if (prev_dist is not None) and (distance<prev_dist):
            #         dist_decreased = True
            #     prev_dist = distance
        else:
            candidate_index = self.previous_closest_wp_index

        min_distance = 1E6
        min_index = -1
        for index in range(candidate_index-self.search_range,candidate_index+self.search_range+1):
            wp = self.base_waypoints[index % len(self.base_waypoints)].pose.pose.position
            # get direction from vehicle to waypoint
            direction = self.make_vector(self.position, wp)
            rospy.logdebug("orientation = {0}, direction = {1}".format(self.orientation, direction))
            # only waypoints ahead are relevant
            if not self.is_matching_orientation(self.orientation, direction):
                continue;
            # is it the nearest waypoint so far?
            distance = self.distance(self.position, wp)
            if distance < min_distance:
                min_distance = distance
                min_index = index % len(self.base_waypoints)
        rospy.logdebug("found nearest waypoint ahead: {0}".format(
                      self.base_waypoints[min_index].pose.pose.position))
        self.previous_closest_wp_index = min_index
        return min_index

    def is_matching_orientation(self, a, b):
        """
        Scalar product test if vectors point have common angle inside [-90, 90]
        :param a: first orientation
        :param b: second orientation
        :return: true if orientations point to the same half-space
        """

        #print "dp",a.x * b.x + a.y * b.y + a.z * b.z
        return a.x * b.x + a.y * b.y + a.z * b.z > 0;

    def distance(self, a, b):
        """
        Euclidean distance between two 3D points
        :param a: first point
        :param b: second point
        :return: the distance
        """
        xdiff = a.x - b.x
        ydiff = a.y - b.y
        zdiff = a.z - b.z
        return math.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)

    def distance_from_line(self, p, a, b):
        da = self.distance(p, a)
        v1 = self.make_vector(p, a)
        v2 = self.make_vector(b, a)
        cos = (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z) / self.distance(a,b)
        sgn = 1 if v1.x*v2.y-v1.y*v2.x>0 else -1
        if da<cos:
            rospy.logerror("ERROR in distance from line p={0} a={1} b={2}".format(p,a,b))
            return 0.
        else:
            return math.sqrt(da*da-cos*cos)*sgn

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

        if self.nearest_waypoint_display_count == 0:
            rospy.loginfo("nearest waypoint {0} of {1}".format(\
                    i, 0 if self.base_waypoints is None else len(self.base_waypoints)))
        self.nearest_waypoint_display_count = (self.nearest_waypoint_display_count+1) % \
                self.nearest_waypoint_display_interval

        rospy.logdebug("nearest waypoint for position {0} index = {1} of {2}".format(\
                self.position, i, \
                0 if self.base_waypoints is None else len(self.base_waypoints)))
        if i == -1:
            return [], 1E6
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

        # TODO Peter: why is this i+scan_direction? shouldn't
        # the closest point ahead also be included?
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
        # TODO CTE should be based on final waypoints, not base_waypoints
        cte = self.distance_from_line(self.position, \
                self.base_waypoints[i].pose.pose.position, \
                self.base_waypoints[(i-scan_direction) % len(self.base_waypoints)].pose.pose.position)
        return result, cte

    def make_cte_message(self, frame_id, cte):
        msg = CTE()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.cte = cte
        return msg

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
