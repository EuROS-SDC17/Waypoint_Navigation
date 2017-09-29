#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint, CTE
from visualization_msgs.msg import Marker, MarkerArray

import math
import tf


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

def mph2kmph(mph):
    return mph * 1.6093

def kmph2mps(kmph):
    return kmph / 3.6

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
# DEFAULT_VELOCITY = 10 # default velocity for 1st phase waypoint updater
DEFAULT_VELOCITY = 40 # default velocity for 2nd phase waypoint updater
MIN_STOP_DISTANCE = 10.
STOP_DISTANCE= 30.

class WaypointUpdater(object):
    """
    Responsible for updating vehicle waypoints
    """
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.cte_pub = rospy.Publisher('cte', CTE, queue_size=1)

        self.vis_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
        
        self.search_range = int(rospy.get_param('~search_range'))
        self.nearest_waypoint_display_interval = int(rospy.get_param('~nearest_waypoint_info_interval', 1))
        self.nearest_waypoint_display_count = 0

        self.previous_closest_wp_index = None
        self.position = None
        self.orientation = None
        self.base_waypoints = None
        # TODO where is self.traffic_light used?
        # self.traffic_light = None
        self.red_traffic_light_index = None
        self.obstacle = None

        rate = rospy.Rate(50) # 10hz
        while not rospy.is_shutdown():
            self.update_waypoints()
            rate.sleep()

    def pose_cb(self, msg):
        """
        Callback for receiving position and orientation of the vehicle
        :param msg: geometry_msgs/Pose message
        """
        rospy.logdebug("received pose: {0}".format(msg))
        self.position = msg.pose.position

        marker = Marker()
        marker.header.frame_id = "/world"
        marker.ns = "pose"
        marker.id = 0
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 5.
        marker.scale.y = 2.
        marker.scale.z = 1.
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.orientation.x = msg.pose.orientation.x
        marker.pose.orientation.y = msg.pose.orientation.y
        marker.pose.orientation.z = msg.pose.orientation.z
        marker.pose.orientation.w = msg.pose.orientation.w
        marker.pose.position.x = msg.pose.position.x
        marker.pose.position.y = msg.pose.position.y
        marker.pose.position.z = msg.pose.position.z
        # publish pose in purple
        self.vis_pub.publish(marker)
        
        orient = msg.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])[2]
        self.orientation = Point(math.cos(yaw), math.sin(yaw), 0.)


    def waypoints_cb(self, msg):
        """
        Callback for receiving all base waypoints of a track
        :param msg: styx_msgs/Lane message
        """
        rospy.loginfo("received waypoints: {0}".format(len(msg.waypoints)))

        if self.base_waypoints is None:
            self.base_waypoints = msg.waypoints


    def traffic_cb(self, msg):
        """
        Callback for receiving traffic lights
        :param msg: the index of the waypoint that is nearest to the upcoming red light
        """
        rospy.logdebug("received traffic light: {0}".format(msg))

        index = msg.data
        self.red_traffic_light_index = index if index >= 0 else None

        rospy.logdebug("nearest red traffic light at waypoint: {}".format(index))
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.ns = "traffic_light"
        marker.id = 10e6
        marker.type = marker.SPHERE

        if index != -1 and self.base_waypoints:
            marker.action = marker.ADD
            marker.pose.position.x = self.base_waypoints[index].pose.pose.position.x
            marker.pose.position.y = self.base_waypoints[index].pose.pose.position.y
            marker.pose.position.z = self.base_waypoints[index].pose.pose.position.z
        else:
            marker.action = marker.DELETE
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0

        marker.scale.x = 1.
        marker.scale.y = 1.
        marker.scale.z = 1.
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0

        # Publish the red traffic light marker (red)
        self.vis_pub.publish(marker)


    def obstacle_cb(self, msg):
        """
        Callback for receiving obstacle positions
        :param msg: the index of the waypoint that is nearest to the upcoming obstacle
        """
        rospy.logdebug("received obstacle: {0}".format(msg))

        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle = msg

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
        :return: the total distance of the piecewise linear arc from wp1 to wp2
        """
        dist = 0
        for i in range(wp1, wp2):
            dist += self.distance(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
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
        for index in range(candidate_index-self.search_range, candidate_index+self.search_range+1):
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

        wp = self.base_waypoints[min_index]
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.ns = "waypoints"
        marker.id = 100
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 1.
        marker.scale.y = 1.
        marker.scale.z = 1.
        marker.color.a = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = wp.pose.pose.position.x
        marker.pose.position.y = wp.pose.pose.position.y
        marker.pose.position.z = wp.pose.pose.position.z
        # Publish the nearest waypoint ahead marker (grey)
        self.vis_pub.publish(marker)

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
        next_i = (i + 1) % len(self.base_waypoints)
        n_wb = self.base_waypoints[next_i].pose.pose.position
        next_direction = self.make_vector(self.position, n_wb)
        scan_direction = 1 # default direction is towards next waypoint in sequence
        if not self.is_matching_orientation(self.orientation, next_direction):
            # if orientation with next waypoint doesn't match, we need to scan backwards
            scan_direction = -1;
        rospy.logdebug("scanning {}".format("forward" if scan_direction else "backward"))

        result = []
        tf_wp_index = None
        for j in range(0, LOOKAHEAD_WPS):
            index = (i + j*scan_direction) % len(self.base_waypoints)
            self.set_waypoint_velocity(self.base_waypoints, index, kmph2mps(mph2kmph(DEFAULT_VELOCITY)))
            waypoint = self.base_waypoints[index]
            result.append(waypoint)
            if j>0:
                total_dist += self.distance(result[j].pose.pose.position, result[j-1].pose.pose.position)
            else:
                total_dist = self.distance(self.position, result[0].pose.pose.position)
            if (index == self.red_traffic_light_index) and (total_dist > MIN_STOP_DISTANCE):
                tf_wp_index = j

        if tf_wp_index is not None:
            total_dist = 0.
            j = tf_wp_index
            while total_dist < STOP_DISTANCE and j>0:
                total_dist += self.distance(result[j-1].pose.pose.position, result[j].pose.pose.position)
                j -= 1
            while j < len(result):
                result[j].twist.twist.linear.x = 0.
                j += 1
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

    def update_waypoints(self):
        final_waypoints, cte = self.prepare_waypoints()
        rospy.logdebug("prepared waypoints: {0}".format(final_waypoints))

        if not final_waypoints:
           return

        count = 10
        for wp in final_waypoints:
            marker = Marker()
            marker.header.frame_id = "/world"
            marker.ns = "waypoints"
            marker.id = count
            count += 1
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 1.
            marker.scale.y = 1.
            marker.scale.z = 1.
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = wp.pose.pose.position.x
            marker.pose.position.y = wp.pose.pose.position.y
            marker.pose.position.z = wp.pose.pose.position.z
            # Publish the nearest waypoint marker (blue)
            self.vis_pub.publish(marker)

        final_wp_msg = self.make_waypoints_message("/world", final_waypoints)
        cte_msg = self.make_cte_message("/world", cte)

        self.final_waypoints_pub.publish(final_wp_msg)
        self.cte_pub.publish(cte_msg)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
