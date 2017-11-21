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

def kmph2mph(kmph):
    return kmph / 1.6093

def kmph2mps(kmph):
    return kmph / 3.6

def mps2kmph(mps):
    return mps * 3.6

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish.
# DEFAULT_VELOCITY = 10 # default velocity for 1st phase waypoint updater
MAXIMAL_VELOCITY = 50 # default velocity for 2nd phase waypoint updater
DEFAULT_VELOCITY = 40 # default velocity for 2nd phase waypoint updater
MIN_STOP_DISTANCE = 10.
STOP_DISTANCE= 60.  #

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
        self.red_traffic_light_index = None
        self.previous_red_traffic_light_index = None
        self.obstacle = None
        self.velocity_plan = {}  # velocities for next waypoints

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

        # RViz markers
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
        # publish pose in pink
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

        global MAXIMAL_VELOCITY
        global DEFAULT_VELOCITY

        if self.base_waypoints is None:
            self.base_waypoints = msg.waypoints
            MAXIMAL_VELOCITY = DEFAULT_VELOCITY = self.base_waypoints[0].twist.twist.linear.x
            # print "!!velocity===", MAXIMAL_VELOCITY, DEFAULT_VELOCITY

        # RViz markers
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.ns = "road"
        marker.id = 1000
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        # Line strip only uses x for width, not y or z
        marker.scale.x = 3.
        marker.color.a = 1.0
        marker.color.r = 0.82
        marker.color.g = 0.82
        marker.color.b = 0.82
        marker.pose.orientation.w = 1.0
        for wp in self.base_waypoints:
            p = Point()
            p.x = wp.pose.pose.position.x
            p.y = wp.pose.pose.position.y
            p.z = wp.pose.pose.position.z
            marker.points.append(p)

        # Add the first waypoint again to complete the loop
        p = Point()
        p.x = self.base_waypoints[0].pose.pose.position.x
        p.y = self.base_waypoints[0].pose.pose.position.y
        p.z = self.base_waypoints[0].pose.pose.position.z
        marker.points.append(p)


        # Publish the road waypoints (grey)
        self.vis_pub.publish(marker)


    def traffic_cb(self, msg):
        """
        Callback for receiving traffic lights
        :param msg: the index of the waypoint that is nearest to the upcoming red light
        """
        rospy.logdebug("received traffic light: {0}".format(msg))

        index = msg.data
        self.previous_red_traffic_light_index = self.red_traffic_light_index
        self.red_traffic_light_index = index if index >= 0 else None

        rospy.logdebug("nearest red traffic light at waypoint: {}".format(index))
        # RViz markers
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.ns = "traffic_light"
        marker.id = 10e6
        marker.type = marker.CUBE

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
        marker.scale.y = 5.
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

        if self.previous_closest_wp_index is None:
            candidate_index = None
            for index,waypoint in enumerate(self.base_waypoints):
                wp = waypoint.pose.pose.position
                distance = self.distance(self.position, wp)
                if distance < min_distance:
                    min_distance = distance
                    candidate_index = index
        else:
            candidate_index = self.previous_closest_wp_index

        min_distance = 1E6
        min_index = -1
        length = len(self.base_waypoints)
        for index in range(candidate_index-self.search_range, candidate_index+self.search_range+1):
            wp = self.base_waypoints[index % length].pose.pose.position
            # get direction from vehicle to waypoint
            direction = self.make_vector(self.position, wp)
            rospy.logdebug("orientation = {0}, direction = {1}".format(self.orientation, direction))
            # only waypoints ahead are relevant
            if not self.is_matching_orientation(self.orientation, direction):
                continue
            # is it the nearest waypoint so far?
            distance = self.distance(self.position, wp)
            if distance < min_distance:
                min_distance = distance
                min_index = index % length
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
        return a.x * b.x + a.y * b.y + a.z * b.z > 0

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
            rospy.logdebug("nearest waypoint {0} of {1}".format(\
                    i, 0 if self.base_waypoints is None else len(self.base_waypoints)))
        self.nearest_waypoint_display_count = (self.nearest_waypoint_display_count+1) % \
                self.nearest_waypoint_display_interval

        rospy.logdebug("nearest waypoint for position {0} index = {1} of {2}".format(\
                self.position, i, \
                0 if self.base_waypoints is None else len(self.base_waypoints)))
        if i == -1:
            return [], [], 1E6

        # now decide which way to go
        length = len(self.base_waypoints)
        next_i = (i + 1) % length
        n_wb = self.base_waypoints[next_i].pose.pose.position
        next_direction = self.make_vector(self.position, n_wb)

        scan_direction = 1 # default direction is towards next waypoint in sequence
        if not self.is_matching_orientation(self.orientation, next_direction):
            # if orientation with next waypoint doesn't match, we need to scan backwards
            scan_direction = -1;
        rospy.logdebug("scanning {}".format("forward" if scan_direction else "backward"))

        final_waypoints = []
        final_indices = []
        tf_wp_index = None
        for j in range(0, LOOKAHEAD_WPS):
            index = (i + j*scan_direction) % length
            global DEFAULT_VELOCITY
            self.set_waypoint_velocity(self.base_waypoints, index, kmph2mph(mps2kmph(DEFAULT_VELOCITY)))
            waypoint = self.base_waypoints[index]
            final_waypoints.append(waypoint)
            final_indices.append(index)
            if j>0:
                total_dist += self.distance(final_waypoints[j].pose.pose.position, final_waypoints[j-1].pose.pose.position)
            else:
                total_dist = self.distance(self.position, final_waypoints[0].pose.pose.position)

        self.update_waypoint_speed(i, scan_direction, final_waypoints, final_indices, MAXIMAL_VELOCITY)

        # TODO if obstacle avoidance / lane change is needed in future implementation
        # then CTE should be based on final waypoints, not base_waypoints
        cte = self.distance_from_line(self.position, \
                self.base_waypoints[i].pose.pose.position, \
                self.base_waypoints[(i-scan_direction) % length].pose.pose.position)
        return final_waypoints, final_indices, cte

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
        final_waypoints, final_indices, cte = self.prepare_waypoints()
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

    def get_braking_distance(self, velocity):
        """
        Estimate braking distance at a given velocity
        :param self:
        :param velocity: current vehicle velocity
        :return: braking distance at a given velocity
        """
        # scale linearly to 60mph
        ratio = min(1.0, velocity / 40.)
        return ratio * STOP_DISTANCE

    def update_waypoint_speed(self, nearest_ahead_index, scan_direction, final_waypoints, final_indices, maximal_velocity):
        """
        Updates velocities at waypoints
        :param self:
        :param nearest_ahead_index: index of nearest waypoint ahead
        :param scan_direction: direction of scanning the waypoint array to move forward
        :param final_waypoints: currently chosen waypoints to drive through
        :param final_indices: indices of chosen waypoints
        :param maximal_velocity: maximal vehicle velocity
        :return: waypoints with velocities set
        """
        if self.red_traffic_light_index == None:
            # if there is no red/yellow traffic light nearby, set velocity to maximum
            rospy.logdebug("No red light, setting maximal velocity")
            for waypoint in final_waypoints:
                waypoint.twist.twist.linear.x = kmph2mph(mps2kmph(MAXIMAL_VELOCITY))
        else:
            length = len(self.base_waypoints)
            # either there is a red, velocities differ, or there is no plan
            if self.red_traffic_light_index != self.previous_red_traffic_light_index or not self.velocity_plan:
                self.velocity_plan = {}
                rospy.logdebug("Preparing velocity plan, rwp={0}, prwp={1}".format(self.red_traffic_light_index, self.previous_red_traffic_light_index))
                # prepare a velocity plan taking into account velocity; operates on base_waypoints not on final ones
                stop_distance = self.get_braking_distance(maximal_velocity)
                rospy.logdebug("Stop distance={:.2f}".format(stop_distance))
                # indices in base_waypoints that need their speed set based on red traffic light
                cumulative_distance = 0
                i = self.red_traffic_light_index
                nodes = [i]
                rospy.logdebug("Red light position:{0}".format(self.red_traffic_light_index))
                sentinel = 5
                while cumulative_distance < stop_distance:
                    j = i
                    rospy.logdebug("i={0} sd={1} l={2}".format(i, scan_direction, length))
                    i -= (1 * scan_direction)
                    i = i % length
                    cumulative_distance += self.distance(self.base_waypoints[i].pose.pose.position,
                                                         self.base_waypoints[j].pose.pose.position)
                    nodes.insert(0, i)

                count = len(nodes)
                rospy.logdebug("Identified waypoints: {0}".format(nodes))

                for i, node in enumerate(nodes):
                    target_velocity = float(count - sentinel - i) / float(count) * kmph2mph(mps2kmph(MAXIMAL_VELOCITY))
                    target_velocity = max(0., target_velocity)
                    self.velocity_plan[node] = target_velocity
                for i in range(sentinel):
                    self.velocity_plan[nodes[count - i - 1]] = 0

                rospy.logdebug("Velocity plan: {0}".format(self.velocity_plan))
                rospy.logdebug("Final indices: {0}".format(final_indices))
            # use current velocity plan
            in_brake_zone = False
            for i, idx in enumerate(final_indices):
                if idx in self.velocity_plan:
                    final_waypoints[i].twist.twist.linear.x = self.velocity_plan[idx]
                    in_brake_zone = True
                else:
                    maximal_velocity = 0 if in_brake_zone else kmph2mph(mps2kmph(MAXIMAL_VELOCITY))
                    final_waypoints[i].twist.twist.linear.x = maximal_velocity

        return final_waypoints

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
