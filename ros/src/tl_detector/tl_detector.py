#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
from traffic_light_config import config
import yaml
import numpy as np
import math
from datetime import datetime
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
MAX_DISTANCE = 200  # Ignore traffic lights that are further

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_tree = None
        self.traffic_lights_tree = None
        self.light_states = {0: "RED", 1:"YELLOW", 2:"GREEN", 4:"UNKNOWN"}

        rospy.spin()

    def Quaternion_toEulerianAngle(self, x, y, z, w):
        """
        See: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        """

        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return (X, Y, Z)

    def clockwise_rotation(self, target_x, target_y, x_origin=0.0, y_origin=0.0, yaw_degrees=0.0):
        """
        see:  https://stackoverflow.com/questions/20104611/find-new-coordinates-of-a-point-after-rotation

        Ywc = -Xwt * Sin(psi) + Ywt * Cos(psi);
        Xwc =  Xwt * Cos(psi) + Ywt * Sin(psi);
        wc =  Zwt

        Psi = angle of camera rotation
        (Xwc,Ywc,Zwc) = world coordinates of object transformed to camera orientation
        """
        # Converting degrees into radiants
        rad_yaw = math.radians(yaw_degrees)

        # Centering to the origin
        centered_x = target_x - x_origin
        centered_y = target_y - y_origin

        # y' = y*cos(a) - x*sin(a)
        rotated_y = centered_y * math.cos(rad_yaw) - centered_x * math.sin(rad_yaw)

        # x' = y*sin(a) + x*cos(a)
        rotated_x = centered_y * math.sin(rad_yaw) + centered_x * math.cos(rad_yaw)

        return (rotated_x, rotated_y)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
            solved using a k-d tree of waypoints coordinates
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        index_closest_waypoint = self.waypoints_tree.query([pose])[1][0]
        return index_closest_waypoint

    def get_closest_traffic_light(self, pose, car_position):
        """Identifies the closest traffic light ahead to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
            solved using a k-d tree of waypoints coordinates
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: position of the the closest traffic light ahead in config['light_positions']
            int: index of the closest waypoint to the closest traffic light ahead in self.waypoints
            int: distance of the car to the closest traffic light ahead

        """
        index_closest_light = self.traffic_lights_tree.query([pose], k=self.no_lights)[1][0]
        for index in index_closest_light:
            # Setting the index to int type
            int_index = int(index)
            # Getting the x,y coordinates of the closest light
            light_x, light_y = config['light_positions'][int_index]
            # Getting the nearest waypoint of the closest light
            light_wp = self.get_closest_waypoint((light_x, light_y))
            # Computing the distance
            distance = light_wp - car_position
            # Returning only a traffic light that is ahead of us
            if distance > 0:
                return int_index, light_wp, distance

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        # The focal lengths expressed in pixel units
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']

        # The image shape
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # Principal x,y points that are at the image center
        cx = float(image_width/2.0)
        cy = float(image_height/2.0)

        # Deriving roll, pitch and yaw from car's position expressed in quaterions
        q_x = self.pose.pose.orientation.x
        q_y = self.pose.pose.orientation.y
        q_z = self.pose.pose.orientation.z
        q_w = self.pose.pose.orientation.w

        # Deriving our target position in space
        target_x = point_in_world[0]
        target_y = point_in_world[1]
        try:
            target_z = point_in_world[2]
        except:
            target_z = 0

        # Note that yaw is expressed in degrees
        self.roll, self.pitch, self.yaw = self.Quaternion_toEulerianAngle(q_x, q_y, q_z, q_w)
        # Get transform between pose of camera and world frame
        trans = None
        try:
            latency = 1.0
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", (now - rospy.Duration(latency)))

            tvec = tf.transformations.translation_matrix(trans)
            rvec = tf.transformations.quaternion_matrix(rot)
            homogeneous_coords = np.array([target_x, target_y, target_z, 1.0])

            # Combine all matrices
            camera_matrix = tf.transformations.concatenate_matrices(tvec, rvec)
            projection = camera_matrix.dot(homogeneous_coords)

            x, y, z = (projection[1], projection[2], projection[0])

            print("projection:", x, y, z)

            # Project to image coordinates
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)

            print(str(datetime.now()), "Traffic light position on screen: ", u, v)
            return (u, v)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
            return(int(cx), int(cy))

        # Using tranform and rotation to calculate 2D position of light in image
        # based on http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        # https://stackoverflow.com/questions/4490570/math-formula-for-first-person-3d-game
        # https://stackoverflow.com/questions/28180413/why-is-cv2-projectpoints-not-behaving-as-i-expect

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        image_height = cv_image.shape[0]
        image_width  = cv_image.shape[1]

        # Updating our config information
        self.config['camera_info']['image_width'] = image_width
        self.config['camera_info']['image_height'] = image_height

        x, y = self.project_to_image_plane(light)
        print(x,y)
        cv2.rectangle(cv_image, (x - 100, y - 100), (x + 100, y + 300), (0, 0, 255), 2)

        cv2.imshow('image', cv_image)
        cv2.waitKey(1)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self, debugging=True):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Initialization of waypoints and traffic lights k-d trees for fast search
        # k-d tree (https://en.wikipedia.org/wiki/K-d_tree)
        # as implemented in https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html

        ###########################################################
        # IMPORTANT ASSUMPTION:
        # WAYPOINTS AND TRAFFIC LIGHTS NEVER CHANGE DURING OUR RUN!
        # OTHERWISE WE SHOULD RE-INITIALIZE IF GIVEN NEW ONES
        ###########################################################

        if not self.waypoints_tree:
            waypoints_array = np.array([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                                              for waypoint in self.waypoints.waypoints])
            self.waypoints_tree = KDTree(waypoints_array)

        if not self.traffic_lights_tree:
            lights_array = np.array([(x, y) for x,y in config['light_positions']])
            self.traffic_lights_tree = KDTree(lights_array)
            self.no_lights = len(config['light_positions'])

        # If we are receiving car's position, we can also process traffic lights

        if(self.pose):

            # Record car's x,y,z position
            self.car_x = self.pose.pose.position.x
            self.car_y = self.pose.pose.position.y
            self.car_z = self.pose.pose.position.z
            pose = (self.car_x, self.car_y)

            # Transforming the car position into the closest waypoint
            car_position = self.get_closest_waypoint(pose)
            print (str(datetime.now()), "Car position:", car_position)

            # Finding the closest traffic light by comparing waypoints

            closest_light_index, closest_light_wp, closest_distance = self.get_closest_traffic_light(pose, car_position)
            if closest_distance < MAX_DISTANCE:
                closest_light = config['light_positions'][closest_light_index]
            else:
                closest_light = None

            if closest_light:
                print(str(datetime.now()), "Detected traffic light no", closest_light_index ,"at distance:", closest_distance)
                state = self.get_light_state(closest_light)

                # Depending if we are debugging or not, we can get the traffic light location
                # from the topic /vehicle/traffic_lights which is the ground truth
                # the topic is incorporated in self.lights whose x,y,z positions are
                # in light.pose.pose.position
                if self.lights and debugging:
                    true_state = self.lights[closest_light_index].state
                    print(str(datetime.now()), "Detected traffic light state is:", self.light_states[state],
                          "Ground truth state is:", self.light_states[true_state])
                else:
                    print(str(datetime.now()), "Detected traffic light state is:", self.light_states[state])

                return closest_light_wp, state
            else:
                return -1, TrafficLight.UNKNOWN


            #if closest_light:
            #    +            state = closest_light.state  # Temporary use the state from the simulator
            #+
            #return closest_light_wp, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
