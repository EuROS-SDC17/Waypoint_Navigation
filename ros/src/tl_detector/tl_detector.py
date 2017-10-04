#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from light_classification.tl_classifier_cnn import CNNTLStateDetector
import tf
import cv2
import yaml
import numpy as np
import math
from datetime import datetime
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
MAX_DISTANCE = 200  # Ignore traffic lights that are further
DEBUGGING = False
CLASSIFIER_DISABLED = False


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # Define if printing debugging information
        self.DEBUGGING = DEBUGGING
        self.CLASSIFIER_DISABLED = CLASSIFIER_DISABLED

        # Initializing key variables
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # KD-Trees placeholders
        self.waypoints_tree = None
        self.traffic_lights_tree = None
        self.traffic_lights_stops_tree = None

        # Traffic lights dictionary
        self.light_states = {0: "RED", 1:"YELLOW", 2:"GREEN", 4:"UNKNOWN"}

        # Hash memory
        self.hash_waypoints = 0
        self.hash_lights = 0

        # Initializing classifiers
        if not self.CLASSIFIER_DISABLED:
            self.light_classifier = TLClassifier()
            self.light_classifier_cnn = CNNTLStateDetector()

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def Quaternion_toEulerianAngle(self, x, y, z, w):
        """
        Conversion from quaternion to Eulerian angle
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
        """
        Initializes a KD tree for fast recovery of the nearest waypoint

        :param waypoints:
        :return: self.waypoints_tree is set with KDTree containing waypoints
        """
        hash_waypoints = hash(str(waypoints.waypoints))
        # We keep trace if the waypoints have changed since the last time
        if hash_waypoints != self.hash_waypoints:
            if self.DEBUGGING:
                print("Updating waypoints and its KDTree")
            self.hash_waypoints = hash_waypoints
            self.waypoints = waypoints
            # Initialization of waypoints k-d tree for fast search
            # k-d tree (https://en.wikipedia.org/wiki/K-d_tree)
            # as implemented in https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html
            waypoints_array = np.array([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                                        for waypoint in self.waypoints.waypoints])
            self.waypoints_tree = KDTree(waypoints_array)

    def traffic_cb(self, msg):
        """
        Initilizes data structures for traffic lights and stop lines

        :param msg:
        :return: self.stop_positions is an array containing stop lines
        self.traffic_lights_stops_tree is set with KDTree containing traffic lights
        self.no_lights is set with the number of expected traffic lights
        """
        self.lights = msg.lights
        lights_array = [(light.pose.pose.position.x, light.pose.pose.position.y) for light in self.lights]
        hash_lights = hash(str(lights_array))
        if hash_lights != self.hash_lights:
            if self.DEBUGGING:
                print("Updating traffic lights KDTree")
            self.hash_lights = hash_lights
            # Initialization of traffic lights k-d tree for fast search
            # k-d tree (https://en.wikipedia.org/wiki/K-d_tree)
            # as implemented in https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html
            lights_array = np.array(lights_array)
            self.traffic_lights_tree = KDTree(lights_array)
            # self.stop_positions records the x,y position of stops
            self.stop_positions = np.array([(x, y) for x,y in self.config['stop_line_positions']])
            self.traffic_lights_stops_tree = KDTree(self.stop_positions)
            self.no_lights = len(self.config['stop_line_positions'])

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
        if self.traffic_lights_tree:
            index_closest_light = self.traffic_lights_tree.query([pose], k=self.no_lights)[1][0]
            for index in index_closest_light:
                # Setting the index to int type
                int_index = int(index)
                # Getting the x,y coordinates of the closest light
                light_x, light_y = self.config['stop_line_positions'][int_index]
                # Getting the nearest waypoint of the closest light
                light_wp = self.get_closest_waypoint((light_x, light_y))
                # Computing the distance
                distance = light_wp - car_position
                # Returning only a traffic light that is ahead of us
                if distance > 0:
                    return int_index, light_wp, distance
        else:
            return None, None, None

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

        # Deriving our target position in space
        target_x = point_in_world[0]
        target_y = point_in_world[1]
        try:
            target_z = point_in_world[2]
        except:
            target_z = 1.2

        # Get transform between pose of camera and world frame
        trans = None
        try:
            latency = 0.0
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

            # Project to image coordinates
            u = int((-fx * (x / z) * image_width + cx))
            v = int((-fy * (y / z) * image_height + cy))

            if self.DEBUGGING:
                print(str(datetime.now()), "Traffic light position on screen: ", u, v)

            return (u, v)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
            return(int(cx), int(cy))

        # Using tranform and rotation to calculate 2D position of light in image
        # based on http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        # https://stackoverflow.com/questions/4490570/math-formula-for-first-person-3d-game
        # https://stackoverflow.com/questions/28180413/why-is-cv2-projectpoints-not-behaving-as-i-expect

    def get_light_state(self, light, ground_truth=None, hsv=False, cnn=True, debugging=False):
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

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        if hsv:
            x, y = self.project_to_image_plane(light)
            min_x = max(x - 100, 0)
            max_x = min(x + 100, image_width - 1)
            min_y = max(y - 100, 0)
            max_y = min(y + 300, image_height - 1)
            state_hsv = self.light_classifier.get_classification(cv_image[min_y:max_y, min_x:max_x])
            if debugging:
                cv2.rectangle(cv_image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                cv2.imshow('image', cv_image[min_y:max_y, min_x:max_x])
                cv2.waitKey(1)
        else:
            state_hsv = 4
        if cnn:
            state_cnn = self.light_classifier_cnn.get_classification(cv_image)
        else:
            state_cnn = 4

        if self.DEBUGGING:
            print(str(datetime.now()))
            if hsv:
                print("HSV Detected traffic light state is:", self.light_states[state_hsv])
            if cnn:
                print("CNN Detected traffic light state is:", self.light_states[state_cnn])
        return state_cnn

    def process_traffic_lights(self, debugging=True):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # If we are receiving car's position, we can also process traffic lights

        if self.pose:
            # Record car's x,y,z position
            self.car_x = self.pose.pose.position.x
            self.car_y = self.pose.pose.position.y
            self.car_z = self.pose.pose.position.z
            pose = (self.car_x, self.car_y)

            # Transforming the car position into the closest waypoint
            car_position = self.get_closest_waypoint(pose)

            if self.DEBUGGING:
                print (str(datetime.now()), "Car position:", car_position)

            # Finding the closest traffic light by comparing waypoints
            closest_light_index, closest_light_wp, closest_distance = self.get_closest_traffic_light(pose, car_position)
            if closest_distance < MAX_DISTANCE:
                if self.lights:
                    closest_light_position = self.lights[closest_light_index].pose.pose.position
                    closest_light = [closest_light_position.x, closest_light_position.y, closest_light_position.z]
                else:
                    closest_light = config['light_positions'][closest_light_index]
            else:
                closest_light = None

            if closest_light:
                ground_truth = self.lights[closest_light_index].state

                if self.DEBUGGING:
                    print("Ground truth state is:", self.light_states[ground_truth])
                    print(str(datetime.now()), "Detected traffic light no", closest_light_index, "at distance:", closest_distance)
                    print("the light is at", closest_light, "the stop sign is at ", self.stop_positions[closest_light_index])

                # Depending if we are debugging or not, we can get the traffic light location
                # from the topic /vehicle/traffic_lights which is the ground truth
                # the topic is incorporated in self.lights whose x,y,z positions are
                # in self.lights.pose.pose.position
                if self.CLASSIFIER_DISABLED:
                    state = ground_truth
                else:
                    state = self.get_light_state(closest_light, ground_truth, hsv=False, cnn=True)

                #Yellow as red
                if state == TrafficLight.YELLOW:
                    state = TrafficLight.RED

                return closest_light_wp, state
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
