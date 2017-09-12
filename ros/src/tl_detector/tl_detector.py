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
		
		Ywc = -Xwt * Sin(psi) + Ywt * Cos(psi);
		Xwc =  Xwt * Cos(psi) + Ywt * Sin(psi);
		Zwc =  Zwt

Psi = angle of camera rotation
(Xwc,Ywc,Zwc) = world coordinates of object transformed to camera orientation

    def pose_cb(self, msg):
        self.pose = msg

        # Record car's x,y,z position
        self.car_x = self.pose.pose.position.x
        self.car_y = self.pose.pose.position.y
        self.car_z = self.pose.pose.position.z

        # Deriving roll, pitch and yaw from car's position expressed in quaterions
        q_x = self.pose.pose.orientation.x
        q_y = self.pose.pose.orientation.y
        q_z = self.pose.pose.orientation.z
        q_w = self.pose.pose.orientation.w
        # Note that yaw is expressed in degrees
        self.roll, self.pitch, self.yaw = self.Quaternion_toEulerianAngle(q_x, q_y, q_z, q_w)

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoints_array = np.array([(waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                                         for waypoint in self.waypoints.waypoints])

        # k-d tree (https://en.wikipedia.org/wiki/K-d_tree)
        # as implemented in https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html
        self.waypoints_tree = KDTree(self.waypoints_array)

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
        index_closest_waypoint = self.waypoints_tree.query([(pose.position.x, pose.position.y)])[1][0]
        return index_closest_waypoint

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
        cx = int(image_width/2.0)
        cy = int(image_height/2.0)

        # Get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        # Using tranform and rotation to calculate 2D position of light in image
        # based on http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
		# https://stackoverflow.com/questions/4490570/math-formula-for-first-person-3d-game
		# https://stackoverflow.com/questions/28180413/why-is-cv2-projectpoints-not-behaving-as-i-expect

        rvec, _ = cv2.Rodrigues(np.identity(3)) # Rotation vector : no need because the camera is calibrated
        tvec = rvec.copy() # Translation vector : no need because the camera is centered

        cameraMatrix = np.array([[fx,  0, cx],
                                 [ 0, fy, cy],
                                 [ 0,  0,  1]], dtype=np.float32)

        # Deriving our target position in space
        target_x = point_in_world.x
        target_y = point_in_world.y
        target_z = point_in_world.z

        # Rotating target position in respect of our car yaw and car position
        rotated_x, rotated_y = self.clockwise_rotation(target_x, target_y, self.car_x, self.car_y, self.yaw)

        objectPoints = np.array([[rotated_x, rotated_y, target_z]], dtype=np.float32)

        imagePoints, jacobian = cv2.projectPoints([objectPoints], rvec, tvec, cameraMatrix, distCoeffs = None, aspectRatio=0)

        x = imagePoints[0][0][0]
        y = imagePoints[0][0][1]

        print(imagePoints)

        return (x, y)

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

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        # Finds the closest traffic light by comparing waypoints
        closest_distance = MAX_DISTANCE
        closest_light = None
        closest_light_wp = None

        for light in self.lights:
            light_wp = self.get_closest_waypoint(light.pose.pose)
            distance = light_wp - car_position

            # Ignore traffic lights that are behind us
            if (distance < closest_distance and distance > 0):
                closest_distance = distance
                closest_light = light
                closest_light_wp = light_wp

        if closest_light:
            state = self.get_light_state(light)
            return closest_light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
