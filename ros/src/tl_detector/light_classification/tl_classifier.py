from styx_msgs.msg import TrafficLight
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        return TrafficLight.UNKNOWN

    def get_classification_by_HSV(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        # color thresholds in HSV space for the traffic lights
        # RED
        TRACK_RED_MIN = np.array([21, 84, 225])
        TRACK_RED_MAX = np.array([30, 96, 255])

        SIMULATOR_RED_MIN = np.array([150, 200, 225])
        SIMULATOR_RED_MAX = np.array([180, 216, 255])

        SIMULATOR_RED2_MIN = np.array([0, 105, 180])
        SIMULATOR_RED2_MAX = np.array([5, 115, 190])

        red_threshold = 15

        # YELLOW
        SIMULATOR_YELLOW_MIN = np.array([25, 215, 225])
        SIMULATOR_YELLOW_MAX = np.array([35, 225, 255])

        SIMULATOR_YELLOW2_MIN = np.array([27, 105, 180])
        SIMULATOR_YELLOW2_MAX = np.array([33, 115, 190])

        yellow_threshold = 15

        # GREEN
        TRACK_GREEN_MIN = np.array([85, 80, 225])
        TRACK_GREEN_MAX = np.array([95, 115, 255])

        SIMULATOR_GREEN_MIN = np.array([55, 215, 225])
        SIMULATOR_GREEN_MAX = np.array([65, 225, 255])

        SIMULATOR_GREEN2_MIN = np.array([55, 105, 180])
        SIMULATOR_GREEN2_MAX = np.array([65, 115, 190])

        green_threshold = 15

        # changing color to hsv space
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Computing thresholds
        red_threshed = np.sum(cv2.inRange(hsv_img, TRACK_RED_MIN, TRACK_RED_MAX) + \
                              cv2.inRange(hsv_img, SIMULATOR_RED_MIN, SIMULATOR_RED_MAX) + \
                              cv2.inRange(hsv_img, SIMULATOR_RED2_MIN, SIMULATOR_RED2_MAX))

        yellow_threshed = np.sum(cv2.inRange(hsv_img, SIMULATOR_YELLOW_MIN, SIMULATOR_YELLOW_MAX) + \
                                 cv2.inRange(hsv_img, SIMULATOR_YELLOW2_MIN, SIMULATOR_YELLOW2_MAX))

        green_threshed = np.sum(cv2.inRange(hsv_img, TRACK_GREEN_MIN, TRACK_GREEN_MAX) + \
                                cv2.inRange(hsv_img, SIMULATOR_GREEN_MIN, SIMULATOR_GREEN_MAX) + \
                                cv2.inRange(hsv_img, SIMULATOR_GREEN2_MIN, SIMULATOR_GREEN2_MAX))


        print ("Color detected:", red_threshed, yellow_threshed, green_threshed)

        if red_threshed >= red_threshold:
            # detecting red which has highest priority
            return TrafficLight.RED
        elif yellow_threshed >= yellow_threshold:
            # the detecting yellow
            return TrafficLight.YELLOW
        elif green_threshed >= green_threshold:
            # finally detecting green
            return TrafficLight.GREEN
        else:
            # if everything fails
            return TrafficLight.UNKNOWN

