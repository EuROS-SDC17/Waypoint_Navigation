from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2

class CNNTLStateDetector(object):
    def __init__(self):
        self.session = tf.Session()
        loader = tf.train.import_meta_graph('state_detector.meta')
        loader.restore(self.session, "./state_detector")
        graph = tf.get_default_graph()
        self.out = graph.get_tensor_by_name("output_class:0")
        self.input_layer = graph.get_tensor_by_name("input_layer:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")

    def get_classification(self, image):
        """
        Determines the color of the traffic light in a 
        cropped portion of the image

        Args:
            image (cv::Mat): image containing the cropped 
            image of a traffic light

        Returns:
            int: ID of traffic light color (as specified in styx_msgs/TrafficLight)

        """
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.session.run([self.out],
                                    feed_dict={self.input_layer: [RGB_image],
                                               self.keep_prob: 1.0})
        if int(result[0][0]) == 0:
            return TrafficLight.RED
        elif int(result[0][0]) == 1:
            return TrafficLight.YELLOW
        elif int(result[0][0]) == 2:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN

class CNNTLClassifier(object):
    def __init__(self):
        self.session = tf.Session()
        loader = tf.train.import_meta_graph('classifier.meta')
        loader.restore(self.session, "./classifier")
        graph = tf.get_default_graph()
        self.out = graph.get_tensor_by_name("output_class:0")
        self.input_layer = graph.get_tensor_by_name("input_layer:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.counter = 0

    def get_classification(self, image):
        """
        Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        resized_image = cv2.resize(image, (80, 60))
        fixed_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        result_3 = self.session.run([self.out],
                                    feed_dict={self.input_layer: [fixed_image],
                                               self.keep_prob: 1.0})
        if int(result_3[0][0]) == 1:
            return TrafficLight.RED
        else:
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
