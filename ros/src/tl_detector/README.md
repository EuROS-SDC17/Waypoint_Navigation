# tl_detector (Traffic Light Detection Node Overview)

Once the vehicle is able to process waypoints, generate steering and throttle commands, and traverse the course, it will also need stop for obstacles. Traffic lights are the first (and only) obstacle that we'll focus on.

The traffic light detection node (tl_detector.py) subscribes to three topics:

*    /base_waypoints provides the complete list of waypoints for the course.
*    /current_pose can be used used to determine the vehicle's location.
*    /camera/image_raw which provides an image stream from the car's camera. These images are used to determine the color of upcoming traffic lights.

The node should publish the index of the waypoint for nearest upcoming red light to a single topic:

*    /traffic_waypoint

For example, if waypoints is the complete list of waypoints, and an upcoming red light is nearest to waypoints[12], then 12 should be published /traffic_waypoint. This index can later be used by the waypoint updater node to set the target velocity for waypoints[12] to 0 and smoothly decrease the vehicle velocity in the waypoints leading up to waypoints[12].

The permanent (x, y) coordinates for each traffic light are provided by the config dictionary, which is imported from traffic_light_config.py:

	from traffic_light_config import config

Note that config is an [AttrDict](https://github.com/bcj/AttrDict), so you can access values using the dictionary keys as usual, or you can use attribute-style dot notation in your code:

    #Traditional dictionary access using 'light_positions' key:
    first_traffic_light = config['light_positions'][0]

    #Attribute-style access using dot notation with 'light_positions' key:
    first_traffic_light = config.light_positions[0]

Your task for this portion of the project can be broken into two steps:

* Use the vehicle's location and the (x, y) coordinates for traffic lights to find the nearest visible traffic light ahead of the vehicle. This takes place in the process_traffic_lights method of tl_detector.py. You will want to use the get_closest_waypoint method to find the closest waypoints to the vehicle and lights. Using these waypoint indices, you can determine which light is ahead of the vehicle along the list of waypoints.
* Locate the traffic light in the camera image data and classify it's state. The core functionality of this step takes place in the get_light_state method of tl_detector.py.

Note that the code to publish the results of process_traffic_lights is written for you already in the image_cb method.

##Traffic Light Detection package files
Within the traffic light detection package, you will find the following:

###tl_detector.py

    This python file processes the incoming traffic light data and camera images. It uses the light classifier to get a color prediction, and publishes the location of any upcoming red lights.

###tl_classifier.py

    This file contains the TLClassifier class. You can use this class to implement traffic light classification. For example, the get_classification method can take a camera image as input and return an ID corresponding to the color state of the traffic light in the image. Note that it is not required for you to use this class. It only exists to help you break down the classification problem into more manageable chunks.

###traffic_light_config

    This config file contains information about the camera (such as focal length) and the 3D position of the traffic lights in world coordinates. These values will change when your project is tested on the Udacity self-driving car.

##Helper Tool in the Simulator

In order to help you acquire an accurate ground truth data source for the traffic light classifier, the Udacity simulator publishes the location and current color state of all traffic lights in the simulator to the /vehicle/traffic_lights topic. This state can be used to generate classified images or subbed into your solution to help you work on another single component of the node. This topic won't be available when running your solution in real life so don't rely on it in the final submission.
