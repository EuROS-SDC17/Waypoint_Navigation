# Styx

Styx is a package that contains a server for communicating with the simulator, and a bridge to translate and publish simulator messages to ROS topics.

It is part of the packages that are not necessary to be changed for achieving the project. Consequently is just to be understood in its functioning.

The directory contains the following Python scripts:

### conf.py

the script contains an [AttrDict](https://pypi.python.org/pypi/attrdict/2.0.0) (pratically a dictionary whose values are also keys and viceversa) reporting a list of subscrivers (issuing orders to the car) and publishers (reporting the status of the car).

The subscrivers are:
* /vehicle/steering_cmd (name: 'steering_angle')
* /vehicle/throttle_cmd (name: 'throttle')
* /vehicle/brake_cmd (name: 'brake')

the publishers instead are:
* /current_pose (name: 'current_pose')
* /current_velocity (name: 'current_velocity')
* /vehicle/steering_report (name: 'steering_report')
* /vehicle/throttle_report (name: 'throttle_report')
* /vehicle/brake_report (name: 'brake_report')
* /vehicle/obstacle (name: 'obstacle')
* /vehicle/obstacle_points (name: 'obstacle_points' )
* /vehicle/lidar (name: 'lidar')
* /vehicle/traffic_lights (name: 'trafficlights')
* /vehicle/dbw_enabled (name: 'dbw_status')
* /camera/image_raw (name: 'image')

### bridge.py

it contains the class Bridge, which initializes the connection with ROS using rospy, a Python client library for ROS. 
When initialized it records on ROS all the subscribers and publishers as present in conf.py.

if features the following functions for publishing to ROS:

publish_odometry : publishing 'current_pose', 'current_velocity'
publish_controls : publishing 'steering_report', 'throttle_report' and 'brake_report'
publish_obstacles : publishing 'obstacle' (xyz positions) and 'obstacle_points' 
publish_lidar : publishing 'lidar'
publish_traffic : publishing 'trafficlights'
publish_dbw_status : publishing 'dbw_status'
publish_camera : publishing 'image'

and the following functions for callbacks (commands) to ROS:

callback_steering (sending steering_wheel_angle_cmd)
callback_throttle (sending pedal_cmd)
callback_brake (sending again pedal_cmd, thus throttle and brake are exacted by the same command)

### server.py

it acts as a server for communicating with the simulator. When started, it:

* starts a Python Socket.IO server (variable sio)
* uses Flask to serve the client application, the simulator (using the variable app, later to be wrapped with socketio's middleware)
* Initializes a Bridge class in order to translate and publish simulator messages to ROS topics (variable bridge)
* Initializes a message list (list msgs) whose purpose is to append messages as tuples of the kind (topic, data) and be used by the bridge
* Defines a series of event handlers (connect, telemetry, control, obstacle, lidar, trafficlights, image), which, upon receiving specific events, all leverage the bridge to publish to ROS topics
* Finally deploys an eventlet WSGI server and start listening to port 4567 in order to feed the variable app