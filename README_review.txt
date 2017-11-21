Notes for the reviewer:

- please note that the traffic sign classifier uses advanced Deep Learning (SSD + CNN) and won't run well in a VM; 
  project was tested running on real hardware ROS + GPU install
  in order to run the project in VM with acceptable speed, please uncomment the line 19 in tl_detector.py:
  # CLASSIFIER_DISABLED = True
  This switches traffic sign classification from Tensorflow to the information provided by the simulator. For Carla you need to
  have this line commented out though.

- In case you don't turn classifier off while running under VM, the latency between recognizing green light and car moving
  can be around 3 seconds; car might miss multiple green lights as a consequence. Please either run it on a real hardware ROS
  installation or switch classifier off

- in case you run into CR/LF issues, fetch the project directly from GitHub:
  git clone https://github.com/EuROS-SDC17/Waypoint_Navigation.git

