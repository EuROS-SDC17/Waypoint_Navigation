# waypoint_loader

After launching `waypoint_loader.launch` (or 
`waypoint_loader_site.launch`), reads waypoints from a CSV 
file in the `data` directory and publishes the entire list 
of waypoints to `/base_waypoints` at regular intervals.

For the simulator the CSV file used is `wp_yaw_const.txt`.
The waypoints and yaw information can be visualized using 
`draw_waypoints.py`. The plot corresponding to the simulator
is in `{PROJECT_ROOT}/imgs/wp.png`.

The script `wp_loader_test.py` shows how to read one message 
from `/base_waypoints` and do some basic processing.

The message contains a `Header` and an array of `Waypoint`s.
Each `Waypoint` contains the following
* `pose.pose.position`: `x`, `y`, and `z` coordinates, unit
is meters, `z` is always `0`;
* `pose.pose.orientation`: `x`, `y`, `z` and `w`, orientation 
in quaternion, `x` and `y` are always `0`;
* `twist.twist.linear`: linear velocity, unit `m/s`, set by 
the `velocity` parameter in the launch file (the unit in the 
launch file is `kph`). 

    The velocity for the last few waypoints are such that the 
    car decelerates to `0` at the final waypoint. The 
    calculations doesn't do exactly that but it is a good enough
    approximation.
* `twist.twist.angular`: this is always `0`, not so interesting
at the moment.

