import rospy
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
DEFAULT_STEER_RATIO = 14.8
GEAR_RATIO = 2.4/1  ## 1st gear ratio 2.4:1
THROTTLE_ADJUSTMENT = 1000.  # magic constant for throttle
BRAKE_ADJUSTMENT = 60  # magic constat for brake

class Controller(object):
    """
    Drive-by-Wire Twist controller
    """
    def __init__(self, vehicle_mass=1736.35, fuel_capacity=13.5, brake_deadband=.1, decel_limit=-5, accel_limit=1.,
                 wheel_radius=0.2413, wheel_base=2.8498, steer_ratio=14.8, max_lat_accel=3., max_steer_angle=8.):
        """
        Twist controller initialization
        """
        # limits
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle

        # timestamp of the last update
        self.last_update = 0
        self.max_velocity = 50

        self.steer_pid = PID(.4,0,.7, mn=-2.5, mx=2.5)
        self.steer_lowpass = LowPassFilter(1e-4)
        self.throttle_pid = PID(1.,0.,.5, mn=0, mx=1)
        self.throttle_lowpass = LowPassFilter(5e-2)
        self.brake_pid = PID(1.,0.,1.,mn=0,mx=1)
        self.brake_lowpass = LowPassFilter(5e-2)

        # if CTE is too high, this factor will be reduced
        self.speed_factor = 1.

    def reset(self, time, cte):
        """
        reset internal states and timestamp
        """
        self.last_update = time
        self.steer_pid.reset()
        self.steer_lowpass.reset()
        self.brake_pid.reset()
        self.brake_lowpass.reset()
        self.throttle_pid.reset()
        self.throttle_lowpass.reset()

    def control(self, timestamp, target_velocity, current_velocity, cte):
        """
        Control vehicle
        :args: timestamp, target v, current v, cte
        :return: throttle, brake, steering angle
        """
        t_delta = timestamp-self.last_update
        if cte>0:
            cte = max(0., cte-0.1)
        elif cte<0:
            cte = min(0., cte+0.1)

        # adjust speed if CTE is increasing
        if abs(cte) > .75:
            self.speed_factor = max(.4, self.speed_factor*.99)
        elif abs(cte) < .4:
            self.speed_factor = min(1., self.speed_factor/.95)
        target_velocity *= self.speed_factor
        # slow down dramatically if vehicle too close to lane line
        if abs(cte) > 1.2:
            target_velocity = min(3., target_velocity)

        if target_velocity > 0:
            max_acceleration = (target_velocity-current_velocity) / t_delta
            max_torque = self.vehicle_mass * GEAR_RATIO * self.wheel_radius * max_acceleration
            acceleration_normalization = max_torque / (1080. * GEAR_RATIO * 0.335 * self.accel_limit * THROTTLE_ADJUSTMENT)
            throttle = self.throttle_pid.step(
                    (target_velocity-current_velocity)/self.max_velocity, t_delta)
            throttle *= max(5.,current_velocity)*self.max_velocity / acceleration_normalization
        else:
            throttle = 0

        if (target_velocity == 0 and current_velocity > 0) or (current_velocity * .99 > target_velocity):
            max_deceleration = (target_velocity-current_velocity) / t_delta
            max_torque = self.vehicle_mass * self.wheel_radius * max_deceleration
            deceleration_normalization = max_torque / (1080. * 0.335 * self.decel_limit * BRAKE_ADJUSTMENT)
            brake = self.brake_pid.step(
                (current_velocity-target_velocity)/self.max_velocity, t_delta)
            brake *= current_velocity*self.max_velocity
            brake = min(10, brake / deceleration_normalization)
        else:
            # if no brake is needed then reset, otherwise low pass
            # filter increases latency
            self.brake_lowpass.reset()
            self.brake_lowpass.filt(0., 0.)
            brake = 0

        steer = self.steer_pid.step(cte, t_delta) * self.steer_ratio / DEFAULT_STEER_RATIO  # PID normalized to simulator steer ratio

        # rospy.loginfo("thr {:.2f} brake {:.2f} steer {:.2f}".format(throttle, brake, steer))

        self.last_update = timestamp

        # reduce steering magnitude as velocity increases
        throttle = self.throttle_lowpass.filt(throttle, t_delta)
        brake = self.brake_lowpass.filt(brake, t_delta)*100
        steer = self.steer_lowpass.filt(steer/(current_velocity*1.+0.01), t_delta)*10

        return throttle, brake, steer


