import rospy
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    """
    Drive-by-Wire Twist controller
    """
    def __init__(self, *args, **kwargs):
        """
        Twist controller initialization
        :param args: TODO
        :param kwargs: TODO
        """
        self.last_update = 0
        self.max_velocity = 30

        self.steer_pid = PID(.4,0,.7, mn=-2.5, mx=2.5)
        self.steer_lowpass = LowPassFilter(4e-3)
        self.throttle_pid = PID(1.,0.,.5, mn=0, mx=1)
        self.throttle_lowpass = LowPassFilter(5e-2)
        self.brake_pid = PID(1.,0.,1.,mn=0,mx=1)
        self.brake_lowpass = LowPassFilter(5e-2)

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
        :param args: TODO
        :param kwargs: TODO
        :return: throttle, brake, steering angle
        """
        t_delta = timestamp-self.last_update
        if cte>0:
            cte = max(0., cte-0.1)
        elif cte<0:
            cte = min(0., cte+0.1)

        if abs(cte) > .75:
            self.speed_factor = max(.4, self.speed_factor*.99)
        elif abs(cte) < .4:
            self.speed_factor = min(1., self.speed_factor/.95)

        target_velocity *= self.speed_factor
        if abs(cte) > 3:
            target_velocity = min(3., target_velocity)

        if target_velocity > 0:
            throttle = self.throttle_pid.step(\
                    (target_velocity-current_velocity)/self.max_velocity, t_delta)
            throttle *= max(5.,current_velocity)*self.max_velocity
        else:
            throttle = 0

        if (target_velocity == 0 and current_velocity > 0) or \
                (target_velocity > 6. and current_velocity > target_velocity*1.05) or \
                (current_velocity * .99 > target_velocity):
            brake = self.brake_pid.step(\
                    (current_velocity-target_velocity)/self.max_velocity, t_delta)
            brake *= current_velocity*self.max_velocity
            brake = min(10, brake)
        else:
            self.brake_lowpass.reset()
            self.brake_lowpass.filt(0., 0.)
            brake = 0

        steer = self.steer_pid.step(cte, t_delta)

        # rospy.loginfo("thr {:.2f} brake {:.2f} steer {:.2f}".format(throttle, brake, steer))

        self.last_update = timestamp
        # return throttle, 0., steer
        return self.throttle_lowpass.filt(throttle, t_delta), \
                self.brake_lowpass.filt(brake, t_delta)*100, \
                self.steer_lowpass.filt(steer/(current_velocity*1.+0.01), t_delta)*10

