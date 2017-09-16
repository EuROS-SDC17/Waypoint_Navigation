from pid import PID

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
        self.steer_pid = PID(.4,5e-6,1, mn=-1.5, mx=1.5)

    def reset(self, time, cte):
        """
        reset internal states and timestamp
        """
        self.last_update = time
        self.steer_pid.reset()

    def control(self, timestamp, target_velocity, current_velocity, cte):
        """
        Control vehicle
        :param args: TODO
        :param kwargs: TODO
        :return: throttle, brake, steering angle
        """
        if current_velocity > target_velocity*1.1:
            throttle = 0
        elif current_velocity > target_velocity:
            throttle = 0.4
        elif current_velocity > target_velocity*0.8:
            throttle = 0.7
        else:
            throttle = 1.
        steer = self.steer_pid.step(cte, timestamp-self.last_update)
        self.last_update=timestamp
        return throttle, 0., steer

