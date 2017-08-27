
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
        # TODO: Implement
        pass

    def control(self, *args, **kwargs):
        """
        Control vehicle
        :param args: TODO
        :param kwargs: TODO
        :return: throttle, brake, steering angle
        """
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.
