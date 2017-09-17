import math

class LowPassFilter(object):
    """
    Low-pass filter
    """
    def __init__(self, tau):
        """
        Initialize low-pass filter with cut-off frequency
        :param tau: scaled inverse cut-off frequency
        :param ts: sampling period
        """
        self.b = math.log(tau / (tau + 1.))

        self.last_val = 0.
        self.ready = False

    def reset(self):
        self.ready=False

    def get(self):
        """
        Returns the last filtered value
        :return: last filtered value
        """
        return self.last_val

    def filt(self, val, t_delta):
        """
        Perform low-pass filtering
        :param val: value to filter
        :return: filtered value
        """
        if self.ready:
            b = math.exp(self.b*t_delta)
            val = (1-b) * val + b * self.last_val
        else:
            self.ready = True

        self.last_val = val
        return val

