
class LowPassFilter(object):
    """
    Low-pass filter
    """
    def __init__(self, tau, ts):
        """
        Initialize low-pass filter with cut-off frequency
        :param tau: scaled inverse cut-off frequency
        :param ts: sampling period
        """
        self.a = 1. / (tau / ts + 1.)
        self.b = tau / ts / (tau / ts + 1.);

        self.last_val = 0.
        self.ready = False

    def get(self):
        """
        Returns the last filtered value
        :return: last filtered value
        """
        return self.last_val

    def filt(self, val):
        """
        Perform low-pass filtering
        :param val: value to filter
        :return: filtered value
        """
        if self.ready:
            val = self.a * val + self.b * self.last_val
        else:
            self.ready = True

        self.last_val = val
        return val
