
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    """
    PID controller
    """
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        """
        Initializes PID controller
        :param kp: Proportional constant
        :param ki: Integral constant
        :param kd: Derivative constant
        :param mn: Minimal allowed value
        :param mx: Maximal allowed value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = self.last_int_val = self.last_error = 0.

    def reset(self):
        """
        Resets PID controller
        """
        self.int_val = 0.0
        self.last_int_val = 0.0

    def step(self, error, sample_time):
        """
        Performs a single PID computation step
        :param error: controller error
        :param sample_time: time delta since last computation
        :return: PID value
        """
        # remember last PID controller value
        self.last_int_val = self.int_val

        # integral part
        integral = self.int_val + error * sample_time;
        # derivative part
        derivative = (error - self.last_error) / sample_time;

        # new PID controller value
        y = self.kp * error + self.ki * self.int_val + self.kd * derivative;
        val = max(self.min, min(y, self.max))

        # limit value to safe ranges
        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min

        self.int_val = integral
        self.last_error = error

        return val
