import rospy
import math

MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    """
    PID controller
    """
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM, idecay=.99):
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
        self.i_d = math.log(idecay)
        self.kd = kd
        self.min = mn
        self.max = mx

        self.d_buffer_size = 3
        self.d_buffer = [0.]*self.d_buffer_size
        self.d_buffer_t = [0.]*self.d_buffer_size
        self.d_ptr = 0
        self.d_sum = 0.
        self.d_sum_t = 0.

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
        integral = self.int_val*math.exp(sample_time*self.i_d) + error * sample_time
        # derivative part

        derivative = (error - self.last_error)
        self.d_sum += (derivative-self.d_buffer[self.d_ptr])
        self.d_buffer[self.d_ptr] = derivative
        self.d_sum_t += (sample_time-self.d_buffer_t[self.d_ptr])
        self.d_buffer_t[self.d_ptr] = sample_time
        self.d_ptr = (self.d_ptr+1) % self.d_buffer_size

        derivative = self.d_sum / self.d_sum_t

        # new PID controller value

        y = self.kp * error + self.ki * self.int_val + self.kd * derivative
        # rospy.loginfo("err {:.2f} d {:.2f} y {:.2f}".format(error, derivative, y))
        val = max(self.min, min(y, self.max))


        # limit value to safe ranges
        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min

        self.int_val = integral
        self.last_error = error

        return val
