from math import atan


class YawController(object):
    """
    Yaw angle controller
    """
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        """
        Initializes yaw controller
        :param wheel_base: vehicle wheel base
        :param steer_ratio: vehicle steer ratio
        :param min_speed: minimal vehicle speed
        :param max_lat_accel: maximal vehicle lateral acceleration
        :param max_steer_angle: maximal vehicle steering angle
        """
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle


    def get_angle(self, radius):
        """
        Computes steering angle from turn radius
        :param radius: turn radius
        :return: angle
        """
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        """
        Computes vehicle steering angle
        :param linear_velocity: linear vehicle velocity
        :param angular_velocity: angular vehicle velocity
        :param current_velocity: current vehicle velocity
        :return:
        """
        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.

        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity);
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))

        return self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0;
