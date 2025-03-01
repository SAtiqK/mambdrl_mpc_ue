class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current_value, norm):
        error = (setpoint - current_value)/norm

        # Proportional term
        p = self.kp * error

        # print('Error: ' + str(error))

        # Derivative term
        derivative = error - self.prev_error
        d = self.kd * derivative

        # print('Derivative: ' + str(derivative))

        if derivative == 0 or derivative == error:
            self.integral = self.integral
        else:
        # Integral term
            self.integral += error
        i = self.ki * self.integral

        # print('Intergral: ' + str(self.integral))




        # Compute the control output
        output = p + i + d

        # Store the current error for the next iteration
        self.prev_error = error

        return output