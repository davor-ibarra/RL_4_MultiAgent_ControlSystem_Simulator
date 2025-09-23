from interfaces import Controller

class PIDController(Controller):
    def __init__(self, kp, ki, kd, setpoint, dt, gain_step, variable_step=False, reset_gains_each_episode=True):
        self.initial_kp, self.initial_ki, self.initial_kd = kp, ki, kd
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.dt = dt
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.gain_step = gain_step
        self.variable_step = variable_step
        self.reset_gains_each_episode = reset_gains_each_episode

    def compute(self, error):
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        u = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        self.prev_error = error
        return u

    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in ['kp', 'ki', 'kd']:
                setattr(self, key, value)

    def reset_gains(self):
        if self.reset_gains_each_episode:
            self.kp, self.ki, self.kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.prev_error = 0.0
        self.integral_error = 0.0
