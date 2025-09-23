from interfaces.controller import Controller

class PIDController(Controller):
    def __init__(self, kp, ki, kd, setpoint, dt):
        self.initial_kp, self.initial_ki, self.initial_kd = kp, ki, kd
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_kp = kp  # Add previous gains
        self.prev_ki = ki  # Add previous gains
        self.setpoint, self.dt = setpoint, dt
        self.prev_error = 0.0
        self.derivative_error = 0.0
        self.prev_integral_error = 0.0
        self.integral_error = 0.0

    def compute_action(self, state):
        error = state[2] - self.setpoint
        self.integral_error += ((self.prev_kp * self.prev_error) + (self.prev_ki * self.prev_integral_error * self.dt) - (self.kp * error))/self.ki if self.ki!=0 else 0
        derivative_error = (error - self.prev_error) / self.dt

        u = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)
        
        self.prev_kp = self.kp
        self.prev_ki = self.ki
        self.prev_error = error
        self.prev_integral_error = self.integral_error
        return u

    def update_params(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd

    def reset(self):
        self.kp, self.ki, self.kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.prev_kp = self.initial_kp
        self.prev_ki = self.initial_ki
        self.prev_error, self.integral_error = 0.0, 0.0
    
    def reset_episode(self):
        self.prev_error, self.integral_error = 0.0, 0.0