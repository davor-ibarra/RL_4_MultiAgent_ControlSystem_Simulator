import numpy as np
from scipy.integrate import odeint
from interfaces import DynamicSystem

class InvertedPendulum(DynamicSystem):
    def __init__(self, m1=1.0, m2=1.0, l=1.0, g=9.81, cr=0.1, ca=0.01, x0=None):
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.c_cart = cr
        self.c_pendulum = ca
        self.state = x0 if x0 is not None else [0, 0, np.pi/4, 0]

    def initialize_state(self):
        self.state = [0, 0, np.pi/4, 0]

    def dynamics(self, x, t, u):
        x1, x2, x3, x4 = x
        cos_x3, sin_x3 = np.cos(x3), np.sin(x3)
        total_force = u - self.c_cart * x2
        dx1dt = x2
        dx2dt = (total_force + self.m2 * self.l * x4**2 * sin_x3 - self.m2 * self.g * sin_x3 * cos_x3) / (self.m1 + self.m2 * (1 - cos_x3**2))
        dx3dt = x4
        dx4dt = (-total_force * cos_x3 + (self.m1 + self.m2) * self.g * sin_x3 - self.m2 * self.l * x4**2 * sin_x3 * cos_x3 - self.c_cart * x2) / (self.l * (self.m1 + self.m2 * (1 - cos_x3**2)))
        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def update_state(self, dt, control_input):
        t = [0, dt]
        new_state = odeint(self.dynamics, self.state, t, args=(control_input,))
        self.state = new_state[-1].tolist()
        self.state[2] = self.normalize_angle(self.state[2])
        return self.state

    def get_state(self):
        return self.state

    def reset(self):
        self.initialize_state()

    def check_termination(self, state):
        angle_limit = np.pi/3
        cart_limit = 5.0
        if abs(state[2]) > angle_limit or abs(state[0]) > cart_limit:
            return True
        if abs(state[2]) < 0.01 and abs(state[3]) < 0.01:
            return True
        return False

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
