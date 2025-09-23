import numpy as np
from scipy.integrate import odeint
from interfaces.dynamic_system import DynamicSystem

class InvertedPendulumSystem(DynamicSystem):
    def __init__(self, m1, m2, l, g, cr, ca):
        self.params = {'m1': m1, 'm2': m2, 'l': l, 'g': g, 'cr': cr, 'ca': ca}

    def dynamics(self, x, t, u):
        m1, m2, l, g, cr, ca = self.params.values()
        x = np.array(x).flatten()  # <-- Asegura que x es siempre un arreglo plano
        x1, x2, x3, x4 = x
        cosx3, sinx3 = np.cos(x3), np.sin(x3)
        force = u - cr * x2

        dx1dt = x2
        dx2dt = (force + m2 * l * x4**2 * sinx3 - m2 * g * sinx3 * cosx3) / (m1 + m2 * (1 - cosx3**2))
        dx3dt = x4
        dx4dt = (-force * cosx3 + (m1 + m2) * g * sinx3 - m2 * l * x4**2 * sinx3 * cosx3 - ca * x4)/(l * (m1 + m2 * (1 - cosx3**2)))

        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def apply_action(self, state, action, t, dt):
        state = np.array(state).flatten()  # <-- Corrige el error asegurando estado correcto
        next_state = odeint(self.dynamics, state, [t, t + dt], args=(action,))[1]
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        return next_state

    def reset(self, initial_conditions):
        return np.array(initial_conditions).flatten()
