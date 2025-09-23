# dynamic_system.py (Sin cambios mayores, solo se adapta a la nueva RewardFunction)
import numpy as np
from scipy.integrate import odeint
from interfaces import DynamicSystem, Controller, Environment
from factories import ComponentRegistry
from typing import List, Dict, Tuple, Any

@ComponentRegistry.register_component("system", "InvertedPendulum")
class InvertedPendulum(DynamicSystem, Environment):
    def __init__(self, m1: float = 1.0, m2: float = 1.0, l: float = 1.0, g: float = 9.81,
                 cr: float = 0.1, ca: float = 0.01, x0: List[float] = None,
                 angle_limit:float = np.pi/3, cart_limit:float = 5.0):
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.c_cart = cr
        self.c_pendulum = ca
        self.initial_state = x0 if x0 is not None else [0, 0, np.pi / 4, 0]
        self.state = self.initial_state.copy()
        self.angle_limit = angle_limit
        self.cart_limit = cart_limit
        self.controller = None

    def initialize_state(self) -> None:
        self.state = self.initial_state.copy()

    def dynamics(self, x: List[float], t: float, u: float) -> List[float]:
        x1, x2, x3, x4 = x
        cos_x3, sin_x3 = np.cos(x3), np.sin(x3)
        total_force = u - self.c_cart * x2
        dx1dt = x2
        dx2dt = (total_force + self.m2 * self.l * x4**2 * sin_x3 - self.m2 * self.g * sin_x3 * cos_x3) / (
                    self.m1 + self.m2 * (1 - cos_x3**2))
        dx3dt = x4
        dx4dt = (-total_force * cos_x3 + (self.m1 + self.m2) * self.g * sin_x3 - self.m2 * self.l * x4**2 * sin_x3 * cos_x3 - self.c_cart*x2) / (
                    self.l * (self.m1 + self.m2 * (1 - cos_x3**2)))
        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def update_state(self, dt: float, control_input: float) -> List[float]:
        t = [0, dt]
        new_state = odeint(self.dynamics, self.state, t, args=(control_input,))
        self.state = new_state[-1].tolist()
        self.state[2] = self.normalize_angle(self.state[2])  # Normaliza el ángulo
        return self.state

    def get_state(self) -> List[float]:
        return self.state

    def reset(self) -> List[float]:
        self.initialize_state()
        return self.get_state()

    def check_termination(self, state: List[float]) -> bool:
        if abs(state[2]) > self.angle_limit or abs(state[0]) > self.cart_limit:
            return True
        if abs(state[2]) < 0.01 and abs(state[3]) < 0.01: #Criterio estabilización
            return True
        return False

    def normalize_angle(self, angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def connect_controller(self, controller: Controller) -> None:
        self.controller = controller

    def step(self, action: float) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        next_state = self.update_state(dt=0.02, control_input=action) #El dt se debe obtener del config
        done = self.check_termination(next_state)
        reward = 0.0 # Se calculará fuera, en el orchestrator
        info = {"termination_reason":""}
        if done:
          if abs(next_state[2]) > self.angle_limit:
            info["termination_reason"] = "angle_limit"
          elif abs(next_state[0]) > self.cart_limit:
            info["termination_reason"] = "cart_limit"
          else:
            info["termination_reason"] = "stabilized"
        return next_state, reward, done, info