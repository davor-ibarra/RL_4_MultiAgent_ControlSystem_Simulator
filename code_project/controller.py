from interfaces import Controller
from factories import ComponentRegistry
from typing import Dict

@ComponentRegistry.register_component("controller", "PIDController")
class PIDController(Controller):
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float, dt: float,
                 gain_step: float = 0.0,
                 reset_gains_each_episode: bool = True,
                 kp_min: float = 0.0, kp_max: float = 100.0,
                 ki_min: float = 0.0, ki_max: float = 10.0,
                 kd_min: float = 0.0, kd_max: float = 10.0):  # Añadidos límites
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.error_prev = 0
        self.integral = 0
        self.gain_step = gain_step
        self.reset_gains_each_episode = reset_gains_each_episode
        self.initial_gains = (kp, ki, kd)
        # Límites de las ganancias
        self.kp_min = kp_min
        self.kp_max = kp_max
        self.ki_min = ki_min
        self.ki_max = ki_max
        self.kd_min = kd_min
        self.kd_max = kd_max


    def compute(self, state: list) -> float:
        error = state[2] - self.setpoint
        self.integral += error * self.dt
        derivative = (error - self.error_prev) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.error_prev = error
        return output

    def reset(self):
        self.integral = 0
        self.error_prev = 0
        if self.reset_gains_each_episode:
            self.kp, self.ki, self.kd = self.initial_gains

    def update_parameters(self, **kwargs: Dict[str, float]) -> None:
        kp_action_index = kwargs.get('kp')
        ki_action_index = kwargs.get('ki')
        kd_action_index = kwargs.get('kd')

        if kp_action_index is not None:
          kp_action = self.gain_step*kp_action_index
          self.kp = np.clip(self.kp + kp_action, self.kp_min, self.kp_max)  # Limitar kp
        if ki_action_index is not None:
          ki_action = self.gain_step*ki_action_index
          self.ki = np.clip(self.ki + ki_action, self.ki_min, self.ki_max)  # Limitar ki
        if kd_action_index is not None:
          kd_action = self.gain_step*kd_action_index
          self.kd = np.clip(self.kd + kd_action, self.kd_min, self.kd_max)   # Limitar kd