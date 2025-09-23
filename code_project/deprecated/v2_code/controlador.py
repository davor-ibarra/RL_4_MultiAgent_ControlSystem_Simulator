from abc import ABC, abstractmethod
import numpy as np

class Controlador(ABC):
    @abstractmethod
    def calcular_accion(self, estado, dt):
        pass

    @abstractmethod
    def actualizar_parametros(self, **kwargs):
        pass

class ControladorPID(Controlador):
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0):
        self.kp = kp  # Ganancia proporcional
        self.ki = ki  # Ganancia integral
        self.kd = kd  # Ganancia derivativa
        self.setpoint = setpoint  # Valor deseado
        self.error_anterior = 0.0
        self.integral = 0.0

    def calcular_accion(self, estado, dt):
        # Suponiendo que estado[0] es la variable a controlar
        error = self.setpoint - estado[0]
        self.integral += error * dt
        derivada = (error - self.error_anterior) / dt if dt > 0 else 0.0
        accion = self.kp * error + self.ki * self.integral + self.kd * derivada
        self.error_anterior = error
        return accion

    def actualizar_parametros(self, **kwargs):
        if 'kp' in kwargs:
            self.kp = kwargs['kp']
        if 'ki' in kwargs:
            self.ki = kwargs['ki']
        if 'kd' in kwargs:
            self.kd = kwargs['kd']
        if 'setpoint' in kwargs:
            self.setpoint = kwargs['setpoint']
        if kwargs.get('reset_integral', False):
            self.integral = 0.0
        if kwargs.get('reset_error', False):
            self.error_anterior = 0.0

class ControladorLQR(Controlador):
    def __init__(self, K):
        self.K = np.array(K)  # Matriz de ganancias

    def calcular_accion(self, estado, dt=None):
        accion = -np.dot(self.K, estado)
        return accion

    def actualizar_parametros(self, **kwargs):
        if 'K' in kwargs:
            self.K = np.array(kwargs['K'])

class GestorControlador:
    def __init__(self, controlador: Controlador):
        self.controlador = controlador

    def cambiar_controlador(self, nuevo_controlador: Controlador):
        self.controlador = nuevo_controlador

    def calcular_accion(self, estado, dt):
        return self.controlador.calcular_accion(estado, dt)

    def actualizar_parametros_controlador(self, **kwargs):
        self.controlador.actualizar_parametros(**kwargs)