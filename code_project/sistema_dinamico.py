from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint

class SistemaDinamico(ABC):
    def __init__(self, parametros):
        self.parametros = parametros
        self.estado = None
        self.controlador_activo = False
        self.controlador = None
        self.inicializar_estado()

    @abstractmethod
    def inicializar_estado(self):
        pass

    @abstractmethod
    def aplicar_accion(self, accion):
        pass

    @abstractmethod
    def obtener_estado(self):
        pass

    @abstractmethod
    def actualizar_estado(self, dt):
        pass

    def conectar_controlador(self, controlador):
        self.controlador = controlador
        self.controlador_activo = True

    def desconectar_controlador(self):
        self.controlador = None
        self.controlador_activo = False

class PenduloInvertido(SistemaDinamico):
    def inicializar_estado(self):
        self.estado = np.array([
            self.parametros.get('angulo_inicial', 0.0),
            self.parametros.get('velocidad_inicial', 0.0)
        ])
        self.torque = 0.0

    def aplicar_accion(self, accion):
        self.torque = accion

    def obtener_estado(self):
        return self.estado.copy()

    def ecuaciones_movimiento(self, estado, t, torque):
        angulo, velocidad = estado
        masa = self.parametros.get('masa', 1.0)
        largo = self.parametros.get('largo', 1.0)
        gravedad = self.parametros.get('gravedad', 9.81)

        dtheta_dt = velocidad
        dvelocidad_dt = (torque - masa * gravedad * largo * np.sin(angulo)) / (masa * largo**2)
        return [dtheta_dt, dvelocidad_dt]

    def actualizar_estado(self, dt):
        if self.controlador_activo:
            accion = self.controlador.calcular_accion(self.obtener_estado(), dt)
            self.aplicar_accion(accion)
        else:
            self.aplicar_accion(0.0)
        t = [0, dt]
        self.estado = odeint(self.ecuaciones_movimiento, self.estado, t, args=(self.torque,))[-1]

class LunarLander(SistemaDinamico):
    def inicializar_estado(self):
        self.estado = np.array([
            self.parametros.get('posicion_y_inicial', 1000.0),
            self.parametros.get('velocidad_y_inicial', 0.0),
            self.parametros.get('posicion_x_inicial', 0.0),
            self.parametros.get('velocidad_x_inicial', 0.0),
            self.parametros.get('orientacion_inicial', 0.0),
            self.parametros.get('velocidad_angular_inicial', 0.0)
        ])
        self.fuerza_motor_izquierdo = 0.0
        self.fuerza_motor_derecho = 0.0

    def aplicar_accion(self, accion):
        self.fuerza_motor_izquierdo, self.fuerza_motor_derecho = accion

    def obtener_estado(self):
        return self.estado.copy()

    def ecuaciones_movimiento(self, estado, t, fuerzas):
        y, vy, x, vx, theta, omega = estado
        masa = self.parametros.get('masa', 1000.0)
        gravedad = self.parametros.get('gravedad', 1.62)
        distancia_motores = self.parametros.get('distancia_motores', 1.0)
        fuerza_motor_izquierdo, fuerza_motor_derecho = fuerzas

        fuerza_total = fuerza_motor_izquierdo + fuerza_motor_derecho
        torque_total = (fuerza_motor_derecho - fuerza_motor_izquierdo) * (distancia_motores / 2)

        dy_dt = vy
        dvy_dt = (fuerza_total * np.cos(theta) / masa) - gravedad
        dx_dt = vx
        dvx_dt = (fuerza_total * np.sin(theta) / masa)
        dtheta_dt = omega
        domega_dt = torque_total / (masa * (distancia_motores / 2)**2)

        return [dy_dt, dvy_dt, dx_dt, dvx_dt, dtheta_dt, domega_dt]

    def actualizar_estado(self, dt):
        if self.controlador_activo:
            accion = self.controlador.calcular_accion(self.obtener_estado(), dt)
            self.aplicar_accion(accion)
        else:
            self.aplicar_accion((0.0, 0.0))
        t = [0, dt]
        fuerzas = (self.fuerza_motor_izquierdo, self.fuerza_motor_derecho)
        self.estado = odeint(self.ecuaciones_movimiento, self.estado, t, args=(fuerzas,))[-1]

class WaterRecoverySystem(SistemaDinamico):
    def inicializar_estado(self):
        self.niveles = {
            'acumulador1': self.parametros.get('nivel_inicial_acumulador1', 50.0),
            'acumulador2': self.parametros.get('nivel_inicial_acumulador2', 50.0),
            'acumulador3': self.parametros.get('nivel_inicial_acumulador3', 50.0),
            'acumulador4': self.parametros.get('nivel_inicial_acumulador4', 0.0),
            'acumulador_refrigerante': self.parametros.get('nivel_inicial_refrigerante', 100.0)
        }
        self.calidad = {
            'acumulador1': self.parametros.get('calidad_inicial_acumulador1', 0.5),
            'acumulador2': self.parametros.get('calidad_inicial_acumulador2', 0.6),
            'acumulador3': self.parametros.get('calidad_inicial_acumulador3', 0.7),
            'acumulador4': self.parametros.get('calidad_inicial_acumulador4', 0.0)
        }
        self.estado = {**self.niveles, **self.calidad}
        self.acciones = {
            'bomba1': 0.0,
            'bomba2': 0.0,
            'bomba3': 0.0,
            'valvula1': 0.0,
            'valvula2': 0.0,
            'calentador': 0.0,
            'energia_baterias': self.parametros.get('energia_inicial_baterias', 1000.0)
        }

    def aplicar_accion(self, accion):
        self.acciones.update(accion)

    def obtener_estado(self):
        estado_actual = {**self.estado, **self.acciones}
        return estado_actual

    def ecuaciones_movimiento(self, estado_vector, t, acciones):
        idx = 0
        niveles = {}
        calidad = {}
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4', 'acumulador_refrigerante']:
            niveles[acumulador] = estado_vector[idx]
            idx += 1
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4']:
            calidad[acumulador] = estado_vector[idx]
            idx += 1
        energia_baterias = estado_vector[idx]
        idx += 1

        # Parámetros y eficiencias
        eficiencia_device1 = self.parametros.get('eficiencia_device1', 0.9)
        eficiencia_device2 = self.parametros.get('eficiencia_device2', 0.95)
        eficiencia_device3 = self.parametros.get('eficiencia_device3', 0.98)
        consumo_energia_bombas = self.parametros.get('consumo_energia_bombas', 5.0)
        consumo_energia_calentador = self.parametros.get('consumo_energia_calentador', 10.0)
        generacion_energia_paneles = self.parametros.get('generacion_energia_paneles', 20.0)
        tasa_recirculacion = self.parametros.get('tasa_recirculacion', 0.1)

        # Variables de control
        bomba1 = acciones['bomba1']
        bomba2 = acciones['bomba2']
        bomba3 = acciones['bomba3']
        calentador = acciones['calentador']

        # Umbrales para verificar si el siguiente subsistema puede recibir flujo
        umbral_acumulador2 = self.parametros.get('umbral_acumulador2', 90.0)
        umbral_acumulador3 = self.parametros.get('umbral_acumulador3', 90.0)
        umbral_acumulador4 = self.parametros.get('umbral_acumulador4', 90.0)

        # Estado de los siguientes subsistemas
        acumulador2_listo = niveles['acumulador2'] < umbral_acumulador2
        acumulador3_listo = niveles['acumulador3'] < umbral_acumulador3
        acumulador4_listo = niveles['acumulador4'] < umbral_acumulador4

        # Ajuste de flujos en función del estado del siguiente subsistema
        flujo_bomba1 = bomba1 if acumulador2_listo else 0.0
        flujo_bomba2 = bomba2 if acumulador3_listo else 0.0
        flujo_bomba3 = bomba3 if acumulador4_listo else 0.0

        # Reducción del flujo al pasar por cada device
        flujo_device1 = flujo_bomba1 * eficiencia_device1
        flujo_device2 = flujo_bomba2 * eficiencia_device2
        flujo_device3 = flujo_bomba3 * eficiencia_device3

        # Dinámicas de niveles
        d_niveles_dt = {}
        d_calidad_dt = {}

        # Acumulador 1
        d_niveles_dt['acumulador1'] = -flujo_bomba1
        d_calidad_dt['acumulador1'] = 0  # Calidad constante en entrada

        # Device1 y Acumulador 2
        d_niveles_dt['acumulador2'] = flujo_device1 - flujo_bomba2
        d_calidad_dt['acumulador2'] = (flujo_device1 * calidad['acumulador1'] - flujo_bomba2 * calidad['acumulador2']) / (niveles['acumulador2'] + 1e-6)

        # Device2 y Acumulador 3 con recirculación
        recirculacion = tasa_recirculacion * niveles['acumulador3']
        d_niveles_dt['acumulador3'] = flujo_device2 + recirculacion - flujo_bomba3 - recirculacion
        d_calidad_dt['acumulador3'] = (flujo_device2 * calidad['acumulador2'] + recirculacion * calidad['acumulador3'] - flujo_bomba3 * calidad['acumulador3']) / (niveles['acumulador3'] + 1e-6)

        # Device3 y Acumulador 4
        d_niveles_dt['acumulador4'] = flujo_device3
        d_calidad_dt['acumulador4'] = (flujo_device3 * calidad['acumulador3']) / (niveles['acumulador4'] + 1e-6)

        # Acumulador de refrigerante
        d_niveles_dt['acumulador_refrigerante'] = -calentador * 0.1

        # Energía de baterías
        consumo_total = (
            flujo_bomba1 * consumo_energia_bombas +
            flujo_bomba2 * consumo_energia_bombas +
            flujo_bomba3 * consumo_energia_bombas +
            calentador * consumo_energia_calentador
        )
        d_energia_baterias_dt = generacion_energia_paneles - consumo_total

        # Construir el vector de derivadas
        derivadas = []
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4', 'acumulador_refrigerante']:
            derivadas.append(d_niveles_dt.get(acumulador, 0.0))
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4']:
            derivadas.append(d_calidad_dt.get(acumulador, 0.0))
        derivadas.append(d_energia_baterias_dt)

        return derivadas

    def actualizar_estado(self, dt):
        if self.controlador_activo:
            accion = self.controlador.calcular_accion(self.obtener_estado(), dt)
            self.aplicar_accion(accion)
        else:
            # Mantener acciones actuales si no hay controlador
            pass

        estado_vector = []
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4', 'acumulador_refrigerante']:
            estado_vector.append(self.niveles[acumulador])
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4']:
            estado_vector.append(self.calidad[acumulador])
        estado_vector.append(self.acciones['energia_baterias'])

        t = [0, dt]
        resultado = odeint(self.ecuaciones_movimiento, estado_vector, t, args=(self.acciones,))[-1]

        # Actualizar estados
        idx = 0
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4', 'acumulador_refrigerante']:
            self.niveles[acumulador] = max(resultado[idx], 0.0)
            idx += 1
        for acumulador in ['acumulador1', 'acumulador2', 'acumulador3', 'acumulador4']:
            self.calidad[acumulador] = np.clip(resultado[idx], 0.0, 1.0)
            idx += 1
        self.acciones['energia_baterias'] = max(resultado[idx], 0.0)
        self.estado = {**self.niveles, **self.calidad}