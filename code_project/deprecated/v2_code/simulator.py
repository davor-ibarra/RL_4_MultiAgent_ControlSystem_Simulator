import time
import numpy as np
from monitor import Monitor
from gestor_multiagente import GestorMultiagente

class Simulator:
    def __init__(self, sistema, controlador, gestor_agentes, configuracion):
        self.sistema = sistema
        self.controlador = controlador
        self.gestor_agentes = gestor_agentes
        self.configuracion = configuracion
        self.tiempo_total = configuracion.general.tiempo_total
        self.paso_tiempo = configuracion.general.paso_tiempo
        self.num_episodios = configuracion.general.parametros.get('num_episodios', 1)
        self.monitor = Monitor(configuracion)
        self.resultados = []

    def ejecutar_simulacion(self):
        tiempo_inicial = time.time()
        for episodio in range(1, self.num_episodios + 1):
            print(f"Iniciando episodio {episodio}/{self.num_episodios}")
            recompensas_totales = self.ejecutar_episodio()
            # Ajustar hiperparámetros basados en los rendimientos de los agentes
            self.monitor.ajustar_hiperparametros_multiagente(self.gestor_agentes, recompensas_totales)
        tiempo_final = time.time()
        print(f"Simulación finalizada en {tiempo_final - tiempo_inicial:.2f} segundos.")
        self.monitor.finalizar()

    def ejecutar_episodio(self):
        num_pasos = int(self.tiempo_total / self.paso_tiempo)
        recompensas_totales = [0.0 for _ in self.gestor_agentes.agentes]

        # Reiniciar el estado del sistema
        self.sistema.inicializar_estado()

        for paso in range(num_pasos):
            # Obtener el estado actual del sistema
            estado = self.sistema.obtener_estado()
            estado_dict = self.convertir_estado_a_dict(estado)

            # Agentes perciben el entorno
            self.gestor_agentes.percibir_entorno(estado_dict)

            # Agentes deciden acciones
            acciones_agentes = self.gestor_agentes.decidir_acciones()

            # Integrar acciones de agentes en el controlador o sistema
            # *** Implementar lógica para combinar acciones de agentes
            # Por ejemplo, ajustar parámetros del controlador basados en acciones de agentes
            for accion_agente in acciones_agentes:
                if accion_agente is not None:
                    self.controlador.actualizar_parametros_controlador(**accion_agente)

            # Controlador calcula la acción de control
            accion_control = self.controlador.calcular_accion(estado, self.paso_tiempo)

            # Sistema aplica la acción de control y actualiza el estado
            self.sistema.aplicar_accion(accion_control)
            self.sistema.actualizar_estado(self.paso_tiempo)

            # Obtener el nuevo estado después de la acción
            nuevo_estado = self.sistema.obtener_estado()
            nuevo_estado_dict = self.convertir_estado_a_dict(nuevo_estado)

            # Calcular recompensas para cada agente
            recompensas = self.calcular_recompensas(estado_dict, accion_control, nuevo_estado_dict)

            # Acumular recompensas totales
            for i in range(len(recompensas_totales)):
                recompensas_totales[i] += recompensas[i]

            # Agentes reciben las recompensas y el nuevo estado
            self.gestor_agentes.recibir_recompensas(recompensas, [nuevo_estado_dict]*len(self.gestor_agentes.agentes))

            # Registrar datos en el monitor
            self.monitor.registrar_multiagente(
                estado=estado_dict,
                acciones={'accion_control': accion_control, 'acciones_agentes': acciones_agentes},
                recompensas=recompensas
            )

        return recompensas_totales

    def calcular_recompensas(self, estado, accion_control, nuevo_estado):
        # Calcula recompensas específicas para cada agente
        # *** Personalizar la lógica de recompensa para cada agente si es necesario
        recompensa_compartida = self.calcular_recompensa_global(estado, accion_control, nuevo_estado)
        recompensas = [recompensa_compartida for _ in self.gestor_agentes.agentes]
        return recompensas

    def calcular_recompensa_global(self, estado, accion_control, nuevo_estado):
        # Función original para calcular la recompensa global
        tipo_sistema = self.configuracion.sistema_dinamico.tipo_sistema

        if tipo_sistema == 'PenduloInvertido':
            angulo = nuevo_estado.get('angulo', 0.0)
            recompensa = - (abs(angulo) + 0.01 * abs(accion_control))
        elif tipo_sistema == 'LunarLander':
            velocidad_y = nuevo_estado.get('velocidad_y', 0.0)
            posicion_y = nuevo_estado.get('posicion_y', 0.0)
            recompensa = - (abs(velocidad_y) + abs(posicion_y) + 0.01 * np.sum(np.abs(accion_control)))
        elif tipo_sistema == 'WaterRecoverySystem':
            nivel_acumulador4 = nuevo_estado.get('acumulador4', 0.0)
            calidad_acumulador4 = nuevo_estado.get('calidad_acumulador4', 0.0)
            recompensa = nivel_acumulador4 * calidad_acumulador4 - 0.01 * np.sum(np.abs(accion_control))
        else:
            recompensa = 0.0
        return recompensa

    def convertir_estado_a_dict(self, estado):
        # Convierte el estado a un diccionario para facilitar su manejo
        variables_estado = self.obtener_variables_estado()
        estado_dict = {}
        if isinstance(estado, np.ndarray):
            for i, var in enumerate(variables_estado):
                estado_dict[var] = estado[i]
        elif isinstance(estado, dict):
            estado_dict = estado
        else:
            estado_dict['estado'] = estado
        return estado_dict

    def obtener_variables_estado(self):
        # Obtiene la lista de variables de estado según el sistema
        tipo_sistema = self.configuracion.sistema_dinamico.tipo_sistema
        if tipo_sistema == 'PenduloInvertido':
            return ['angulo', 'velocidad']
        elif tipo_sistema == 'LunarLander':
            return ['posicion_y', 'velocidad_y', 'posicion_x', 'velocidad_x', 'orientacion', 'velocidad_angular']
        elif tipo_sistema == 'WaterRecoverySystem':
            return list(self.sistema.estado.keys())
        else:
            return []