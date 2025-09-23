import matplotlib.pyplot as plt
import json
import os
import threading
import time

class Monitor:
    def __init__(self, configuracion):
        self.configuracion = configuracion
        self.historial_estados = []
        self.historial_acciones = []
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.tiempos = []
        self.inicio_tiempo = time.time()
        self.lock = threading.Lock()
        self.directorio_resultados = self.configuracion.general.parametros.get('directorio_resultados', 'resultados')
        os.makedirs(self.directorio_resultados, exist_ok=True)
        self.frecuencia_guardado = self.configuracion.general.parametros.get('frecuencia_guardado', 10)  # En segundos
        self.frecuencia_visualizacion = self.configuracion.general.parametros.get('frecuencia_visualizacion', 10)  # En segundos
        self.ultima_epoca_guardado = 0
        self.ultima_epoca_visualizacion = 0

        # Iniciar hilos para guardado y visualización
        self.hilo_guardado = threading.Thread(target=self.guardar_periodicamente)
        self.hilo_visualizacion = threading.Thread(target=self.visualizar_periodicamente)
        self.hilo_guardado.daemon = True
        self.hilo_visualizacion.daemon = True
        self.hilo_guardado.start()
        self.hilo_visualizacion.start()

    def registrar(self, estado, accion, recompensa, perdida=None):
        with self.lock:
            self.historial_estados.append(estado)
            self.historial_acciones.append(accion)
            self.historial_recompensas.append(recompensa)
            if perdida is not None:
                self.historial_perdidas.append(perdida)
            tiempo_actual = time.time() - self.inicio_tiempo
            self.tiempos.append(tiempo_actual)

    def registrar_multiagente(self, estado, acciones, recompensas):
        with self.lock:
            self.historial_estados.append(estado)
            self.historial_acciones.append(acciones)
            self.historial_recompensas.append(recompensas)
            tiempo_actual = time.time() - self.inicio_tiempo
            self.tiempos.append(tiempo_actual)

    def guardar_datos(self, nombre_archivo='datos_monitoreo.json'):
        with self.lock:
            datos = {
                'tiempos': self.tiempos,
                'estados': self.historial_estados,
                'acciones': self.historial_acciones,
                'recompensas': self.historial_recompensas,
                'perdidas': self.historial_perdidas
            }
            ruta_archivo = os.path.join(self.directorio_resultados, nombre_archivo)
            with open(ruta_archivo, 'w') as archivo:
                json.dump(datos, archivo, indent=4)
            print(f"Datos guardados en '{ruta_archivo}'")

    def visualizar_datos(self):
        with self.lock:
            # Gráfico de recompensas acumuladas
            recompensas_totales = []
            if isinstance(self.historial_recompensas[0], list):
                # Multiagente: sumar recompensas de todos los agentes
                for recompensas in self.historial_recompensas:
                    recompensas_totales.append(sum(recompensas))
            else:
                recompensas_totales = self.historial_recompensas

            recompensas_acumuladas = np.cumsum(recompensas_totales)
            plt.figure(figsize=(10, 6))
            plt.plot(self.tiempos, recompensas_acumuladas, label='Recompensa Acumulada')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Recompensa Acumulada')
            plt.title('Recompensa Acumulada vs Tiempo')
            plt.legend()
            plt.grid(True)
            ruta_grafico_recompensas = os.path.join(self.directorio_resultados, 'recompensa_acumulada.png')
            plt.savefig(ruta_grafico_recompensas)
            plt.close()
            print(f"Gráfico de recompensa acumulada guardado en '{ruta_grafico_recompensas}'")

            # *** Agregar visualizaciones adicionales si es necesario

    def guardar_periodicamente(self):
        while True:
            tiempo_actual = time.time() - self.inicio_tiempo
            if tiempo_actual - self.ultima_epoca_guardado >= self.frecuencia_guardado:
                self.guardar_datos(nombre_archivo=f'datos_monitoreo_{int(tiempo_actual)}s.json')
                self.ultima_epoca_guardado = tiempo_actual
            time.sleep(1)

    def visualizar_periodicamente(self):
        while True:
            tiempo_actual = time.time() - self.inicio_tiempo
            if tiempo_actual - self.ultima_epoca_visualizacion >= self.frecuencia_visualizacion:
                self.visualizar_datos()
                self.ultima_epoca_visualizacion = tiempo_actual
            time.sleep(1)

    def ajustar_hiperparametros_multiagente(self, gestor_agentes, rendimientos):
        # Ajuste de hiperparámetros para múltiples agentes
        for agente, rendimiento_actual in zip(gestor_agentes.agentes, rendimientos):
            rendimiento_objetivo = agente.parametros.get('rendimiento_objetivo', 100)
            if rendimiento_actual >= rendimiento_objetivo and agente.parametros.get('epsilon', 0.1) > agente.parametros.get('epsilon_min', 0.01):
                nuevo_epsilon = agente.parametros['epsilon'] * agente.parametros.get('epsilon_decay', 0.995)
                agente.parametros['epsilon'] = max(nuevo_epsilon, agente.parametros['epsilon_min'])
                print(f"Se ajustó epsilon del agente a {agente.parametros['epsilon']:.5f}")

    def finalizar(self):
        # Guardar datos finales y detener hilos
        self.guardar_datos(nombre_archivo='datos_monitoreo_final.json')
        # Como los hilos son daemon, finalizarán cuando el programa termine