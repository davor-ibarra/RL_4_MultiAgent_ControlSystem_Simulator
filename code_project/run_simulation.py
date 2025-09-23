import sys
import json
import argparse
import time
from config import Configuracion
from sistema_dinamico import PenduloInvertido, LunarLander, WaterRecoverySystem
from controlador import ControladorPID, ControladorLQR, GestorControlador
from gestor_multiagente import GestorMultiagente
from simulator import Simulator

def main(config_path):
    # Medir el tiempo total de ejecución
    tiempo_inicio = time.time()

    # Cargar configuración desde el archivo JSON
    configuracion = Configuracion.cargar_desde_archivo(config_path)
    
    # Crear instancia del sistema dinámico
    tipo_sistema = configuracion.sistema_dinamico.tipo_sistema
    parametros_sistema = configuracion.sistema_dinamico.parametros
    
    if tipo_sistema == 'PenduloInvertido':
        sistema = PenduloInvertido(parametros_sistema)
    elif tipo_sistema == 'LunarLander':
        sistema = LunarLander(parametros_sistema)
    elif tipo_sistema == 'WaterRecoverySystem':
        sistema = WaterRecoverySystem(parametros_sistema)
    else:
        raise ValueError(f"Tipo de sistema dinámico '{tipo_sistema}' no reconocido")
    
    # Crear instancia del controlador
    tipo_controlador = configuracion.controlador.tipo_controlador
    parametros_controlador = configuracion.controlador.parametros
    
    if tipo_controlador == 'ControladorPID':
        controlador = ControladorPID(**parametros_controlador)
    elif tipo_controlador == 'ControladorLQR':
        K = parametros_controlador.get('K', [])
        controlador = ControladorLQR(K=K)
    else:
        raise ValueError(f"Tipo de controlador '{tipo_controlador}' no reconocido")
    
    # Crear gestor de controlador
    gestor_controlador = GestorControlador(controlador=controlador)
    
    # Conectar el controlador al sistema dinámico
    sistema.conectar_controlador(gestor_controlador)
    
    # Crear instancias de agentes múltiples
    config_agentes = configuracion.agentes  # Lista de configuraciones de agentes
    
    # Definir variables de estado basadas en el sistema dinámico
    if tipo_sistema == 'PenduloInvertido':
        variables_estado = ['angulo', 'velocidad']
    elif tipo_sistema == 'LunarLander':
        variables_estado = ['posicion_y', 'velocidad_y', 'posicion_x', 'velocidad_x', 'orientacion', 'velocidad_angular']
    elif tipo_sistema == 'WaterRecoverySystem':
        variables_estado = list(sistema.estado.keys())
    else:
        variables_estado = []
    
    gestor_agentes = GestorMultiagente(config_agentes, variables_estado)
    
    # Crear instancia del simulador
    simulador = Simulator(
        sistema=sistema,
        controlador=gestor_controlador,
        gestor_agentes=gestor_agentes,
        configuracion=configuracion
    )
    
    # Ejecutar simulación
    try:
        simulador.ejecutar_simulacion()
    except Exception as e:
        print(f"Ocurrió un error durante la simulación: {e}")
    finally:
        # Medir el tiempo total de ejecución
        tiempo_fin = time.time()
        tiempo_total = tiempo_fin - tiempo_inicio
        print(f"Simulación completada en {tiempo_total:.2f} segundos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ejecutar simulación')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración JSON')
    args = parser.parse_args()
    main(args.config)