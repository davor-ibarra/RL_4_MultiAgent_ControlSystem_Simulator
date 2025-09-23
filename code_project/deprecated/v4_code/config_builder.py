import json
import os
from typing import Dict, Any, List
import numpy as np

class ConfigBuilder:
    def __init__(self):
        self.config = {
            "simulation": {},
            "system": {"type": "", "parameters": {}},
            "controller": {"type": "", "parameters": {}},
            "rl": {"agent_type": "", "parameters": {}, "sub_agents":{}},
            "reward": {"type": "", "parameters": {}},
            "state": {}
        }

    def build_config(self):
        self.get_simulation_params()
        self.get_system_config()
        self.get_controller_config()
        self.get_rl_config()
        self.get_reward_config()
        self.get_state_config() #Para la discretización
        return self.config

    def get_simulation_params(self):
        print("--- Configuración de la Simulación ---")
        self.config["simulation"]["max_episodes"] = int(input("Número máximo de episodios: "))
        self.config["simulation"]["time_step"] = float(input("Paso de tiempo (dt): "))
        self.config["simulation"]["decision_interval"] = int(input("Intervalo de decisión del agente: "))
        self.config["simulation"]["episodes_per_file"] = int(input("Episodios por archivo: "))
        self.config["simulation"]["results_folder"] = input("Carpeta de resultados (dejar en blanco para 'results_history'): ") or "results_history"

    def get_system_config(self):
        print("\n--- Configuración del Sistema Dinámico ---")
        system_type = input("Tipo de sistema (ej. InvertedPendulum): ")
        self.config["system"]["type"] = system_type
        if system_type == "InvertedPendulum":
            print("Parámetros del Péndulo Invertido (dejar en blanco para valores por defecto):")
            self.config["system"]["parameters"]["m1"] = float(input("  m1 (masa del carro): ") or 1.0)
            self.config["system"]["parameters"]["m2"] = float(input("  m2 (masa del péndulo): ") or 1.0)
            self.config["system"]["parameters"]["l"] = float(input("  l (longitud del péndulo): ") or 1.0)
            self.config["system"]["parameters"]["g"] = float(input("  g (gravedad): ") or 9.81)
            self.config["system"]["parameters"]["cr"] = float(input("  cr (coef. fricción del carro): ") or 0.1)
            self.config["system"]["parameters"]["ca"] = float(input("  ca (coef. fricción del péndulo): ") or 0.01)
            x0_str = input("  x0 (estado inicial, ej. 0,0,0.1,0): ")
            self.config["system"]["parameters"]["x0"] = [float(x) for x in x0_str.split(',')] if x0_str else [0, 0, np.pi/4, 0]
            self.config["system"]["parameters"]["angle_limit"] = float(input("Límite del ángulo (radianes):") or np.pi/3)
            self.config["system"]["parameters"]["cart_limit"] = float(input("Límite del carro (metros):") or 5.0)

    def get_controller_config(self):
        print("\n--- Configuración del Controlador ---")
        controller_type = input("Tipo de controlador (ej. PIDController): ")
        self.config["controller"]["type"] = controller_type
        if controller_type == "PIDController":
            print("Parámetros del Controlador PID:")
            self.config["controller"]["parameters"]["kp"] = float(input("  kp (ganancia proporcional): ") or 1.0)
            self.config["controller"]["parameters"]["ki"] = float(input("  ki (ganancia integral): ") or 0.1)
            self.config["controller"]["parameters"]["kd"] = float(input("  kd (ganancia derivativa): ") or 0.01)
            self.config["controller"]["parameters"]["setpoint"] = float(input("  setpoint: ") or 0.0)
            self.config["controller"]["parameters"]["dt"] = float(input("  dt (paso de tiempo):") or 0.02)
            self.config["controller"]["parameters"]["gain_step"] = float(input("  gain_step: ") or 0.1)
            self.config["controller"]["parameters"]["reset_gains_each_episode"] = input("Resetear ganancias cada episodio? (y/n):") == 'y'
            print("  Límites de las ganancias:")
            self.config["controller"]["parameters"]["kp_min"] = float(input("    kp_min: ") or 0.0)
            self.config["controller"]["parameters"]["kp_max"] = float(input("    kp_max: ") or 100.0)
            self.config["controller"]["parameters"]["ki_min"] = float(input("    ki_min: ") or 0.0)
            self.config["controller"]["parameters"]["ki_max"] = float(input("    ki_max: ") or 10.0)
            self.config["controller"]["parameters"]["kd_min"] = float(input("    kd_min: ") or 0.0)
            self.config["controller"]["parameters"]["kd_max"] = float(input("    kd_max: ") or 10.0)

    def get_rl_config(self):
        print("\n--- Configuración del Agente RL ---")
        agent_type = input("Tipo de agente (ej. QLearning, PIDQLearning): ")
        self.config["rl"]["agent_type"] = agent_type

        if agent_type == "QLearning":
            self.get_qlearning_config(self.config["rl"]["parameters"])

        elif agent_type == "PIDQLearning":
            self.config["rl"]["parameters"]["enabled"] = input("Habilitar PIDQLearning? (y/n): ") == 'y'
            self.config["rl"]["success_reward_factor"] = float(input("Factor de recompensa para trayectorias exitosas: ") or 10.0)
            agent_mode = input("Modo de agente (1: Agentes Q individuales, 2: Un solo agente Q): ")
            if agent_mode == '1':
              self.get_pidqlearning_config_individual(self.config["rl"]["sub_agents"])
            elif agent_mode == '2':
              self.get_pidqlearning_config_single(self.config["rl"]["sub_agents"])

        #Configuración para guardar las Q-Tables
        self.config["rl"]["q_table_save_frequency"] = int(input("Cada cuántos episodios guardar la Q-table? ") or 100)



    def get_qlearning_config(self, config_dict: Dict):
        print("  Parámetros de QLearning:")
        config_dict["learning_rate"] = float(input("    Tasa de aprendizaje: "))
        config_dict["discount_factor"] = float(input("    Factor de descuento: "))
        config_dict["exploration_rate"] = float(input("    Tasa de exploración inicial: "))
        config_dict["exploration_decay"] = float(input("    Decaimiento de la exploración: "))
        config_dict["min_exploration_rate"] = float(input("    Tasa de exploración mínima: "))
        action_space_str = input("    Espacio de acciones (variaciones, separadas por comas, ej. -0.1,0,0.1): ")
        config_dict["actions"] = [float(x) for x in action_space_str.split(',')]

        variables_to_include = input("    Variables a incluir en el estado (separadas por comas, ej. theta,theta_dot,kp): ").split(',')
        config_dict["variables_to_include"] = [v.strip() for v in variables_to_include]
        config_dict["q_table_filename"] = input("    Nombre del archivo para cargar/guardar la Q-table (opcional): ") or None

    def get_pidqlearning_config_individual(self, config_dict: Dict):
        print("  Configuración de agentes Q individuales para cada ganancia (Kp, Ki, Kd):")
        for gain in ["kp", "ki", "kd"]:
            print(f"    Configuración para {gain}:")
            config_dict[gain] = {"type": "QLearning", "parameters": {}}
            self.get_qlearning_config(config_dict[gain]["parameters"]) #Reutiliza

    def get_pidqlearning_config_single(self, config_dict:Dict):
        print(" Configuración del agente Q unico (PID)")
        config_dict['kp'] = {"type": "SingleQAgent", "parameters": {}}
        #Se configura un solo agente, pero igual se debe especificar las acciones de los otros
        for gain in ["kp", "ki", "kd"]:
            print(f"    Configuración para {gain}:")
            action_space_str = input(f"    Espacio de acciones para {gain} (variaciones, separadas por comas, ej. -0.1,0,0.1): ")
            config_dict[gain]["parameters"]['actions'] = [float(x) for x in action_space_str.split(',')]
        print(" Parametros generales")
        self.get_qlearning_config(config_dict['kp']["parameters"]) #Se reusa, pero ahora solo para setear los parametros generales

    def get_reward_config(self):
        print("\n--- Configuración de la Función de Recompensa ---")
        reward_type = input("Tipo de función de recompensa (ej. GaussianReward): ")
        self.config["reward"]["type"] = reward_type

        if reward_type == "GaussianReward":
            print("  Parámetros de GaussianReward:")
            state_variables_str = input("    Variables de estado (separadas por comas, ej. theta,x_dot): ")
            self.config["reward"]["parameters"]["state_variables"] = [v.strip() for v in state_variables_str.split(',')]

            std_devs_str = input("    Desviaciones estándar (separadas por comas, ej. 0.2,1.0): ")
            self.config["reward"]["parameters"]["std_devs"] = [float(x) for x in std_devs_str.split(',')]

            weights_str = input("    Pesos (separados por comas, ej. 1.0,0.5): ")
            self.config["reward"]["parameters"]["weights"] = [float(x) for x in weights_str.split(',')]
            self.config["reward"]["parameters"]["use_next_state"] = input("    Usar el siguiente estado para la recompensa? (y/n): ").lower() == 'y'


    def get_state_config(self):
        print("\n--- Configuración de la Discretización del Estado ---")
        print("Para cada variable, define si está habilitada, los valores mínimo y máximo, y el número de bins.")
        for var in ["x", "x_dot", "theta", "theta_dot","kp","ki","kd"]:
            enabled = input(f"  Habilitar variable {var}? (y/n): ") == 'y'
            self.config["state"][var] = {"enabled": enabled}
            if enabled:
                self.config["state"][var]["min"] = float(input(f"    Valor mínimo para {var}: "))
                self.config["state"][var]["max"] = float(input(f"    Valor máximo para {var}: "))
                self.config["state"][var]["bins"] = int(input(f"    Número de bins para {var}: "))