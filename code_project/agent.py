from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Dict, Any

class Agente(ABC):
    @abstractmethod
    def percibir_entorno(self, estado):
        pass

    @abstractmethod
    def decidir_accion(self):
        pass

    @abstractmethod
    def recibir_recompensa(self, recompensa, estado_siguiente):
        pass

    @abstractmethod
    def actualizar_politica(self):
        pass

    @abstractmethod
    def esta_activo(self):
        pass

class AgenteQLearning(Agente):
    def __init__(self, estados_discretos, acciones_discretas, parametros):
        self.estados_discretos = estados_discretos
        self.acciones_discretas = acciones_discretas
        self.parametros = parametros

        # Parámetros de aprendizaje
        self.alpha = parametros.get('alpha', 0.1)
        self.gamma = parametros.get('gamma', 0.99)
        self.epsilon = parametros.get('epsilon', 0.1)
        self.epsilon_min = parametros.get('epsilon_min', 0.01)
        self.epsilon_decay = parametros.get('epsilon_decay', 0.995)

        # Inicializar Q-table
        num_estados = np.prod([v['bins'] for v in self.estados_discretos.values()])
        num_acciones = len(acciones_discretas)
        self.Q = np.zeros((num_estados, num_acciones))

        # Mapeo de estados a índices
        self.estado_actual_idx = None
        self.accion_actual_idx = None

        # Estado de activación
        self.activo = True

    def discretizar_estado(self, estado_continuo):
        estado_discreto = {}
        for var, rango_bins in self.estados_discretos.items():
            valor = estado_continuo.get(var, 0.0)
            rango = rango_bins['rango']
            bins = rango_bins['bins']
            estado_discreto[var] = self.discretizar_variable(valor, rango, bins)
        return estado_discreto

    def obtener_indice_estado(self, estado_discreto):
        indices = []
        for var in sorted(estado_discreto.keys()):
            indices.append(estado_discreto[var])
        estado_idx = np.ravel_multi_index(indices, [self.estados_discretos[var]['bins'] for var in sorted(self.estados_discretos.keys())])
        return estado_idx

    def discretizar_variable(self, valor, rango, bins):
        min_val, max_val = rango
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        bin_idx = np.digitize([valor], bin_edges)[0] - 1
        bin_idx = max(0, min(bin_idx, bins - 1))
        return bin_idx

    def percibir_entorno(self, estado):
        estado_discreto = self.discretizar_estado(estado)
        self.estado_actual_idx = self.obtener_indice_estado(estado_discreto)

    def decidir_accion(self):
        if random.uniform(0, 1) < self.epsilon:
            self.accion_actual_idx = random.randint(0, len(self.acciones_discretas) - 1)
        else:
            self.accion_actual_idx = np.argmax(self.Q[self.estado_actual_idx, :])
        accion = self.acciones_discretas[self.accion_actual_idx]
        return accion

    def recibir_recompensa(self, recompensa, estado_siguiente):
        estado_siguiente_discreto = self.discretizar_estado(estado_siguiente)
        estado_siguiente_idx = self.obtener_indice_estado(estado_siguiente_discreto)

        td_target = recompensa + self.gamma * np.max(self.Q[estado_siguiente_idx, :])
        td_error = td_target - self.Q[self.estado_actual_idx, self.accion_actual_idx]
        self.Q[self.estado_actual_idx, self.accion_actual_idx] += self.alpha * td_error

        self.estado_actual_idx = estado_siguiente_idx

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def actualizar_politica(self):
        pass

    def esta_activo(self):
        return self.activo

    def activar(self):
        self.activo = True

    def desactivar(self):
        self.activo = False

class AgenteFactory:
    @staticmethod
    def crear_agente(config_agente: Dict[str, Any], variables_estado):
        tipo_agente = config_agente.get('tipo_agente')
        parametros = config_agente.get('parametros', {})

        estados_discretos = {}
        for var in variables_estado:
            rango = parametros.get(f'{var}_rango', (-1.0, 1.0))
            bins = parametros.get(f'{var}_bins', 10)
            estados_discretos[var] = {'rango': rango, 'bins': bins}

        acciones_discretas = parametros.get('acciones_discretas', [-1, 0, 1])

        if tipo_agente == 'Q-Learning':
            return AgenteQLearning(estados_discretos, acciones_discretas, parametros)
        else:
            raise ValueError(f"Tipo de agente '{tipo_agente}' no reconocido")