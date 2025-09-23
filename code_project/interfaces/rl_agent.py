# interfaces/rl_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List, TYPE_CHECKING 
import numpy as np

if TYPE_CHECKING: # Evitar importación circular
    from interfaces.reward_strategy import RewardStrategy
    from interfaces.controller import Controller # Añadido para el type hint en learn

class RLAgent(ABC):

    # --- MÉTODOS PRINCIPALES DE INTERACCIÓN Y APRENDIZAJE ---

    @abstractmethod
    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """
        Selecciona una acción para cada variable de ganancia (o sub-agente) que gestiona,
        basado en el estado actual del agente y su política (e.g., epsilon-greedy sobre Q-values).

        Args:
            agent_state_dict (Dict[str, Any]): El estado actual relevante para el agente,
                                               generalmente un diccionario de características.
                                               Ej: {'angle': 0.1, 'kp': 20.0, ...}

        Returns:
            Dict[str, int]: Un diccionario donde las claves son los identificadores de las
                            variables de ganancia (e.g., 'kp', 'ki', 'kd') y los valores son
                            los índices de acción seleccionados (e.g., 0, 1, 2).
        """
        pass

    @abstractmethod
    def learn(self,
              current_agent_state_dict: Dict[str, Any], # S
              actions_dict: Dict[str, int],             # A
              reward_info: Dict[str, Any],              # R_info
              next_agent_state_dict: Dict[str, Any],    # S'
              controller: 'Controller',                 # Controlador actual para acceso a parámetros o estado si es necesario por la estrategia
              done: bool                                # Flag de terminación del episodio
             ) -> Dict[str, float]:                     # Devolverá un dict con métricas de aprendizaje
        """
        Actualiza los conocimientos del agente (e.g., Q-tables, modelos) basado en la
        transición (S, A, R_info, S') y si el episodio ha terminado.

        La `reward_info` puede ser una recompensa escalar simple, una tupla (recompensa, estabilidad),
        o un diccionario de recompensas (e.g., para estrategias tipo Echo Baseline).
        El agente internamente usará su `RewardStrategy` para procesar `reward_info`
        y obtener la R_learn final para cada actualización de Q-value.

        Args:
            current_agent_state_dict (Dict[str, Any]): Estado S al inicio del intervalo.
            actions_dict (Dict[str, int]): Acciones A tomadas para todas las gains.
            reward_info (Union[float, Tuple[float, float], Dict[str, float]]):
                Información de recompensa del intervalo. La estructura depende de la
                configuración de la simulación (e.g., si se usa Echo Baseline).
            next_agent_state_dict (Dict[str, Any]): Estado S' al final del intervalo.
            controller (Controller): La instancia del controlador actual.
            done (bool): True si el episodio terminó después de este intervalo.
        Returns:
            Dict[str, float]: Un diccionario con las métricas de aprendizaje generadas,
                              como {'r_learn_kp': ..., 'td_error_kp': ...}.
        """
        pass

    @abstractmethod
    def reset_agent(self):
        """
        Resetea el estado interno del agente al final de un episodio.
        Esto típicamente incluye:
        - Actualizar tasas de exploración (epsilon) y aprendizaje (alpha) si usan decaimiento.
        - Resetear contadores o estados para lógicas de terminación temprana específicas del agente.
        - Cualquier otra limpieza necesaria antes del próximo episodio.
        No necesariamente resetea las tablas Q aprendidas.
        """
        pass

    @abstractmethod
    def build_agent_state(self,
                          raw_state_vector: Any,
                          controller: 'Controller', # Para obtener ganancias actuales si son parte del estado
                          state_config_for_build: Dict, # La sección 'state_config' del agente
                          env_state_dict: Dict[str, Any],
                         ) -> Dict[str, Any]:
        """
        Construye el diccionario de estado del agente (agent_state_dict) a partir del
        vector de estado crudo del sistema y el estado actual del controlador.

        Args:
            raw_state_vector (Any): El vector de estado del sistema dinámico (e.g., [x, x_dot, theta, theta_dot]).
            controller (Controller): La instancia del controlador actual, para extraer parámetros como Kp, Ki, Kd si son parte del estado del agente.
            state_config_for_build (Dict): La configuración de estado del agente (usualmente `self.state_config`)
                                           que define qué variables incluir y cómo.
            env_state_dict (Dict[str, Any]): El diccionario de estado con nombres explícitos
                                             proporcionado por el entorno (e.g., {'angle': 0.1}).

        Returns:
            Dict[str, Any]: El estado del agente como un diccionario de características.
                            Ej: {'angle': 0.05, 'angular_velocity': -0.1, 'kp_value': 25.0}
        """
        pass

    @abstractmethod
    def set_reward_strategy(self, strategy: 'RewardStrategy'):
        """
        Establece la estrategia de recompensa para el agente.
        Este método permite la inyección tardía para romper dependencias circulares.
        """
        pass

    # --- PROPIEDADES Y MÉTODOS DE CONFIGURACIÓN/ESTADO INTERNO ---

    @property
    @abstractmethod
    def reward_strategy(self) -> 'RewardStrategy':
        """
        Retorna la instancia de RewardStrategy que el agente está utilizando
        para interpretar la información de recompensa y calcular R_learn.
        """
        pass

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Tasa de exploración actual del agente."""
        pass

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        """Tasa de aprendizaje actual del agente."""
        pass

    @property
    @abstractmethod
    def early_termination_enabled(self) -> bool: # Nueva propiedad
        """Indica si la lógica de terminación temprana del agente está habilitada."""
        pass

    @abstractmethod
    def get_agent_defining_vars(self) -> List[str]:
        """
        Devuelve una lista de strings que identifican las variables o "sub-agentes"
        que este agente gestiona individualmente (ej: ['kp', 'ki', 'kd'] para un agente
        que aprende los parámetros PID). Esto es crucial para que las estrategias
        y los sistemas de logging puedan iterar sobre cada "cabeza" de aprendizaje.
        """
        pass

    @abstractmethod
    def should_episode_terminate_early(self) -> bool:
        """
        Indica si, basado en la lógica interna del agente (e.g., contadores de no mejora),
        el episodio actual debería terminarse prematuramente.
        SimulationManager consultará esto.
        """
        pass

    @abstractmethod
    def get_last_early_termination_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Devuelve un diccionario con las últimas métricas relacionadas con la lógica de
        terminación temprana para cada variable/sub-agente que gestiona.
        Facilita el logging y análisis de cómo opera la terminación temprana.
        Ej: {'kp': {'patience_M': 20, 'no_improvement_counter_c_hat': 3, ...}, ...}
        """
        pass

    # --- MÉTODOS AUXILIARES PARA INSPECCIÓN Y GUARDADO ---

    @abstractmethod
    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """
        Prepara y devuelve el estado completo del agente (Q-tables, contadores de visita,
        tablas auxiliares, hiperparámetros relevantes) en un formato serializable (diccionario),
        adecuado para guardar en disco (e.g., JSON).
        """
        pass

    @abstractmethod
    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """
        Para un `agent_state_dict` dado, devuelve los Q-values [Q(S,a0), Q(S,a1), ...]
        para cada variable de ganancia (sub-agente) que gestiona.
        Devuelve NaNs si el estado no es válido o no hay Q-values.
        """
        pass

    @abstractmethod
    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """
        Para un `agent_state_dict` dado, devuelve los contadores de visita N(S,A)
        para cada variable de ganancia (sub-agente).
        Devuelve -1 o NaNs si no hay contadores aplicables.
        """
        pass

    @abstractmethod
    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """
        Obtiene el valor del baseline B(S) para el `agent_state_dict` dado, para cada
        variable de ganancia (sub-agente) habilitada que use baselines.
        Específico para agentes que interactúan con estrategias que usan baselines
        (e.g., `ShadowBaselineRewardStrategy`).
        Implementaciones pueden devolver NaNs o valores por defecto si no aplica.
        """
        pass

    # --- Métodos para manejo genérico de tablas auxiliares (e.g., Baseline, Modelos del Mundo) ---
    @abstractmethod
    def get_auxiliary_table_value(self, table_name: str, gain: str, state_indices: tuple) -> Optional[float]:
        """
        Recupera un valor de una tabla auxiliar nombrada (e.g., 'baseline', 'visit_counts_baseline')
        para una 'gain' (sub-agente) específica y un conjunto de 'state_indices'.
        Devuelve None si la tabla o la entrada no existe, o si no aplica.
        """
        pass

    @abstractmethod
    def update_auxiliary_table_value(self, table_name: str, gain: str, state_indices: tuple, value: float):
        """
        Actualiza un valor en una tabla auxiliar nombrada para una 'gain' específica
        y 'state_indices'. El agente decide cómo y dónde almacenar esto basado en su
        estructura interna y las tablas que `required_auxiliary_tables` de su
        `RewardStrategy` le indicó que necesitaba.
        """
        pass

    @abstractmethod
    def get_auxiliary_table_names(self) -> List[str]:
        """
        Devuelve una lista de nombres de las tablas auxiliares que el agente
        está gestionando internamente. Esto puede ser usado por las `RewardStrategy`
        para saber con qué tablas pueden interactuar.
        """
        pass

    @abstractmethod
    def get_params_log(self) -> Dict[str, Any]:
        """
        Returns a dictionary of agent parameters for logging purposes.
        This method centralizes the exposure of loggable data.
        """
        pass