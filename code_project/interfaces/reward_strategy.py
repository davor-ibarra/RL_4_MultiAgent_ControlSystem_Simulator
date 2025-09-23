# interfaces/reward_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Union, Tuple, Optional, List

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

class RewardStrategy(ABC):
    # --- Atributos declarativos (a ser definidos/sobrescritos por implementaciones) ---
    needs_virtual_simulation: bool = False
    """
    Flag que indica si esta estrategia requiere que SimulationManager 
    ejecute un VirtualSimulator para el cálculo de recompensas contrafactuales.
    Si es True, SimulationManager intentará resolver y usar una instancia de VirtualSimulator.
    """

    required_auxiliary_tables: List[str] = []
    """
    Lista de nombres de tablas auxiliares que esta estrategia espera que el RLAgent maneje.
    El agente puede usar esto en su inicialización para crear las tablas necesarias 
    (e.g., Q-tables adicionales, tablas de baseline, contadores de visita específicos).
    Ejemplo: ['baseline'] para una estrategia que usa un baseline almacenado por el agente.
    """

    @abstractmethod
    def compute_reward_for_learning(
        self,
        # --- Context for the update ---
        gain: str, # La "ganancia" o variable base del agente/Q-table que se está actualizando (e.g., 'kp', 'ki', 'kd')
        agent: 'RLAgent', # La instancia del agente que está aprendiendo
        controller: 'Controller', # La instancia del controlador actual
        # --- State Information ---
        current_agent_state_dict: Dict[str, Any], # Estado S (discretizado o continuo según el agente) en formato de diccionario
        current_state_indices: tuple,             # Índices discretos para S (para la tabla Q de 'gain') si aplica
        # --- Action Information ---
        actions_dict: Dict[str, int],             # Acciones A tomadas para todas las gains/agentes en el intervalo
        action_taken_idx: int,                    # Acción (e.g., 0,1,2) para la 'gain'/agente específico que se está actualizando
        # --- Raw Reward/Stability Information from the interval ---
        # 'interval_reward' es la recompensa acumulada del entorno real durante el intervalo.
        # 'avg_stability_score' es el promedio de w_stab (stability_score) del entorno real durante el intervalo.
        interval_reward: float,                   # R_real: Recompensa real acumulada
        avg_stability_score: float,               # w_stab promedio del intervalo real (stability_score)
        # --- Pre-calculated Differential Rewards (for Echo Baseline like strategies) ---
        # 'reward_dict' puede contener recompensas diferenciales precalculadas, e.g., {'kp': R_diff_kp, ...}
        # o ser None si la estrategia no las necesita o no fueron calculadas.
        reward_dict: Optional[Dict[str, float]],
        # --- Optional Extra Arguments ---
        **kwargs: Any # Para futura extensibilidad sin romper la interfaz
    ) -> float:
        """
        Calcula la recompensa de aprendizaje (R_learn) que el agente utilizará para 
        actualizar sus tablas de valor (e.g., Q-table) para una 'gain' específica.

        Esta función es el núcleo de la estrategia de recompensa. Puede simplemente
        devolver `interval_reward`, o puede calcular una recompensa más compleja
        basada en baselines, recompensas contrafactuales, etc.

        También puede interactuar con el `agent` para leer/actualizar tablas auxiliares
        (e.g., actualizar un baseline B(S) usando `agent.update_auxiliary_table_value`).

        Args:
            gain (str): Identificador de la ganancia o sub-agente para el cual se calcula R_learn.
            agent (RLAgent): La instancia del agente.
            controller (Controller): La instancia del controlador.
            current_agent_state_dict (Dict[str, Any]): Estado S al inicio del intervalo.
            current_state_indices (tuple): Índices del estado S para la tabla de 'gain'.
            actions_dict (Dict[str, int]): Acciones tomadas para todas las gains.
            action_taken_idx (int): Acción específica tomada para 'gain'.
            interval_reward (float): Recompensa real del intervalo.
            avg_stability_score (float): Estabilidad promedio del intervalo real.
            reward_dict (Optional[Dict[str, float]]): Recompensas diferenciales (si aplica).
            **kwargs: Argumentos adicionales para extensibilidad.

        Returns:
            float: La recompensa de aprendizaje (R_learn) para la actualización del Q-value.
                   Debe ser un valor numérico finito.
        """
        pass