# interfaces/reward_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Union, Tuple, Optional, List

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

class RewardStrategy(ABC):
    # --- Atributos declarativos (a ser definidos/sobrescritos por implementaciones) ---
    needs_virtual_simulation: bool = False
    """Flag que indica si esta estrategia requiere que SimulationManager ejecute un VirtualSimulator."""

    required_auxiliary_tables: List[str] = []
    """
    Lista de nombres de tablas auxiliares que esta estrategia espera que el agente maneje.
    El agente puede usar esto en su inicialización para crear las tablas necesarias.
    Ejemplo: ['baseline'] para ShadowBaselineStrategy.
    """

    @abstractmethod
    def compute_reward_for_learning(
        self,
        # --- Context for the update ---
        gain: str, # La "ganancia" o variable base del agente/Q-table que se está actualizando
        agent: 'RLAgent',
        controller: 'Controller',
        # --- State Information ---
        current_agent_state_dict: Dict[str, Any], # Estado S (dict)
        current_state_indices: tuple,             # Índices discretos para S (para la tabla de 'gain')
        # --- Action Information ---
        actions_dict: Dict[str, int],             # Acciones A tomadas para todas las gains/agentes
        action_taken_idx: int,                    # Acción (0,1,2) para la 'gain'/agente específico
        # --- Raw Reward/Stability Information from the interval ---
        interval_reward: float,                   # R_real: Recompensa real acumulada (o NaN si no aplica)
        avg_w_stab: float,                        # w_stab promedio (o NaN si no aplica)
        # --- Pre-calculated Differential Rewards (for Echo Baseline) ---
        reward_dict: Optional[Dict[str, float]],  # Dict de R_diff (e.g., {'kp': R_diff_kp}) o None
        # --- Optional Extra Arguments ---
        **kwargs
    ) -> float:
        """
        Calcula el R_learn para la actualización del agente.
        También puede interactuar con el agente para actualizar tablas auxiliares.
        """
        pass