# interfaces/rl_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List, TYPE_CHECKING # Añadir List
import numpy as np

if TYPE_CHECKING: # Evitar importación circular
    from interfaces.reward_strategy import RewardStrategy

class RLAgent(ABC):

    # --- MÉTODOS PRINCIPALES ---

    @abstractmethod
    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        pass

    @abstractmethod
    def learn(self,
              current_agent_state_dict: Dict[str, Any],
              actions_dict: Dict[str, int],
              reward_info: Union[float, Tuple[float, float], Dict[str, float]],
              next_agent_state_dict: Dict[str, Any],
              controller: Any, # Debería ser Controller interface
              done: bool):
        pass

    @abstractmethod
    def reset_agent(self):
        pass

    @abstractmethod
    def build_agent_state(self, raw_state_vector: Any, controller: Any, state_config_for_build: Dict) -> Dict[str, Any]:
        pass
    
    # --- MÉTODOS SECUNDARIOS ---

    @property
    @abstractmethod
    def reward_strategy(self) -> 'RewardStrategy': # Exponer la estrategia
        """Returns the reward strategy instance used by the agent."""
        pass

    @property
    @abstractmethod
    def epsilon(self) -> float:
        pass

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        pass
    
    @abstractmethod
    def get_agent_defining_vars(self) -> List[str]:
        """
        Devuelve una lista de strings que identifican las variables o "sub-agentes" 
        que este agente gestiona individualmente (ej: ['kp', 'ki', 'kd'] para PIDQLearningAgent).
        """
        pass
    
    @abstractmethod
    def should_episode_terminate_early(self) -> bool:
        pass

    @abstractmethod
    def get_last_early_termination_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary coscienza las últimas métricas relacionadas con la lógica de 
        terminación temprana para cada variable/agente que gestiona.
        Ej: {'kp': {'patience_M': 20, 'c_hat': 3, ...}, ...}
        """
        pass
    
    # --- MÉTODOS AUXILIARES ---

    @abstractmethod
    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_last_td_errors(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """
        Gets baseline value B(s) for the given state for each enabled gain.
        Specific to agents that use baselines (e.g., with ShadowBaselineStrategy).
        Implementations can return NaNs or default values if not applicable.
        """
        pass

    # --- Métodos para manejo genérico de tablas auxiliares (e.g., Baseline) ---
    @abstractmethod
    def get_auxiliary_table_value(self, table_name: str, gain: str, state_indices: tuple) -> Optional[float]:
        """
        Retrieves a value from a named auxiliary table (e.g., 'baseline') for a specific gain and state.
        Returns None if table or entry doesn't exist or not applicable.
        """
        pass

    @abstractmethod
    def update_auxiliary_table_value(self, table_name: str, gain: str, state_indices: tuple, value: float):
        """
        Updates a value in a named auxiliary table for a specific gain and state.
        The agent decides if/how to store this based on its internal structure.
        """
        pass

    @abstractmethod
    def get_auxiliary_table_names(self) -> List[str]:
        """
        Returns a list of names of auxiliary tables managed by the agent.
        Can be used by strategies to know what tables they might interact with.
        """
        pass