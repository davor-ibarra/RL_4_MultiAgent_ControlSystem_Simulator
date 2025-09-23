# interfaces/rl_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union # For type hinting
import numpy as np # Often needed for return types

# 7.1: Interfaz sin cambios funcionales, docstrings mejorados.
# 7.2: Mantener métodos helper opcionales en la interfaz ya que la implementación los tiene.
class RLAgent(ABC):
    """
    Interface for Reinforcement Learning agents.
    Defines methods for action selection, learning from experience,
    resetting, state building, and accessing internal state.
    """

    @abstractmethod
    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """
        Selects an action for each controllable dimension based on the current
        agent state and the agent's policy (e.g., epsilon-greedy).

        Args:
            agent_state_dict (Dict[str, Any]): The current state representation
                                               as perceived by the agent.

        Returns:
            Dict[str, int]: Maps each controllable dimension (e.g., 'kp')
                            to the selected action index (e.g., 0, 1, 2).
        """
        pass

    @abstractmethod
    def learn(self,
              current_agent_state_dict: Dict[str, Any], # S
              actions_dict: Dict[str, int],             # A
              # 7.3: Reward info pasada ahora por RewardStrategy.compute_reward_for_learning
              #      El SimulationManager pasará la info cruda aquí.
              reward_info: Union[float, Tuple[float, float], Dict[str, float]], # R (Raw info)
              next_agent_state_dict: Dict[str, Any],    # S'
              controller: Any,                          # Controller instance needed by strategy
              done: bool):                               # Done flag
        """
        Updates the agent's internal model (e.g., Q-tables, policy parameters)
        based on the experience tuple (S, A, R_info, S', controller, done).
        The agent will internally use its configured RewardStrategy to process
        R_info and calculate the final reward for the learning update (R_learn).

        Args:
            current_agent_state_dict (Dict[str, Any]): State S.
            actions_dict (Dict[str, int]): Actions A taken.
            reward_info (Union[float, Tuple[float, float], Dict[str, float]]):
                Raw reward information from the interval (R_real, (R_real, w_stab), R_diff_dict).
            next_agent_state_dict (Dict[str, Any]): State S'.
            controller (Any): The controller instance (passed to RewardStrategy).
            done (bool): Episode termination flag.
        """
        pass

    @abstractmethod
    def reset_agent(self):
        """
        Performs end-of-episode updates for the agent, such as decaying
        exploration rate (epsilon) and learning rate (alpha).
        May also reset other internal agent states if needed.
        """
        pass

    @abstractmethod
    def build_agent_state(self, raw_state_vector: Any, controller: Any, state_config_for_build: Dict) -> Dict[str, Any]:
        """
        Constructs the agent's specific state representation (dictionary) from the
        raw environment state vector and potentially controller parameters, based on
        the provided state configuration rules.

        Args:
            raw_state_vector (Any): Raw state vector from the environment/system.
            controller (Any): Controller instance (to get current gains if needed).
            state_config_for_build (Dict): Relevant section of the state discretization
                                           config defining enabled variables and rules.

        Returns:
            Dict[str, Any]: State dictionary used as input for the agent's policy
                            and learning updates. Should contain finite numerical values.
        """
        pass

    @abstractmethod
    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary containing the agent's internal state
        (e.g., Q-tables, visit counts, baseline tables) suitable for saving.

        Returns:
            Dict[str, Any]: Dictionary representing the agent's learnable parameters.
                            Structure depends on the agent type (e.g.,
                            {'q_tables': {...}, 'visit_counts': {...}, ...}).
        """
        pass

    # --- Optional Helper methods for logging/debugging (Kept in interface) ---
    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Returns the current exploration rate."""
        pass

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        """Returns the current learning rate."""
        pass

    @abstractmethod
    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Gets Q-values [Num Actions] for the given state for each enabled gain."""
        pass

    @abstractmethod
    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Gets visit counts N(s,a) [Num Actions] for the given state for each enabled gain."""
        pass

    @abstractmethod
    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """Gets baseline value B(s) for the given state for each enabled gain (if applicable)."""
        pass

    @abstractmethod
    def get_last_td_errors(self) -> Dict[str, float]:
        """Returns the TD errors calculated in the most recent learn step for each gain."""
        pass