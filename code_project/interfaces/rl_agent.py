from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union # For type hinting
import numpy as np # Often needed for return types

class RLAgent(ABC):
    """
    Interface for Reinforcement Learning agents.
    Defines methods for action selection, learning from experience,
    resetting, and accessing internal state for saving or logging.
    """

    @abstractmethod
    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """
        Selects an action for each controllable dimension (e.g., gains) based
        on the current agent state and the agent's policy (e.g., epsilon-greedy).

        Args:
            agent_state_dict (Dict[str, Any]): The current state representation
                                              as perceived by the agent.

        Returns:
            Dict[str, int]: A dictionary mapping each controllable dimension (e.g., 'kp', 'ki', 'kd')
                           to the selected action index (e.g., 0, 1, 2).
        """
        pass

    @abstractmethod
    def learn(self,
              current_agent_state_dict: Dict[str, Any],
              actions_dict: Dict[str, int],
              reward_info: Union[float, Tuple[float, float], Dict[str, float]],
              next_agent_state_dict: Dict[str, Any],
              done: bool):
        """
        Updates the agent's internal model (e.g., Q-tables, policy parameters)
        based on the experience tuple (S, A, R, S', done).
        The format of reward_info depends on the RewardStrategy being used.

        Args:
            current_agent_state_dict (Dict[str, Any]): State before the action (S).
            actions_dict (Dict[str, int]): Actions taken for all dimensions (A).
            reward_info (Union[float, Tuple[float, float], Dict[str, float]]):
                Reward information obtained during the interval. Can be:
                - float: Global reward (R_real).
                - Tuple[float, float]: (R_real, avg_w_stab) for Shadow Baseline.
                - Dict[str, float]: Differential rewards {'kp': R_diff_kp,...} for Echo Baseline.
            next_agent_state_dict (Dict[str, Any]): State after the action (S').
            done (bool): Flag indicating if the episode terminated after this transition.
        """
        pass

    @abstractmethod
    def reset_agent(self):
        """
        Performs end-of-episode updates for the agent, such as decaying
        exploration rate (epsilon) and learning rate (alpha).
        May also reset internal agent states if needed.
        """
        pass

    @abstractmethod
    def build_agent_state(self, raw_state_vector: Any, controller: Any, state_config_for_build: Dict) -> Dict[str, Any]:
        """
        Constructs the agent's specific state representation (dictionary) from the
        raw environment state vector and potentially controller parameters.

        Args:
            raw_state_vector (Any): The raw state vector from the environment/system.
            controller (Any): The controller instance (to get current gains if needed).
            state_config_for_build (Dict): The relevant section of the state discretization
                                           configuration defining enabled variables.

        Returns:
            Dict[str, Any]: The state dictionary used as input for the agent's policy
                            and learning updates.
        """
        pass

    @abstractmethod
    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary containing the agent's internal state
        (e.g., Q-tables, visit counts, baseline tables) suitable for saving to JSON/Excel.

        Returns:
            Dict[str, Any]: A dictionary representing the agent's learnable parameters.
                           Example structure: {'q_tables': {...}, 'visit_counts': {...}, 'baseline_tables': {...}}
        """
        pass

    # --- Optional Helper methods for logging/debugging ---
    @abstractmethod
    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Gets Q-values for all actions for the given state for enabled gains."""
        pass

    @abstractmethod
    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Gets visit counts for all actions for the given state for enabled gains."""
        pass

    @abstractmethod
    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """Gets baseline value B(s) for the given state for enabled gains (if applicable)."""
        pass

    @abstractmethod
    def get_last_td_errors(self) -> Dict[str, float]:
        """Returns the TD errors calculated in the most recent learn step."""
        pass