import sys
import os
import numpy as np
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from utils.config.config_loader import load_and_validate_config
from components.environments.pendulum_environment import PendulumEnvironment
from components.rewards.lagrangian_reward_calculator import LagrangianRewardCalculator
from components.rewards.omni_reward_calculator import OmniRewardCalculator
from components.agents.pid_qlearning_agent import PIDQLearningAgent
from components.reward_strategies.base_reward_strategy import BaseRewardStrategy
from components.stability.stability_calculator import StabilityCalculator

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def verify_reward_flow():
    logger.info("Starting Reward System Verification...")

    # 1. Load Config
    main_config, _, _, _ = load_and_validate_config('super_config.yaml')
    if main_config is None:
        logger.error("Failed to load config.")
        return

    # Ensure Lagrangian is enabled for testing
    reward_setup = main_config['environment']['reward_setup']
    reward_setup['reward_calculation']['reward_low_observability']['enabled'] = True
    
    # 2. Create Components
    logger.info("Creating Components...")
    
    # Stability Calculator (Dependency)
    stability_calc = StabilityCalculator(main_config)
    
    # Environment
    env = PendulumEnvironment(main_config)
    
    # Reward Calculators
    lagrangian_calc = LagrangianRewardCalculator(main_config, stability_calc)
    omni_calc = OmniRewardCalculator(main_config, stability_calc)
    
    # Agent
    # Need to mock state_config and other params usually provided by DI or config
    agent_config = main_config['agent']['pid_q_learning']
    state_config = agent_config['state_space']
    num_actions = agent_config['action_space']['num_actions']
    
    agent = PIDQLearningAgent(
        reward_strategy=None, # Will set later
        state_config=state_config,
        num_actions=num_actions,
        main_config=main_config,
        **agent_config.get('hyperparameters', {})
    )
    
    # Reward Strategy
    strategy = BaseRewardStrategy(main_config)
    agent.set_reward_strategy(strategy)
    
    # 3. Test Lagrangian Flow
    logger.info("Testing Lagrangian Reward Flow...")
    env.reward_function = lagrangian_calc # Inject Lagrangian
    
    env.reset()
    next_state, reward_dict, stability, done = env.step()
    
    logger.info(f"Lagrangian Reward Output Type: {type(reward_dict)}")
    logger.info(f"Lagrangian Reward Output: {reward_dict}")
    
    if not isinstance(reward_dict, dict):
        logger.error("Lagrangian Calculator did not return a dictionary!")
        return
    
    if 'total_reward' not in reward_dict:
        logger.error("Lagrangian Reward dict missing 'total_reward'!")
        return

    # 4. Test Omni Flow
    logger.info("Testing Omni Reward Flow...")
    env.reward_function = omni_calc # Inject Omni
    
    env.reset()
    next_state, reward_dict_omni, stability, done = env.step()
    
    logger.info(f"Omni Reward Output Type: {type(reward_dict_omni)}")
    logger.info(f"Omni Reward Output: {reward_dict_omni}")
    
    if not isinstance(reward_dict_omni, dict):
        logger.error("Omni Calculator did not return a dictionary!")
        return

    # 5. Test Agent Learn
    logger.info("Testing Agent Learn with Dictionary...")
    
    # Mock inputs for learn
    current_s = agent.build_agent_state(env._create_state_dict(env.state), {})
    actions = agent.select_action(current_s)
    next_s = agent.build_agent_state(env._create_state_dict(next_state), {})
    
    # Construct reward_info as SimulationManager would
    reward_info = {
        'interval_reward': reward_dict['total_reward'],
        'reward_components': reward_dict,
        'avg_stability_score_interval': stability
    }
    
    try:
        metrics = agent.learn(
            current_agent_s_dict=current_s,
            taken_actions_map=actions,
            reward_info=reward_info,
            next_agent_s_prime_dict=next_s,
            controllers={},
            is_episode_done=False,
            episode_idx=0
        )
        logger.info(f"Agent Learn Metrics: {metrics}")
        logger.info("Agent Learn executed successfully with dictionary rewards.")
    except Exception as e:
        logger.error(f"Agent Learn failed: {e}", exc_info=True)
        return

    logger.info("VERIFICATION SUCCESSFUL: All checks passed.")

if __name__ == "__main__":
    verify_reward_flow()
