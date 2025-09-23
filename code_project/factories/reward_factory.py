from components.rewards.gaussian_reward import GaussianReward

class RewardFactory:
    @staticmethod
    def create_reward_function(reward_config):
        reward_type = reward_config.get('type')

        if reward_type == 'gaussian':
            return GaussianReward(**reward_config['params'])

        raise ValueError(f"Reward '{reward_type}' not recognized.")
