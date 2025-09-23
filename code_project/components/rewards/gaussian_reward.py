from interfaces.reward_function import RewardFunction
import math

class GaussianReward(RewardFunction):

    def __init__(self, weights, scales):
        self.weights = weights
        self.scales = scales

    def calculate(self, state, action, next_state, t):
        angle_norm = next_state[2]/self.scales['angle']
        vel_norm = next_state[3]/self.scales['angular_velocity']
        force_norm = action/self.scales['force']
        pos_cart_norm = next_state[0]/self.scales['cart_position']
        vel_cart_norm = next_state[1]/self.scales['cart_position']
        time_norm = t/self.scales['time']

        pendulum_stability_reward = self.weights['stability'] * math.exp(-angle_norm**2 -vel_norm**2)
        force_penalty = self.weights['force'] * math.exp(-force_norm**2)
        cart_stability_reward = self.weights['cart'] * math.exp(-vel_cart_norm**2)
        time_reward = self.weights['time'] * math.exp(-time_norm**2)

        return pendulum_stability_reward + force_penalty + cart_stability_reward #- time_reward
