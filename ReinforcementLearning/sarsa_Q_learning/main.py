import gymnasium as gym
import matplotlib.pyplot as plt
from algorithms import QLearning, Sarsa
from utils import render_single_Q, evaluate_Q
import random
import numpy as np


class FrozenLakeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, turcated, info =  super().step(action)
        reward = -1.0 if terminated and not reward else reward # set reward to -1 when falling into a hole
        reward = 2.0 if reward == 1.0 else reward # set reward to 2 when reach the goal
        reward = -0.3 if reward == 0.0 else reward # add a -0.03 penalty each step
        return obs, reward, terminated, turcated, info

def smooth(data:np.ndarray, window_size=70):
    return np.convolve(data, np.ones(window_size, dtype=int), 'valid') / window_size

# Feel free to run your own debug code in main!
def main():
    num_episodes = 7000
    seed = 0
    
    #######=========================######
    # Do NOT modify
    map_name = "4x4"
    def make_env(is_render=False):
        return FrozenLakeRewardWrapper(
            gym.make('FrozenLake-v1', 
                     desc=None, map_name=map_name, is_slippery=True, render_mode="human" if is_render else None))
    env, render_env = make_env(), make_env(is_render=True)
    #######=========================######

    lr_list = [0.01, 0.1, 0.2]
    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 4 * 1))
    axes = axes.flatten()

    for idx, lr in enumerate(lr_list):
        Q1, Q_rewards1 = QLearning(env, num_episodes, lr=lr)
        Q2, Q_rewards2 = Sarsa(env, num_episodes, lr=lr)
        Q_rewards1  = smooth(Q_rewards1)
        Q_rewards2  = smooth(Q_rewards2)
        ax = axes[idx]
        ax.plot(range(len(Q_rewards1)), Q_rewards1, alpha=0.7, label='Q-Learning')
        ax.plot(range(len(Q_rewards2)), Q_rewards2, alpha=0.7, label='Sarsa')
        ax.legend(fontsize=9, loc='best')
        ax.set_title(f'Learning Rate = {lr}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Smoothed Total Reward', fontsize=10)

    for ax in axes[len(lr_list):]: ax.axis('off')
    plt.tight_layout()
    plt.savefig("lr_comparison_curves_more_penalty.png", dpi=300, bbox_inches='tight')
    plt.show()

    # render_single_Q(render_env, Q1)
    # render_single_Q(render_env, Q2)
    
    # evaluate_Q(env, Q1, 200)
    # print([int(np.argmax(i)) for i in Q1])
    
    # evaluate_Q(env, Q2, 200)
    # print([int(np.argmax(i)) for i in Q2])
    
    
    # plt.plot(range(len(Q_rewards1)), Q_rewards1)

    # Plot the learning curves of two methods
    # plt.plot(range(len(Q_rewards1)), Q_rewards1, alpha=0.7, label='Q-Learning')
    # plt.plot(range(len(Q_rewards2)), Q_rewards2, alpha=0.7,label='Sarsa')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Learning Curves')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
