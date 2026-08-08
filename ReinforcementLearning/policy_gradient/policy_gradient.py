import gymnasium as gym
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import random

# see https://gymnasium.farama.org/environments/classic_control/cart_pole/
# understand environment, state, action and other definitions first before your dive in.

ENV_NAME = 'CartPole-v1'

# Hyper Parameters
# Following params work well if your implement Policy Gradient correctly.
# You can also change these params.
EPISODE = 3000  # total training episodes
STEP = 5000  # step limitation in an episode
EVAL_EVERY = 10  # evaluation interval
TEST_NUM = 5  # number of tests every evaluation
GAMMA = 0.95  # discount factor
LEARNING_RATE = 3e-3  # learning rate for mlp and ac


# A simple mlp implemented by PyTorch #
# it receives (N, D_in) shaped torch arrays, where N: the batch size, D_in: input state dimension
# and outputs the possibility distribution for each action and each sample, shaped (N, D_out)
# e.g. 
# state = torch.randn(10, 4)
# outputs = mlp(state)  #  output shape is (10, 2) in CartPole-v0 Game
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class AC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, output_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class REINFORCE:
    def __init__(self, env):
        # init parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        self.last_state = None
        self.net = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def predict(self, observation, deterministic=False):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_score = self.net(observation)
        probs = F.softmax(action_score, dim=1)
        m = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = m.sample()
        return action, probs

    def store_transition(self, s, a, p, r):
        self.states.append(s)
        self.actions.append(a)
        self.action_probs.append(p)
        self.rewards.append(r)

    def learn(self):
        # Please make sure all variables used to calculate loss are of type torch.Tensor, or autograd may not work properly.
        # You need to calculate the loss of each step of the episode and store them in '''loss'''.
        # The variables you should use are: self.rewards, self.action_probs, self.actions.
        # self.rewards=[R_1, R_2, ...,R_T], self.actions=[A_0, A_1, ...,A_(T-1)]
        # self.action_probs corresponds to the probability of different actions of each timestep, see predict() for details

        loss = []
        # -------------------------------
        # Your code goes here
        # TODO Calculate the loss of each step of the episode and store them in '''loss'''
        # Stack action probabilities (preserves computation graph)
        probs = torch.cat(self.action_probs)          # (T, action_dim)
        actions = torch.cat(self.actions)              # (T,) from Categorical.sample()
        rewards = torch.tensor(self.rewards, dtype=torch.float32)  # (T,)

        T = len(rewards)
        # Compute discounted returns: G_t = R_{t+1} + γ * G_{t+1}
        returns = torch.zeros(T)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + GAMMA * G
            returns[t] = G
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Get log-probabilities of selected actions using gather
        selected_probs = torch.gather(probs, dim=1, index=actions.view(-1, 1)).view(-1)
        log_probs = torch.log(selected_probs)

        # Loss for each step: -log(π(a_t|s_t)) * G_t
        step_losses = -log_probs * returns
        loss = list(step_losses.unsqueeze(1))
        # -------------------------------

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()


class TDActorCritic(REINFORCE):
    def __init__(self, env):
        super().__init__(env)
        self.ac = AC(input_dim=self.state_dim, output_dim=self.action_dim)
        # override
        self.net = self.ac.pi
        self.done = None
        self.optim = torch.optim.Adam(self.ac.parameters(), lr=LEARNING_RATE)

    def make_batch(self):
        done_lst = [1.0 if i != len(self.states) - 1 else 0.0 for i in range(len(self.states))]

        self.last_state = torch.tensor(self.last_state, dtype=torch.float).reshape(1, -1)
        self.states = torch.tensor(np.array(self.states), dtype=torch.float)
        self.done = torch.tensor(done_lst, dtype=torch.float).reshape(-1, 1)
        self.actions = torch.tensor(self.actions, dtype=torch.int64).reshape(-1, 1)
        self.action_probs = torch.cat(self.action_probs)
        self.states_prime = torch.cat((self.states[1:], self.last_state))
        self.rewards = torch.tensor(self.rewards, dtype=torch.float).reshape(-1, 1) / 100.0

    def learn(self):
        # Please make sure all variables are of type torch.Tensor, or autograd may not work properly.
        # You only need to calculate the policy loss.
        # The variables you should use are: self.rewards, self.action_probs, self.actions, self.states_prime, self.states.
        # self.states=[S_0, S_1, ...,S_(T-1)], self.states_prime=[S_1, S_2, ...,S_T], self.done=[1, 1, ..., 1, 0]
        # Invoking self.ac.v(self.states) gives you [v(S_0), v(S_1), ..., v(S_(T-1))]
        # For the final timestep T, delta_T = R_T - v(S_(T-1)), v(S_T) = 0
        # You need to use .detach() to stop delta's gradient in calculating policy_loss, see value_loss for an example

        policy_loss = None
        td_target = None
        delta = None
        self.make_batch()
        # -------------------------------
        # Your code goes here
        # TODO Calculate policy_loss
        # Compute state values
        v = self.ac.v(self.states)               # (T, 1)
        v_next = self.ac.v(self.states_prime)    # (T, 1)

        # TD target: R_{t+1} + γ * v(S_{t+1}) * done_t
        # For the last step (done=0): td_target = R_T
        td_target = self.rewards + GAMMA * v_next * self.done

        # TD error: δ_t = td_target - v(S_t)
        delta = td_target - v

        # Log probabilities of selected actions
        selected_probs = torch.gather(self.action_probs, dim=1, index=self.actions)
        log_probs = torch.log(selected_probs)    # (T, 1)

        # Policy loss: -log(π(A_t|S_t)) * δ_t,  δ detached to stop gradient
        policy_loss = (-log_probs * delta.detach()).mean()
        # -------------------------------

        # compute value loss and total loss
        # td_target is used as a scalar here, and is detached to stop gradient
        value_loss = F.smooth_l1_loss(self.ac.v(self.states), td_target.detach())
        loss = policy_loss + value_loss

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()

def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    # initialize OpenAI Gym env and PG agent
    SEEDS = [0, 42, 123]
    all_train_losses = []
    all_eval_rewards = []

    for SEED in SEEDS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        
        train_env = gym.make(ENV_NAME)
        render_env = gym.make(ENV_NAME, render_mode="human")
        RENDER = False
        PLOT = True

        # For plotting
        train_losses, eval_rewards = [], []
        
        # You may uncomment the line below to enable rendering for visualization.
        # RENDER = True
        
        AC = True  # set to False to train REINFORCE, set to True to train TD Actor-Critic
        if AC: agent = TDActorCritic(train_env)
        else: agent = REINFORCE(train_env)

        print(f"\n===== Training with seed: {SEED} =====")
        for episode in range(EPISODE):
            # initialize task
            env = train_env
            state, _ = env.reset(seed=random.randint(0,1e10))
            agent.last_state = state
            # Train
            for step in range(STEP):
                action, probs = agent.predict(state)
                next_state, reward, terminated, turcated, _ = env.step(action.item())
                done = terminated or turcated
                agent.store_transition(state, action, probs, reward)
                state = next_state
                if done:
                    loss = agent.learn()
                    if PLOT:
                        train_losses.append(loss)
                    break

            # Test
            env = render_env if RENDER else train_env
            if episode % EVAL_EVERY == 0:
                total_reward = 0
                for i in range(TEST_NUM):
                    state, _ = env.reset(seed=random.randint(0,1e10))
                    for j in range(STEP):
                        action, _ = agent.predict(state, deterministic=True)
                        next_state, reward, terminated, turcated, _ = env.step(action.item())
                        done = terminated or turcated
                        total_reward += reward
                        state = next_state
                        if done:
                            break
                avg_reward = total_reward / TEST_NUM
                if PLOT:
                    eval_rewards.append(avg_reward)

                # Your avg_reward should reach 200(cartpole-v0)/500(cartpole-v1) after a number of episodes.
                print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)
        
        all_train_losses.append(train_losses)
        all_eval_rewards.append(eval_rewards)

    if PLOT:
        plt.figure(figsize=(12, 5))
        plt.suptitle(f'Training Curves of {"TD Actor-Critic" if AC else "REINFORCE"} on CartPole-v1 (3 Seeds)')
        
        # Loss subplot
        plt.subplot(1, 2, 1)
        for idx, seed in enumerate(SEEDS):
            smooth_loss = moving_average(all_train_losses[idx], 20)
            plt.plot(smooth_loss, label=f'Seed {seed}')
        plt.title('Training Loss (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        # Reward subplot
        plt.subplot(1, 2, 2)
        eval_episodes = list(range(0, EPISODE, EVAL_EVERY))  # generate x-axis ticks
        for idx, seed in enumerate(SEEDS):
            plt.plot(eval_episodes, all_eval_rewards[idx], label=f'Seed {seed}')
        plt.title('Evaluation Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.axhline(y=500, color='r', linestyle='--', label='Max Reward')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{"TD_AC" if AC else "REINFORCE"}_3seeds_training_curve.png')
        plt.show()


if __name__ == '__main__':
    main()
