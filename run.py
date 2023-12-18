import time

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import gym_race
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


def take_action(actor, state):
    state = torch.tensor(np.array([state]), dtype=torch.float)
    probs = actor(state)
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample()
    return action.item()


env_name = "Pyrace-v0"
env = gymnasium.make(env_name)
observation, info = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
actor = PolicyNet(state_dim, hidden_dim, action_dim)
actor.load_state_dict(torch.load("PPO_actor.pth"))
env.unwrapped.set_view(True)
NUM_EPISODES = 1
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    episode_reward = 0
    for t in range(2000):
        action = take_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
        state = next_state
        env.render()
    print("Reward:", episode_reward)
    print("Steps:", t)
env.close()
