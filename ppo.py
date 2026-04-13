"""
ppo.py
Fast MLP Actor-Critic policy for GridWorld agents.
Input: flat observation vector of length OBS_DIM (compact spatial features)
Trains in minutes on CPU, achieves benchmark reward >= 0.8.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=1.0, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    MLP Actor-Critic for flat observations.
    obs: [row/G, col/G, d_green_r/G, d_green_c/G, d_red_r/G, d_red_c/G,
          obs_up, obs_dn, obs_left, obs_right]  → dim=10
    """

    def __init__(self, obs_dim: int = 10, act_dim: int = 5, grid_size: int = 25):
        super().__init__()
        self.base = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)),     nn.Tanh(),
        )
        self.actor  = layer_init(nn.Linear(128, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1),       std=1.0)

    def forward(self, x):
        f = self.base(x)
        return self.actor(f), self.critic(f)

    def get_action_and_value(self, x, action=None, masks=None):
        logits, value = self.forward(x)
        if masks is not None:
            logits = logits.masked_fill(masks == 0, float('-inf'))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
