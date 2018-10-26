import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from torch_utils import tensor
from config import Config


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(Actor, self).__init__()

        # Action mean
        self.mu = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.Tanh(),
            nn.Linear(fc1_units, fc2_units),
            nn.Tanh(),
            nn.Linear(fc2_units, action_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        return self.mu(state)


class Critic(nn.Module):
    def __init__(self, state_size, fc1_units=128, fc2_units=128):
        super(Critic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1),  # Value function
        )

    def forward(self, state):
        return self.value(state)


class ActorCritic(nn.Module):
    def __init__(self,
                 actor=None,
                 critic=None):
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic
        self.to(Config.DEVICE)

    def forward(self, state, action=None):
        state = tensor(state)
        mean = self.actor(state)
        values = self.critic(state)
        dist = torch.distributions.Normal(mean, torch.cuda.FloatTensor(self.actor.logstd.exp()))
        if action is None:
            action = dist.sample()
        log_probs = dist.log_prob(action)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)
        dist_entropy = dist.entropy()
        return action, log_probs, dist_entropy, values


