import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from torch_utils import tensor
from config import Config


class BaseNet:
    def __init__(self):
        pass


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


'''
class FCBody(nn.Module):
    def __init__(self, state_size, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCBody, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_units),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
        )

        self.model.apply(self.init_weights)
        self.feature_dim = fc2_units

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)

    def forward(self, state):
        return self.model(state)
'''


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256, 256), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()

        self.actor_body = actor_body
        self.critic_body = critic_body

        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.std = nn.Parameter(torch.ones(1, action_dim))

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, actor_body, critic_body)
        #self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        '''
        obs = tensor(obs)
        self.network.actor_body.eval()
        with torch.no_grad():
            phi_a = self.network.actor_body(obs)
        self.network.actor_body.train()

        self.network.critic_body.eval()
        with torch.no_grad():
            phi_v = self.network.critic_body(obs)
        self.network.critic_body.train()
        '''

        obs = tensor(obs)
        phi_a = self.network.actor_body(obs)
        phi_v = self.network.critic_body(obs)
        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, torch.cuda.FloatTensor(self.network.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v


