import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Batcher, close_obj
from torch_utils import tensor


class BaseAgent:
    def __init__(self, config):
        self.config = config

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_step(self, state):
        raise Exception('eval_step not implemented')

    def eval_episode(self):

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations

        total_rewards = 0
        while True:
            actions, log_probs, _, values = self.network(states)
            env_info = self.env.step(actions.cpu().detach().numpy())[self.brain_name]
            states = env_info.vector_observations  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            total_rewards += reward
            if done:
                break
        return total_rewards

    def eval_episodes(self):
        rewards = []
        for ep in range(self.config.eval_episodes):
            rewards.append(self.eval_episode())
        print('\nEvaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))


class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.env = config.env
        self.brain_name = self.env.brain_names[0]

        self.actor_critic = config.actor_critic_fn()

        self.opt_act = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=1e-4)
        self.opt_crt = torch.optim.Adam(self.actor_critic.critic.parameters(), lr=1e-3)
        self.opt = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=1e-4)

        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.states = env_info.vector_observations
        self.states = config.state_normalizer(self.states)

    def step(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, _, values = self.actor_critic(states)

            env_info = self.env.step(actions.cpu().detach().numpy())[self.brain_name]
            next_states = env_info.vector_observations  # get the next state
            rewards = np.array(env_info.rewards)  # get the reward
            terminals = np.array(env_info.local_done)  # see if episode has finished

            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0

            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.actor_critic(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()

        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, entropy_loss, values = self.actor_critic(sampled_states, sampled_actions)

                # critic training
                value_loss = 0.5 * F.mse_loss(sampled_returns, values)

                self.opt_crt.zero_grad()
                value_loss.backward()
                self.opt_crt.step()

                # actor training
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()

                self.opt_act.zero_grad()
                policy_loss.backward()
                self.opt_act.step()

                '''
                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), config.gradient_clip)
                self.opt.step()
                '''

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
