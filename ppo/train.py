from collections import deque

import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from agent import PPOAgent
from config import Config
from model import Actor, Critic, ActorCritic
from torch_utils import select_device, set_one_thread, random_seed
from unityagents import UnityEnvironment


def ppo():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64", no_graphics=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    config = Config()
    config.env = env

    config.actor_critic_fn = lambda: ActorCritic(actor=Actor(state_size, action_size),
                                                 critic=Critic(state_size))

    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 5
    config.rollout_length = 2048
    config.optimization_epochs = 5
    config.num_mini_batches = 512
    config.ppo_ratio_clip = 0.2
    config.log_interval = 10*2048
    config.max_steps = 2e7
    config.eval_episodes = 10
    # config.logger = get_logger()

    print("GPU available: {}".format(torch.cuda.is_available()))
    print("GPU tensor test: {}".format(torch.rand(3, 3).cuda()))

    agent = PPOAgent(config)

    random_seed()
    config = agent.config
    t0 = time.time()
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores

    while True:
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            for reward in rewards:
                scores.append(reward)
                scores_window.append(reward)
            agent.episode_rewards = []

            print('\r===> Average Score: {:d} episodes {:.2f}'.format(len(scores), np.mean(scores_window)))
            if np.mean(scores_window) >= 1.0:
                print('\nEnvironment solved in {:d}  episodes!\tAverage Score: {:.2f}'.format(len(scores_window),
                                                                                              np.mean(scores_window)))
                torch.save(agent.actor_critic.state_dict(), '../checkpoints/ppo_checkpoint.pth')
                break

            print('Total steps %d, returns %d/%.2f/%.2f/%.2f/%.2f (count/mean/median/min/max), %.2f steps/s' % (
                agent.total_steps, len(rewards), np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
                config.log_interval / (time.time() - t0)))

            t0 = time.time()

        agent.step()

    return scores


if __name__ == '__main__':
    set_one_thread()
    select_device(0)
    scores = ppo()

    # plot the scores
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
