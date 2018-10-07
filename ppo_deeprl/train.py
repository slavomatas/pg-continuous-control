import time
import torch
import numpy as np

from agent import PPOAgent
from config import Config
from torch_utils import random_seed, select_device, set_one_thread
from model import FCBody, GaussianActorCriticNet
from task import ParallelizedTask


def run_steps(agent):
    random_seed()
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            agent.episode_rewards = []
            config.logger.info('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
                agent.total_steps, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()


def ppo_continuous():
    config = Config()
    config.num_workers = 1
    task_fn = lambda log_dir: Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(ppo_continuous.__name__))
    config.eval_env = task_fn(None)

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim))

    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.num_mini_batches = 32
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 2e7
    #config.logger = get_logger()
    run_steps(PPOAgent(config))


if __name__ == '__main__':
    # mkdir('data/video')
    # mkdir('dataset')
    # mkdir('log')
    set_one_thread()
    # select_device(-1)
    select_device(0)
    ppo_continuous()
