import torch

from config import Config
from utils import run_steps
from torch_utils import random_seed, select_device, set_one_thread

from agent import PPOAgent
from model import FCBody, GaussianActorCriticNet

from unityagents import UnityEnvironment


def ppo_continuous():

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    config.network_fn = lambda: GaussianActorCriticNet(
        state_size, action_size, actor_body=FCBody(state_size),
        critic_body=FCBody(state_size))

    """
    config.network_fn = lambda: CategoricalActorCriticNet(
        state_size, action_size, actor_body=FCBody(state_size),
        critic_body=FCBody(state_size))
    """

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
