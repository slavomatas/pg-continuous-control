import torch

from agent import PPOAgent
from config import Config
from model import Actor, Critic, ActorCritic
from torch_utils import select_device, set_one_thread, random_seed
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64", no_graphics=False)

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
config.rollout_length = 20 * 512
config.optimization_epochs = 10
config.num_mini_batches = 512
config.ppo_ratio_clip = 0.2
config.log_interval = 3 * 200 * 512
config.max_steps = 2e7
config.eval_episodes = 10
# config.logger = get_logger()

select_device(0)

print("GPU available: {}".format(torch.cuda.is_available()))
print("GPU tensor test: {}".format(torch.rand(3, 3).cuda()))

agent = PPOAgent(config)

random_seed()
config = agent.config

agent.actor_critic.load_state_dict(torch.load('../checkpoints/ppo_checkpoint.pth'))

score = 0  # initialize the score

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    for j in range(2000):
        action = agent.act(state)
        env_info = env.step(action.cpu().detach().numpy())[brain_name]
        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        state = next_state
        score += reward
        print('\rScore: {:.2f}'.format(score), end="")
        if done:
            break

env.close()