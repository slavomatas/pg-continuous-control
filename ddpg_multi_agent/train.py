import os
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

sys.path.append("../")

from agent import Agent, ReplayBuffer, BUFFER_SIZE, BATCH_SIZE


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("GPU available: {}".format(torch.cuda.is_available()))
print("GPU tensor test: {}".format(torch.rand(3, 3).cuda()))

env = UnityEnvironment(file_name="../Reacher_Linux_Multi/Reacher.x86_64", no_graphics=True)

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

state = env_info.vector_observations  # get the current state
score = 0  # initialize the score

# Create replay buffer
replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=0)

# Create DDPG Agents with shared replied buffer
agents = [Agent(state_size=state_size, action_size=action_size, random_seed=0, replay_buffer=replay_buffer) for _ in range(len(env_info.agents))]


def ddpg_multi_agents(n_episodes=1000, max_t=1000):
    """Deep Deterministic Policy Gradient

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    env_info = env.reset(train_mode=True)[brain_name]

    agents_scores = [[] for _ in range(len(env_info.agents))]
    scores_windows = [deque(maxlen=100) for _ in range(len(env_info.agents))]

    for i_episode in range(1, n_episodes + 1):

        actions = []
        dones = [0 for _ in range(len(env_info.agents))]
        scores = [0 for _ in range(len(env_info.agents))]

        env_info = env.reset(train_mode=True)[brain_name]
        states = [env_info.vector_observations[i] for i in range(len(env_info.agents))]

        for t in range(max_t):

            actions = [agents[i].act(states[i], True)[0] for i in range(len(env_info.agents))]
            env_info = env.step(np.concatenate(actions))[brain_name]

            for i in range(len(env_info.agents)):
                if not dones[i]:
                    next_state = env_info.vector_observations[i]  # get the next state
                    reward = env_info.rewards[i]  # get the reward
                    dones[i] = env_info.local_done[i]  # see if episode has finished
                    agents[i].step(states[i], actions[i], reward, next_state, dones[i])
                    states[i] = next_state
                    scores[i] += reward

            if np.all(np.array(dones)):
                break

        for i in range(len(env_info.agents)):
            scores_windows[i].append(scores[i])  # save most recent score
            agents_scores[i].append(scores[i])  # save most recent score

        for i in range(len(env_info.agents)):
            print('\rAgent {} Episode {}\tAverage Score: {:.2f}'.format(i, i_episode, np.mean(scores_windows[i])))
            if i_episode % 100 == 0:
                print('\rAgent {} Episode {}\tAverage Score: {:.2f}'.format(i, i_episode, np.mean(scores_windows[i])))
            if np.mean(scores_windows[i]) >= 30.0:
                print(
                    '\nAgent {} Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, i_episode - 100,
                                                                                                    np.mean(
                                                                                                        scores_windows[
                                                                                                            i])))
                torch.save(agents[i].actor_local.state_dict(), '../checkpoints/checkpoint_actor_'+i+'.pth')
                torch.save(agents[i].critic_local.state_dict(), '../checkpoints/checkpoint_critic_'+i+'.pth')
                break

    return agents_scores


agents_scores = ddpg_multi_agents()

'''
# plot the scores
fig = plt.figure()
# ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
'''
