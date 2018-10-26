import os
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

sys.path.append("../")

from agent import Agent

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# Instantiate DDPG Agents
agents = [Agent(state_size=state_size, action_size=action_size, random_seed=0) for _ in range(len(env_info.agents))]

'''
obs, rew, done, info = {}, {}, {}, {}
for i, action in action_dict.items():
    obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
    if done[i]:
        self.dones.add(i)
done["__all__"] = len(self.dones) == len(self.agents)
return obs, rew, done, info
'''


def ddpg(n_episodes=1000, max_t=1000):
    """Deep Deterministic Policy Gradient

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    env_info = env.reset(train_mode=True)[brain_name]

    agent_scores = []
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
                    agents[i].step(state, actions[i], reward, next_state, done)
                    states[i] = next_state
                    scores[i] += reward

            if dones[]:
                break

        for i in range(len(env_info.agents)):
            scores_windows[i].append(scores[i])  # save most recent score
            agent_scores[i].append(scores[i])  # save most recent score

        '''
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            # torch.save(agent.actor_local.state_dict(), '../checkpoints/checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), '../checkpoints/checkpoint_critic.pth')
            break
        '''
    return agent_scores


scores = ddpg()

# plot the scores
fig = plt.figure()
# ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
