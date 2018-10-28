import copy
import os
from collections import deque

import numpy as np
import time
import torch
from agent import PPO
from arguments import get_args
from model import Policy
from storage import RolloutStorage
from unityagents import UnityEnvironment

args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    args = get_args()

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

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

    actor_critic = Policy(state_size, action_size, base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              state_size, action_size,
                              actor_critic.recurrent_hidden_state_size)

    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations

    rollouts.obs[0].copy_(obs[0])
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obs reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            '''
            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            '''

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           len(episode_rewards),
                           np.mean(episode_rewards),
                           np.median(episode_rewards),
                           np.min(episode_rewards),
                           np.max(episode_rewards), dist_entropy,
                           value_loss, action_loss))


if __name__ == "__main__":
    main()
