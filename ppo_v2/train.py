import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from agent import Actor, Critic, AgentA2C, ExperienceSource, RewardTracker
from unityagents import UnityEnvironment

# from tensorboardX import SummaryWriter

GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 5*2049


def test_net(net, env, brain_name, count=10, device="cpu"):
    rewards = 0.0
    steps = 0

    for _ in range(count):
        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        while True:
            # obs_v = float32_preprocessor([obs]).to(device)
            obs_v = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)

            env_info = env.step(action)[brain_name]
            obs = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            rewards += reward
            steps += 1
            if done:
                break

    return rewards / count, steps / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.is_available())
    print(torch.rand(3, 3).cuda())

    env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64", no_graphics=True)
    # test_env = UnityEnvironment(file_name="../Reacher_Linux/Reacher.x86_64", no_graphics=False)

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

    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score

    net_act = Actor(state_size, action_size).to(device)
    net_crt = Critic(state_size).to(device)
    print(net_act)
    print(net_crt)

    # writer = SummaryWriter(comment="-ppo_crawler")

    agent = AgentA2C(net_act, device=device)
    exp_source = ExperienceSource(env, agent, brain_name, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = []
    best_reward = None
    with RewardTracker() as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                # writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = test_net(net_act, env, brain_name, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        # fname = os.path.join(save_path, name)
                        # torch.save(net_act.state_dict(), fname)
                    best_reward = rewards

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)

            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

            # normalize advantages
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()

            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            # print("actor/critic training")

            for epoch in range(PPO_EPOCHES):
                for batch_ofs in range(0, len(trajectory), PPO_BATCH_SIZE):
                    states_v = traj_states_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                    actions_v = traj_actions_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                    batch_adv_v = traj_adv_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE].unsqueeze(-1)
                    batch_ref_v = traj_ref_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                    batch_old_logprob_v = old_logprob_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]

                    # critic training
                    # print("critic training batch_ofs {}".format(batch_ofs))
                    opt_crt.zero_grad()
                    value_v = net_crt(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                    loss_value_v.backward()
                    opt_crt.step()

                    # actor training
                    # print("actor training batch_ofs {}".format(batch_ofs))
                    opt_act.zero_grad()
                    mu_v = net_act(states_v)
                    logprob_pi_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                    ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    clipped_surr_v = batch_adv_v * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                    loss_policy_v.backward()
                    opt_act.step()

                    sum_loss_value += loss_value_v.item()
                    sum_loss_policy += loss_policy_v.item()
                    count_steps += 1

            trajectory.clear()

            '''
            print("Advantage", traj_adv_v.mean().item(), step_idx)
            print("Values", traj_ref_v.mean().item(), step_idx)
            print("Loss_policy", sum_loss_policy / count_steps, step_idx)
            print("Loss_value", sum_loss_value / count_steps, step_idx)
            '''
