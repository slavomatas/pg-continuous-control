# Implementing a DQN Agent

## Summary

In this project I trained a DQN reinforcement learning agent to reach a score of +30 on average over 100 episodes in the Unity-ML environment.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

I have implemented following DRL agents

# 1. DDPG - Deterministic Deep Policy Gradient agent

DDPG is a variation of the A2C method, but has a very nice property of being off-policy.
In A2C, the actor estimates the stochastic policy, which returns the probability distribution over discrete actions or for continuous action spaces the parameters of normal distribution. In both cases, our policy is stochastic, so, the action taken is sampled from this distribution.

Deterministic policy gradients belongs to the A2C family, but the policy is deterministic, which means that it directly provides us with the action to take from the state. This makes it possible to apply the chain rule to the Q-value, and by maximizing the Q, the policy will be improved as well.

The role of actor is to return the action to take for every given state. In a continuous action domain, every action is a number, so the actor network will take the state as an input and return N values, one for every action. This mapping will be deterministic, as the same network always returns the same output if the input is the same.

The role of the critic is to estimate the Q-value, which is a discounted reward of the action taken in some state. However, our action is a vector of numbers, so our critic net accepts two inputs: the state and the action. The output from the critic will be the single number, which corresponds to the Q-value.

We have two functions represented by deep neural networks, one is the actor, let's call it µ(s), which converts the state into the action and the other is the critic, by the state and the action giving us the Q-value: Q(s, a). We can substitute the actor function into the critic and get the expression with only one input parameter of our state: Q(s, µ(s)).

Now the output of the critic gives us the approximation of the entity we're interested in maximizing in the first place: the discounted total reward. This value depends not only on the input state, but also on parameters of the θµ actor and the θQ critic networks. At every step of our optimization, we want to change the actor's weights to improve the total reward that we want to get. In mathematical terms, we want the gradient of our policy.

In his deterministic policy gradient theorem, David Silver has proved that stochastic policy gradient is equivalent to the deterministic policy gradient. In other words, to improve the policy, we just need to calculate the gradient of the Q(s, µ(s)) function. By applying the chain rule, we get the gradient: ∇<sub>a</sub>Q(s, a)∇θ<sub>µ</sub>µ(s).


## Exploration
Since the policy is now deterministic, we have to explore the environment somehow. We can do this by adding noise to the actions returned by the actor before we pass them to the environment. There are several options here. The simplest method is just to add the random noise to the µ(s) + eta N actions. Another approach to the exploration will be to use the stochastic model, which is very popular in the financial world and other domains dealing with stochastic processes: OU processes. In our exploration, we'll add the value of the OU process to the action returned by the actor.

## Implementation

The model consists of two separate networks for the actor and critic and follows the architecture from the paper, Continuous Control with Deep Reinforcement Learning.

The actor is fully connected feed-forward network with two hidden layers, RELU activation and Batch Normalization. The input is an observation/state vector, while the output is a vector with N values, one for each action. The output actions are transformed with hyperbolic tangent non-linearity to squeeze the values to the -1..1 range.

```
  BatchNorm1d
    |
Fully Connected Layer (in=33 -> state size, out=128)
    |
  RELU
  BatchNorm1d
    |
Fully Connected Layer (in=128, out=128)
    |
  RELU
  BatchNorm1d
    |
Fully Connected Layer (in=128, out=4 -> action size)
    |
  tanh
```

The critic includes two separate paths for observation and the actions, and those paths are concatenated together to be transformed into the critic output of one number. The forward() function of the critic first transforms the observations with its first network - model_input, then concatenates the output and given actions to transform them using second network – model_output into one single value of Q.

```
Fully Connected Layer (in=33 -> state size, out=128)
		  |
		RELU
		BatchNorm1d
		  |
Fully Connected Layer (in=128+4, out=128)
		  |
		RELU
		BatchNorm1d
		  |
Fully Connected Layer (in=128, out=1)
```

The Agent.act method
Converts the observations into the appropriate form and ask the actor network to convert them into deterministic actions. Then it adds the exploration noise by applying the OU process. Lastly, it clips the actions to enforce them to fall into the -1..1 range.

The Agent.learn method
On every iteration, we store the experience into the replay buffer and sample the training batch.
Agent performs two separate training steps.

1.) To train the critic, we need to calculate the target Q-value using the one-step Bellman equation, with the target critic network as the approximation of the next state, then we calculate the MSE loss and ask the critic's optimizer to tweak the critic weights.
```
# ---------------------------- update critic ---------------------------- #
# Get predicted next-state actions and Q values from target models
actions_next = self.actor_target(next_states)
Q_targets_next = self.critic_target(next_states, actions_next)
# Compute Q targets for current states (y_i)
Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
# Compute critic loss
Q_expected = self.critic_local(states, actions)
critic_loss = F.mse_loss(Q_expected, Q_targets)
# Minimize the loss
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
```

2.) To train actor, we need to update the actor's weights in a direction that will increase the critic's output. As both the actor and critic are represented as differentiable functions, what we need to do is just pass the actor's output to the critic and then minimize the negated value returned by the critic.

The negated output of the critic could be used as a loss to backpropagate it to the critic network and, finally, the actor. We don't want to touch the critic's weights, so it's important to ask only the actor's optimizer to do the optimization step. The weights of the critic will still keep the gradients from this call, but they will be discarded on the next optimization step.

```
# ---------------------------- update actor ---------------------------- #
# Compute actor loss
actions_pred = self.actor_local(states)
actor_loss = -self.critic_local(states, actions_pred).mean()
# Minimize the loss
self.actor_optimizer.zero_grad()
actor_loss.backward()
self.actor_optimizer.step()
```
As the last step of the training loop, we perform the target networks update using so called ‘soft sync’. The soft sync is carried out on every step, but only a small ratio of the optimized network's weights are added to the target network. This makes a smooth and slow transition from old weight to the new ones.

```
# ----------------------- update target networks ----------------------- #
self.soft_update(self.critic_local, self.critic_target, TAU)
self.soft_update(self.actor_local, self.actor_target, TAU)                     
```

# PPO - Proximal Policy Optimization

Proximal Policy Optimization (PPO) is a policy gradient-based method and is one of the algorithms that have been proven to be stable as well as scalable.

In policy gradient methods, the algorithm performs rollouts to collect samples of transitions and (potentially) rewards, and updates the parameters of the policy using gradient descent to minimize the objective function. The idea is to keep updating the parameters to improve the policy until a good policy is obtained.

The PPO method uses a following objective function: the ratio between the new and the old policy scaled by the advantages, instead of the gradient of logarithm probability of the action taken as used in Advantage Actor-Critic method.

In math form the objective proposed by the PPO is J<sub>θ</sub> = E<sub>t</sub>[ grad<sub>θ</sub>log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>)A<sub>t</sub>].

However, if we just start to blindly maximize this value, it may lead to a very large update to the policy weights. To limit the update, the clipped objective is used. If we write the ratio between the new and the old policy as r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>)/π<sub>θ old</sub>(a<sub>t</sub>|s<sub>t</sub>) the clipped objective could be written as J <sup>clip</sup> <sub>θ</sub> = E<sub>t</sub>[min(r<sub>t</sub>(θ)A<sub>t</sub>, clip(r<sub>t</sub>(θ), 1 − \epsilon, 1 + \epsilon)A<sub>t</sub>)]. This objective limits the ratio between the old and the new policy to be in the interval [1 − \epsilon, 1 + \epsilon], so by varying # we can limit the size of the update.

Implementation

Both the actor and critic are placed in the separate networks without sharing weights.
Actor estimates the mean and the standard deviation for the actions, but it is just a single parameter of the model. This parameter will be adjusted during the training by SGD, but it doesn't depend on the observation.

The actor network is a fully connected feed forward network with two hidden layers - 128 units, each with tanh nonlinearity.  The input is an observation/state vector, while the output is a
vector with N values – means, one for each action.

```
Fully Connected Layer (in=33 -> state size, out=128)
		  |
		tanh
		  |
Fully Connected Layer (in=128, out=128)
		  |
		tanh
		  |
Fully Connected Layer (in=128, out=4 -> action size)
		  |
		tanh
```

The critic network also has two hidden layers of the same size with one single output value, which is the estimation of V(s), which is a discounted value of the state.

```
Fully Connected Layer (in=33 -> state size, out=128)
		  |
		RELU
		  |
Fully Connected Layer (in=128, out=128)
		  |
		RELU
		  |
Fully Connected Layer (in=128, out=1)
```

ActorCritic class comprises both actor and critic networks and calculates action mean and action log probabilities in the forward method.

Agent.step method
1.) generates rollouts/trajectories, calculates rewards, action log probabilities and reference values for the critic for each and every step of trajectory
2.) calculates and normalizes advantages to improve training stability
3.) performs multiple epochs (10) of actor and critic training with batch size

To train the critic, we calculate the Mean Squared Error (MSE) loss with the reference values calculated beforehand.

```
value_loss = 0.5 * F.mse_loss(sampled_returns, values)
```

In the actor training, we minimize the negated clipped objective: E<sub>t</sub>[min(r<sub>t</sub>(θ)A<sub>t</sub>, clip(r<sub>t</sub>(θ), 1 − \epsilon, 1 + \epsilon)A<sub>t</sub>)], where r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>)/π<sub>θ old</sub>(a<sub>t</sub>|s<sub>t</sub>).

```
ratio = (new_log_probs - sampled_log_probs_old).exp()
obj = ratio * sampled_advantages
obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages

policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()
```


### System Setup

 - Python:			3.6.6
 - CUDA:			  9.0
 - Pytorch: 		0.4.1
 - NVIDIA Driver:  	390.77
 - Conda:			4.5.10

 All training was performed on a single Ubuntu 18.04 desktop with an NVIDIA GTX 1080ti.

## Conclusion

Comparing the performance scores shows that Dueling DQN Agent and DQN Agent with Prioritized Experience Replay achieved the best performance.

Navigation with Pixels - so far i have tested basic DQN Agent with Prioritized Experience Replay using visual observations (frames).
Unfortunately DQN Agent performance didnt show significant progress during the learning.
I will try to implement few changes/improvements:
    frame stacking (buffering)
    reward clipping

Categorical DQN - while this DQN agent good shows learning progress during training, i was expecting that the overall performance would exceed Dueling DQN agent
and DQN Agent with Prioritized Experience Replay. I will try to investigate and test further.
