import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from custom_env import ImageEnv
from rl_algorithm import DQN

env = ImageEnv(gym.make('ALE/MsPacman-v5'))
state, info = env.reset()

state_dim = state.shape # (4, 84, 84)
action_dim = env.action_space.n # 9

next_state, reward, terminated, truncated, _ = env.step(0)

# print(next_state.shape)
# print(reward)
# print(terminated) # bool
# print(truncated)  # bool

agent = DQN(state_dim, action_dim)

# print(agent.buffer.next_states.shape)
# print(agent.buffer.states.shape)
# print(agent.buffer.actions.shape)
# print(agent.buffer.rewards.shape)
# print(agent.buffer.terminated.shape)

action = agent.act(state, training=False)
print("act:", action, type(action))

next_state, reward, terminated, truncated, _ = env.step(action)
transition = (state, action, reward, next_state, terminated)
agent.buffer.update(*transition)

batch_size = 64
states, actions, rewards, next_states, terminateds = \
            map(lambda x: x.to('cuda'), agent.buffer.sample(batch_size))

predited_q = agent.network(states).gather(1, actions.type(torch.int64))
next_q = agent.target_network(next_states).detach().max(1)[0]

print(predited_q.shape)
print(next_q.shape)

gamma = .99
print(rewards.shape)
print(terminateds.shape)
td_target = rewards + gamma * next_q.unsqueeze(1) * (1 - terminateds)
print(td_target.shape)
print(td_target.squeeze(1).shape)

loss = F.smooth_l1_loss(predited_q, td_target.detach())
print(loss)