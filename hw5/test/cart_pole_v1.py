import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden) -> None:
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
    
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon,
                 gamma, target_replace_iter, memory_capacity):
        self.eval_net = Net(n_states, n_actions, n_hidden)
        self.target_net = Net(n_states, n_actions, n_hidden)

        # 每個 memory 中的 experience 大小為 (state + next state + reward + action)
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
    
    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action_value = self.eval_net(x)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))

        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        q_eval = self.eval_net(b_state).gather(1, b_action) # 重新計算這些 experience 當下 eval net 所得出的 Q value
        q_next = self.target_net(b_next_state).detach() # detach 才不會訓練到 target net
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())



if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    n_hidden = 50
    batch_size = 32
    lr = .01
    epsilon = .1
    gamma = .9
    target_replace_iter = 100
    memory_capacity = 2000
    n_episodes = 4000

    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma,
              target_replace_iter, memory_capacity)

    # 跑 200 個 episode，每個 episode 都是一次任務嘗試
    for i_episode in range(n_episodes):
        t = 0
        rewards = 0
        state, _ = env.reset() # 讓 environment 重回初始狀態 
        rewards = 0 # 累計各 episode 的 reward 
        for t in range(250): # 設個時限，每個 episode 最多跑 250 個 action
            env.render() # 呈現 environment

            # Key section
            action = choose_action(state)
            # action = env.action_space.sample() # 在 environment 提供的 action 中隨機挑選
            state, reward, terminated, truncated, info = env.step(action) # 進行 action，environment 返回該 action 的 reward 及前進下個 state
            rewards += reward # 累計 reward

            if terminated: # 任務結束返回 done = True
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

    env.close()