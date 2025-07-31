import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt


from torch.optim import Adam

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, action_bound):
        super(Actor, self).__init__()

        #Define Layer
        self.h1 = nn.Linear(input_dim, 64)
        self.h2 = nn.Linear(64,32)
        self.h3 = nn.Linear(32, 16)
        self.mu = nn.Linear(16, action_dim)
        self.std = nn.Linear(16, action_dim)
        
        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        mu = torch.tanh(self.mu(x))
        std = F.softplus(self.std(x))

        # output of tanh -> -1 ~ 1
        # scailing : -action_bound ~ action_bound
        mu = self.action_bound * mu

        return mu, std


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()

        # output dim = 1 (value function)
        output_dim = 1
        
        self.h1 = nn.Linear(input_dim, 64)
        self.h2 = nn.Linear(64, 32)
        self.h3 = nn.Linear(32, 16)
        self.v = nn.Linear(16, output_dim)

    def forward(self, state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        v = self.v(x)

        return v

        

class Agent:
    def __init__(self, env):
        self.env = env
        print(env.spec.max_episode_steps)  # 이게 400이면 커스텀 설정이 들어간 것

        # Torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ' + self.device.type)
        torch.set_default_dtype(torch.float32)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}


        # hyper parameter
        self.gamma = 0.95 # discount factor
        self.batch_size = 32
        self.actor_lr = 0.0001
        self.critic_lr = 0.001

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [0.1, 1.5]

        # Load NN and Optimizer
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(**self.tensor_args)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic = Critic(self.state_dim).to(**self.tensor_args)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.critic_lr)


        self.save_epi_reward = []

    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * torch.log(var * 2 * np.pi)
        # return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
        return torch.sum(log_policy_pdf, dim=1)
    
    def get_action(self, state):
        mu_a, std_a = self.actor(state) # dim : N, A

        std_a = torch.clamp(std_a, self.std_bound[0], self.std_bound[1])
        action = torch.normal(mu_a, std_a)
        return action
    
    def actor_learn(self, states, actions, advantages):
        # states, actions, advantages: torch.Tensor (device 포함)
        self.actor_optim.zero_grad()

        # forward pass
        mu_a, std_a = self.actor(states)
        log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

        # compute loss
        loss_policy = log_policy_pdf * advantages
        loss = torch.sum(-loss_policy)

        # backward
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, state, next_state, reward, dones):
        # dones : N,1
        # reward : N,1
        self.critic_optim.zero_grad()

        v_t = self.critic(state) # N, 1
        v_t_next = self.critic(next_state) # N, 1
        # mask = (~dones.bool()).float()
        # print(mask)
        target = reward + (1 - dones.float()) * v_t_next * self.gamma
        loss = F.mse_loss(v_t, target.detach())
        loss.backward()
        self.critic_optim.step()



    def train(self, max_episoide_num):
        for ep in range(int(max_episoide_num)):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
            time, episode_reward, done = 0, 0, False
            
            # init state
            state, _ = self.env.reset()
            while not done:
                # state_tensor = torch.from_numpy(state.reshape(1, self.state_dim)).float().to(**self.tensor_args)
                # action = self.get_action(state_tensor)
                action = self.get_action(torch.tensor(state.reshape(1, self.state_dim)).to(**self.tensor_args)).squeeze(0) # 1, action_dim
                action_cpu = action.detach().cpu().numpy().reshape(1,self.action_dim)
                action_cpu = np.clip(action_cpu, -self.action_bound, self.action_bound)

                next_state, reward, terminated, truncated, _ = self.env.step(action_cpu)
                done = terminated or truncated

                state = state.reshape(1,self.state_dim)
                next_state = next_state.reshape(1,self.state_dim)
                done = np.array(done).reshape(1, 1)
                reward = np.array(reward).reshape(1, 1)

                # train_reward = (reward + 8)/8
                train_reward = reward 

                batch_state.append(state)
                batch_action.append(action_cpu)
                batch_reward.append(train_reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                if len(batch_state)<self.batch_size:
                    state = next_state.flatten()
                    episode_reward += reward.item()
                    time +=1
                    continue

                states = np.vstack(batch_state)
                actions = np.vstack(batch_action)
                train_rewards = np.vstack(batch_reward)
                next_states = np.vstack(batch_next_state)
                dones = np.vstack(batch_done)

                # print(f"states size : {states.shape}")
                # print(f"actions size : {actions.shape}")
                # print(f"train_rewards size : {train_rewards.shape}")
                # print(f"dones size : {dones.shape}")
                


                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []


                # Critic Update
                self.critic_learn(torch.tensor(states).to(**self.tensor_args), torch.tensor(next_states).to(**self.tensor_args), torch.tensor(train_rewards).to(**self.tensor_args), torch.tensor(dones).to(**self.tensor_args))
                v_values = self.critic(torch.tensor(states).to(**self.tensor_args))
                v_next_values = self.critic(torch.tensor(next_states).to(**self.tensor_args))
                advantages = torch.tensor(train_rewards).to(**self.tensor_args) + self.gamma * v_next_values - v_values

                self.actor_learn(torch.tensor(states).to(**self.tensor_args), torch.tensor(actions).to(**self.tensor_args), advantages)

                state = next_state.flatten()
                episode_reward += reward.item()
                time +=1

            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            if ep % 10 == 0:
                torch.save(self.actor.state_dict(), "./weights/pendulum_actor.pth")
                torch.save(self.critic.state_dict(), "./weights/pendulum_critic.pth")

        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)