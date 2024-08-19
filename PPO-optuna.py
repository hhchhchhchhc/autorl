import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import wandb
import optuna

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = F.softmax(x, dim=-1)
        return prob

# 定义Critic网络  
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# 定义PPO算法
class PPO:
    def __init__(self, 
                 state_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 gamma,
                 K_epochs,
                 eps_clip,
                 hidden_dim,
                 device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.memory = []
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def update(self, memory):     
        states = torch.FloatTensor([t[0] for t in memory]).to(self.device)
        actions = torch.LongTensor([t[1] for t in memory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in memory]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in memory]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in memory]).to(self.device)
        
        V_old = self.critic(states).detach()
        
        for _ in range(self.K_epochs):
            # 计算状态价值和优势函数  
            V_new = self.critic(states)
            advantages = rewards + (1 - dones) * self.gamma * self.critic(next_states).detach() - V_old

            # 计算动作概率和动作对应的对数概率  
            action_probs = self.actor(states)
            action_log_probs = torch.log(action_probs.gather(1, actions)).to(self.device)

            # 计算重要性采样系数            
            ratios = torch.exp(action_log_probs - action_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
            # 计算损失
            actor_loss = -torch.min(surr1, surr2).mean() 
            critic_loss = self.MseLoss(V_new, rewards + (1 - dones) * self.gamma * V_old)
            
            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
      
# 超参数调优的目标函数     
def objective(trial):
    # 从Optuna中获取超参数
    params = {
        'actor_lr': trial.suggest_float('actor_lr', 1e-5, 1e-2, log=True),
        'critic_lr': trial.suggest_float('critic_lr', 1e-5, 1e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512]),
        'steps_per_epoch': trial.suggest_categorical('steps_per_epoch', [500, 1000, 1500, 2000, 2500]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'K_epochs': trial.suggest_categorical('K_epochs', [5, 10, 15, 20]),
        'eps_clip': trial.suggest_categorical('eps_clip', [0.1, 0.2, 0.3])   
    }

    # 初始化环境和智能体
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, params['actor_lr'], params['critic_lr'], params['gamma'], 
                params['K_epochs'], params['eps_clip'], params['hidden_dim'], device)

    rewards = []
    ma_rewards = [] # moving average rewards

    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action) 
            ep_reward += reward
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(agent.memory) >= params['steps_per_epoch']:
                agent.update(agent.memory)
                agent.memory.clear()

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)  

        # 上报当前迭代评估指标给optuna
        trial.report(np.mean(ma_rewards[-10:]), i_episode)

        # 在不满足条件时提前终止实验
        if trial.should_prune():
            raise optuna.TrialPruned()  
    
    return np.mean(rewards[-10:])

# 初始化wandb，使用fork作为启动方法
wandb.init(project="ppo_cartpole", settings=wandb.Settings(start_method="fork"))

# 设置训练参数      
max_episodes = 500
n_trials = 25

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 开始超参数调优
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=n_trials)

# 打印最佳超参数
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 训练最佳模型并可视化结果
best_params = study.best_params
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, action_dim, best_params['actor_lr'], best_params['critic_lr'], best_params['gamma'],   
            best_params['K_epochs'], best_params['eps_clip'], best_params['hidden_dim'], device)

rewards = []
ma_rewards = [] # moving average rewards

for i_episode in range(1, max_episodes+1):
    state, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        ep_reward += reward
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) >= best_params['steps_per_epoch']:
            agent.update(agent.memory)
            agent.memory.clear()

    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
    else:
        ma_rewards.append(ep_reward)

    wandb.log({"Episode Reward": ep_reward, "Moving Average Reward": ma_rewards[-1]})
    
# 保存训练好的模型
torch.save(agent.actor.state_dict(), './ppo_actor.pth')
torch.save(agent.critic.state_dict(), './ppo_critic.pth')
