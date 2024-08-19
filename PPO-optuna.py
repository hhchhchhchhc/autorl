import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import wandb
import optuna

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, x):
        x = self.net(x)
        prob = F.softmax(x, dim=-1)
        return prob

# 定义Critic网络  
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        value = self.net(x)
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
                 device,
                 batch_size):
                 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.batch_size = batch_size
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
        action_logprob = dist.log_prob(action) 
        return action.item(), action_logprob.item()

    def update(self):     
        # 将memory内数据转为tensor
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = torch.LongTensor([t[1] for t in self.memory]).view(-1, 1).to(self.device)
        logprobs_old = torch.FloatTensor([t[2] for t in self.memory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([t[3] for t in self.memory]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in self.memory]).to(self.device)
        
        # 计算状态价值
        V = self.critic(states).detach().squeeze()
        
        # 计算GAE和目标值
        advantages, returns = self.compute_gae(rewards, V, dones)
        
        # mini batch
        for _ in range(self.K_epochs):
            for indices in BatchSampler(SubsetRandomSampler(range(len(self.memory))), self.batch_size, drop_last=False):
                mb_states = states[indices]
                mb_actions = actions[indices]
                mb_logprobs_old = logprobs_old[indices]
                mb_returns = returns[indices]
                mb_advantages = advantages[indices]
                
                # 评估动作概率和状态价值
                action_probs = self.actor(mb_states)
                dist = Categorical(action_probs)
                action_logprobs = dist.log_prob(mb_actions.squeeze()).unsqueeze(1)
                V_new = self.critic(mb_states)
                
                # 计算critic损失
                critic_loss = self.MseLoss(V_new, mb_returns)

                # 计算actor损失(PPO clip)
                ratios = torch.exp(action_logprobs - mb_logprobs_old)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
        self.memory.clear()

    def compute_gae(self, rewards, values, dones, tau=0.95):
        gae = 0
        returns = []
        advantages = []

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0  # 对于最后一步，假设下一个值为0
            else:
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * tau * (1 - dones[step]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns
        
# 超参数调优的目标函数     
def objective(trial):
    # 从Optuna中获取超参数
    params = {
        'actor_lr': trial.suggest_float('actor_lr', 1e-5, 1e-2, log=True),
        'critic_lr': trial.suggest_float('critic_lr', 1e-5, 1e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'K_epochs': trial.suggest_categorical('K_epochs', [5, 10, 15, 20]),
        'eps_clip': trial.suggest_categorical('eps_clip', [0.1, 0.2, 0.3])   
    }

    # 初始化环境和智能体
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, params['actor_lr'], params['critic_lr'], 
                params['gamma'], params['K_epochs'], params['eps_clip'], 
                params['hidden_dim'], device, batch_size=batch_size)

    rewards = []
    ma_rewards = [] # moving average rewards

    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            agent.memory.append((state, action, log_prob, reward, done))
            state = next_state

        agent.update()
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)  

        # 上报当前trial的评估分数给optuna
        trial.report(np.mean(ma_rewards[-10:]), i_episode)

        # 在不满足条件时提前终止实验
        if trial.should_prune():
            raise optuna.TrialPruned()  
    
    return np.mean(rewards[-10:])

# 初始化wandb
wandb.init(project="ppo_cartpole", settings=wandb.Settings(start_method="fork"))

# 设置训练参数      
max_episodes = 400
n_trials = 25
batch_size = 32

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

# 使用最佳超参数训练模型并可视化结果
best_params = study.best_params
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, action_dim, best_params['actor_lr'], best_params['critic_lr'], 
            best_params['gamma'], best_params['K_epochs'], best_params['eps_clip'], 
            best_params['hidden_dim'], device, batch_size=batch_size)

rewards = []
ma_rewards = [] # moving average rewards

for i_episode in range(1, max_episodes+1):
    state, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        agent.memory.append((state, action, log_prob, reward, done))
        state = next_state
        
    agent.update()

    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
    else:
        ma_rewards.append(ep_reward)

    wandb.log({"Episode Reward": ep_reward, "Moving Average Reward": ma_rewards[-1]})
    
# 保存训练好的模型
torch.save(agent.actor.state_dict(), './ppo_actor.pth')
torch.save(agent.critic.state_dict(), './ppo_critic.pth')
