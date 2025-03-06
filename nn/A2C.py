import os
import tqdm
import wandb
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .policy import ContinuousPolicyNetwork
from .network import Critic

'''
A2C.py: 
- Actor-Critic 모델 구현
- Q-network는 q-network.py에 구현되어 있으며, Policy는 policy.py에 구현
- experience(), train(), test() 등이 class 함수로 구현
- V, Q, TD, TD(lambda), Natural 등 다양한 방식의 파라미터 업데이트 구현   
'''

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class A2C:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device

        # Dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Actor & Critic
        self.policy = ContinuousPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['lr'])
        # Scheduler
        self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='max', factor=0.5, patience=50, verbose=True)
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='max', factor=0.5, patience=50, verbose=True)

        # Hyperparameters
        self.gamma = config['gamma']
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.total_episodes = config['total_episodes']

        # Best model tracking
        self.best_score = -1e9
        self.best_episode = 0
        self.best_param = None

        # 경로 설정
        self.base_dir = f"results/{config['env_name']}/{config['model']}"
        self.model_name = f"{config['model']}_{config['env_name']}_{config['lr']}_{config['gamma']}"

    def get_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        mean, std = self.policy(state_t)

        std = torch.clamp(std, min=1e-6)
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action.cpu().numpy(), log_prob, entropy

    def train(self):
        rewards_history = []

        for episode in tqdm.tqdm(range(self.total_episodes)):
            obs, _ = self.env.reset()
            state = np.array(obs, dtype=np.float32)
            done = False

            # 에피소드 내 기록
            states = []
            actions = []
            rewards = []
            log_probs = []
            entropies = []
            next_states = []
            dones = []

            # ---------- 1) 에피소드 수집 ----------
            episode_reward = 0.0
            while not done:
                action, log_prob, entropy = self.get_action(state)
                next_obs, reward, done, truncated, _ = self.env.step(action)

                next_state = np.array(next_obs, dtype=np.float32)
                episode_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                entropies.append(entropy)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                if done or truncated:
                    break

            # ---------- 2) 리워드 정규화 (원한다면) ----------
            rewards = np.array(rewards, dtype=np.float32)
            if self.config.get("reward_normalize", False):
                r_mean = rewards.mean()
                r_std = rewards.std() if rewards.std() > 1e-6 else 1.0
                rewards = (rewards - r_mean) / r_std

            # ---------- 3) Advantage(또는 TD-Return) 계산 ----------
            # 간단히 1-step 혹은 MonteCarlo로 예시
            # (원하면 GAE, TD(lambda) 등 더 복잡한 방법 구현 가능)
            with torch.no_grad():
                values = []
                for s in states:
                    s_t = torch.tensor(s, dtype=torch.float32).to(self.device)
                    values.append(self.critic(s_t).item())

                # 한 에피소드라 done=1, MonteCarlo Return
                # G_t = r_t + gamma*r_{t+1} + ... 
                # 여기서는 간단히 backwards로 누적
                returns = []
                G = 0
                for r, d in zip(reversed(rewards), reversed(dones)):
                    G = r + self.gamma * G
                    returns.insert(0, G)

            # ---------- 4) Critic Update (단일 backward) ----------
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # 가치함수 예측
            pred_values = self.critic(states_t).squeeze(-1)
            critic_loss = F.mse_loss(pred_values, returns_t)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Gradient Clipping for actor
            if self.config["gradient_clipping_"] == True:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.get("max_grad_norm", 0.5))
            self.critic_optimizer.step()

            # ---------- 5) Policy Update (단일 backward) ----------
            # Advantage = G_t - V(s_t)
            advantages = returns_t - pred_values.detach()
            advantages_t = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs_t = torch.stack(log_probs).to(self.device)
            entropies_t = torch.stack(entropies).to(self.device)
            

            # policy loss = - log_prob * advantage - entropy_bonus
            # policy loss > mean() 으로 조정
            policy_loss = -(log_probs_t * advantages_t).mean()
            policy_loss -= self.entropy_coeff * entropies_t.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # Gradient Clipping for actor
            if self.config["gradient_clipping_"] == True:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.get("max_grad_norm", 0.5))
            self.policy_optimizer.step()

            # ---------- 6) Logging & 모델 저장 ----------
            rewards_history.append(episode_reward)
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.best_episode = episode
                self.best_param = self.policy.state_dict()

            # Reward 기반 학습률 스케줄러 업데이트
            if self.config["scheduler_"] == True:
                self.policy_scheduler.step(episode_reward)
                
            if episode % 5000 == 0:
                self.save_model(self.policy.state_dict(), self.model_name, f"_ep{episode}_{episode_reward}.pth")

            # Wandb Record
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "Critic_loss": critic_loss.item(),
                "Policy_loss": policy_loss.item(),
            })

        # 끝난 뒤 Best & 최종 모델 저장
        self.save_model(self.best_param, self.model_name, f"_best_{self.best_score}.pth")
        self.save_model(self.policy.state_dict(), self.model_name, "_last.pth")

        return rewards_history

    def save_model(self, param, file_name, suffix=".pth"):
        pt_dir = os.path.join(self.base_dir, file_name)
        if not os.path.exists(pt_dir):
            os.makedirs(pt_dir)
        file_path = os.path.join(pt_dir, file_name + suffix)
        torch.save(param, file_path)

    def load_model(self, file_name):
        file_path = os.path.join(self.base_dir, file_name)
        self.policy.load_state_dict(torch.load(file_path))
        self.policy.eval()



