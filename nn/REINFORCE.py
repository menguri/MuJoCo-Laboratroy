import os
import tqdm
import random
import wandb
import numpy as np
import torch
from collections import deque
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .policy import ContinuousPolicyNetwork

'''
REINFORCE.py: 
- REINFORCE 모델 구현
- Q-network는 q-network.py에 구현되어 있으며, Policy는 policy.py에 구현
- experience(), train(), test() 등이 class 함수로 구현현   
'''

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class REINFORCE:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device
        self.method = config['method']
        self.batch_size = config['batch_size']

        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.gamma = config['gamma']  # 할인율 (discount factor)
        self.total_episodes = config['total_episodes']
        self.replay_buffer = deque(maxlen=config["buffer_size"])

        # policy
        self.policy = ContinuousPolicyNetwork(self.state_dim, self.action_dim).to(self.device) 
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        # Scheduler
        self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, verbose=True)
        # entropy coefficient for entropy regularization (탐험 촉진)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)

        # best param
        self.best_episode = 0.0
        self.best_score = -10000
        self.best_param = dict()

        # model name
        self.base_dir = f"results/{config['env_name']}/{config['model']}"
        self.model_name = f"{config['model']}_{config['env_name']}_{config['method']}_{config['lr']}_{config['gamma']}"

    # Agent gets action
    def get_action(self, state):
        # 상태로부터 행동 샘플링
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mean, std = self.policy(state)

        # 표준편차가 0 이하로 떨어지지 않도록 방지
        std = torch.clamp(std, min=1e-6)

        # NaN 체크
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("[ERROR] NaN detected in get_action!")
            print("mean:", mean)
            print("std:", std)
            exit()
            
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # 엔트로피 계산 (분포의 불확실성 측정)
        entropy = dist.entropy().sum(dim=-1)
        
        return action.cpu().numpy(), log_prob, entropy

    def train(self):
        rewards_history = []

        for episode in tqdm.tqdm(range(self.total_episodes)):
            obs, _ = self.env.reset()
            state = np.array(obs, dtype=np.float32)
            episode_reward = 0.0
            done = False

            entropies = []   # 각 행동의 엔트로피 기록
            log_probs = []  # 로그 확률 기록
            rewards = []  # 보상 기록


            while not done:
                action, log_prob, entropy = self.get_action(state)  # 행동과 로그 확률
                next_obs, reward, done, truncated, _ = self.env.step(action)  # action에 따른 다음 상태 리턴턴
                next_state = np.array(next_obs, dtype=np.float32)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                log_probs.append(log_prob)  # 행동에 대한 로그 확률 기울기 저장
                entropies.append(entropy)   # 엔트로피 기록 저장
                rewards.append(reward)  # 보상 저장
                if done or truncated:
                    break

            # ----------------------------------------------------
            # ★ 보상 정규화 (Reward Normalization) 적용 ★
            # 현재 에피소드에서 얻은 보상의 분포를 정규화
            rewards = np.array(rewards, dtype=np.float32)
            reward_mean = rewards.mean()         # 에피소드 보상의 평균
            reward_std = rewards.std() if rewards.std() > 0 else 1.0  # 표준편차 (0이면 1로 대체)
            normalized_rewards = (rewards - reward_mean) / reward_std  # 정규화된 보상

            # 누적 할인 보상 계산 (G_t) - 각 배치 내에서 계산
            discounted_rewards = self.compute_discounted_rewards(normalized_rewards)
        
            # 배치 단위 정책 업데이트 수행
            loss = self.update_policy(log_probs, discounted_rewards, entropies)
            total_loss = loss.item()
            
            # 에피소드별 보상 기록 및 best 모델 저장
            rewards_history.append(episode_reward)
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.best_episode = episode
                self.best_param = self.policy.state_dict()
                        
            # Reward 기반 학습률 스케줄러 업데이트
            if self.config["scheduler_"] == True:
                self.policy_scheduler.step(episode_reward)
            
            if episode % 5000 == 0:
                self.save_model(self.policy.state_dict(), self.model_name, f"_episode{episode}_score{episode_reward}.pth")
            
            # wandb 로그 기록
            wandb.log({"episode": episode, "reward": episode_reward})
            wandb.log({"episode": episode, "loss": total_loss})
        
        # Best 모델과 마지막 모델 각각 저장
        self.save_model(self.best_param, self.model_name, f"_episode{self.best_episode}_best{self.best_score}.pth")
        self.save_model(self.policy.state_dict(), self.model_name, ".pth")
        
        return rewards_history

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards

    def update_policy(self, log_probs, discounted_rewards, entropies):
        # 기대 보상 (G) 계산
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs)  # 로그 확률을 텐서로 변환
        entropy_tensor = torch.stack(entropies)  # 엔트로피 값들의 텐서로 변환
        
        # 손실 함수 계산
        loss = -torch.sum(log_probs * discounted_rewards)
        # 엔트로피 보너스 항 추가: 엔트로피가 높을수록 loss를 낮춰 탐험을 촉진
        loss = loss - self.entropy_coeff * torch.sum(entropy_tensor)
        
        # 정책 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping for actor
        if self.config["gradient_clipping_"] == True:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.get("max_grad_norm", 0.5))
        self.optimizer.step()

        return loss
    
    def save_model(self, param, file_name, pt=".pth"):
        # `project/results` 디렉토리가 없으면 생성
        pt_dir = self.base_dir + "/" + file_name
        if not os.path.exists(pt_dir):
            os.makedirs(pt_dir)
        # 모델의 state_dict를 저장합니다.
        file_path = pt_dir + "/" + file_name + pt
        torch.save(param, file_path)

    def load_model(self, file_name):
        file_path = os.path.join(self.base_dir, file_name)
        # 모델을 로드하고 state_dict를 업데이트합니다.
        self.policy.load_state_dict(torch.load(file_path))
        self.policy.eval()  # 평가 모드로 전환



