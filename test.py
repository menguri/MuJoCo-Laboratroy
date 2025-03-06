import argparse
import yaml
import wandb
import gymnasium as gym
import torch
import numpy as np
import matplotlib
# 비대화형 백엔드 설정 (헤드리스 환경에서 plt.show() 없이 사용 가능)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# utils 가져오기
from simulation.sim_utils import render_simulation
from configs.config_utils import load_config

# Model Load
from nn.A2C import A2C
from nn.REINFORCE import REINFORCE


# 오프스크린 렌더링용 환경 변수 설정
os.environ["MUJOCO_GL"] = "osmesa"  # 혹은 "egl"
from nn.policy import PolicyNetwork

'''
test.py: Agent를 불러와서 시뮬레이션을 돌리고, 저장하는 공간
- random simulation 함수는 simulation.py 에서 구현
- 각 agent는 get_action() 함수를 따로 가짐
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Config file path")
    parser.add_argument('--name', type=str, help='실험의 이름')
    args = parser.parse_args()

    # 설정 파일 불러오기
    config = load_config(args.config)
    
    # wandb 초기화
    wandb.init(project=f"{config["env_name"]}_experiment", config=config, name=args.name if args.name else "test_halfcheetah")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Gymnasium 환경 생성 (render_mode="rgb_array" 중요!)
    env = gym.make(config["env_name"], render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # 사전 학습된 정책 네트워크 불러오기
    if config['model'] == 'A2C':
        agent = A2C(env, config, DEVICE)
    elif config['model'] == 'REINFORCE':
        agent = REINFORCE(env, config, DEVICE)
    else:
        print(f"There is no model like {config['model']}")

    agent.load_model(f"{config['model']}_{config['method']}_{config['lr']}_{config['gamma']}.pth")
    
    # 시뮬레이션 실행 & wandb에 업로드
    render_simulation(env, agent, config, device, action_dim, num_episodes=1, save_path="simulation.gif", wandb_stack=True)


if __name__ == "__main__":
    main()

