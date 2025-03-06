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

# 오프스크린 렌더링용 환경 변수 설정
os.environ["MUJOCO_GL"] = "osmesa"  # 혹은 "egl"

def render_simulation(env, 
                      agent,
                      config, 
                      device, 
                      action_dim, 
                      num_episodes=1, 
                      save_path=None,
                      wandb_stack=True):
    """
    Gymnasium 환경을 오프스크린 렌더링(rgba_array)으로 실행하여 프레임을 수집하고,
    wandb에 GIF 형태로 업로드. 필요할 경우 로컬에 GIF 저장도 가능.
    """
    frames = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.float32)
        done = False

        while not done:
            # 한 스텝 실행 전 프레임 캡처
            frame = env.render()  # render_mode="rgb_array"로 설정되어 있어야 함
            if frame is not None:
                # wandb.Video로 업로드하려면 (T, H, W, C) 형태여야 하므로 리스트에 저장
                frames.append(np.array(frame, dtype=np.uint8))

            action, log_prob, entropy = agent.get_action(state)  # 행동과 로그 확률
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated

    env.close()

    # 렌더링된 프레임이 없다면 종료
    if len(frames) == 0:
        print("No frames captured. Check env render_mode.")
        return

    # 1) wandb에 업로드하기
    frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
    if wandb_stack == True:
        wandb.log({f"{config['model']}_{config['method']}_{config['lr']}_{config['gamma']}_video": wandb.Video(frames_np, fps=30, format="gif")})