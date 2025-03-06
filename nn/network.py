import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        # Hidden layer 개수를 증가 (128 → 256)
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)

        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)
        
        self.fc4 = nn.Linear(256, 1)  # 최종 V(s) 출력
        
        # --- Residual Connection 추가 ---
        self.residual_layer = nn.Linear(state_dim, 256)  # 입력을 그대로 더할 수 있도록 맞춤

        # --- Dropout 추가 ---
        self.dropout = nn.Dropout(p=0.2)  # 과적합 방지를 위해 20% 뉴런 비활성화
        
        # --- Orthogonal Initialization ---
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.orthogonal_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.orthogonal_(self.fc4.weight, gain=1.0)
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, state):
        # 첫 번째 층
        x = F.relu(self.ln1(self.fc1(state)))
        
        # 두 번째 층 (Dropout 추가)
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        
        # 세 번째 층 (Dropout 추가)
        x = self.dropout(F.relu(self.ln3(self.fc3(x))))
        
        # --- Residual Connection 적용 ---
        residual = self.residual_layer(state)  # 원본 입력을 변환
        x = x + residual  # 원래의 입력 정보를 추가 (잔차 연결)

        # 최종 V(s) 출력
        v_value = self.fc4(x)
        return v_value