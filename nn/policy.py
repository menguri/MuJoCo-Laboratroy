import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousPolicyNetwork, self).__init__()
        
        # 첫 번째 FC layer: 입력 상태에서 256 차원 특징 추출
        self.fc1 = nn.Linear(state_dim, 256)
        # Layer Normalization을 추가하여 내부 covariate shift를 줄임 → 학습 안정성 향상
        self.ln1 = nn.LayerNorm(256)
        
        # 두 번째 FC layer: 256 차원에서 다시 256 차원으로 변환
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
        # 행동 평균(mean)과 로그 표준편차(log_std)를 위한 출력 layer들
        self.fc3_mean = nn.Linear(256, action_dim)
        self.fc3_log_std = nn.Linear(256, action_dim)
        
        # --- Orthogonal Initialization ---
        # 각 layer에 대해 orthogonal initialization을 적용하면 학습 초기에 gradient flow가 더 안정됨
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0)
        
        # 출력 layer는 작은 gain 값을 사용하여 초기 출력을 안정화함 (tanh 활성화와의 조합)
        nn.init.orthogonal_(self.fc3_mean.weight, gain=0.01)
        nn.init.constant_(self.fc3_mean.bias, 0)
        nn.init.orthogonal_(self.fc3_log_std.weight, gain=0.01)
        nn.init.constant_(self.fc3_log_std.bias, 0)
    
    def forward(self, state):
        # fc1 → layer normalization → ReLU
        x = F.relu(self.ln1(self.fc1(state)))
        # fc2 → layer normalization → ReLU
        x = F.relu(self.ln2(self.fc2(x)))
        
        # 행동 평균: tanh를 사용해 출력 범위를 [-1,1]로 제한 (action clipping)
        mean = torch.tanh(self.fc3_mean(x))
        
        # 로그 표준편차: 범위 제한 후 exp하여 std를 구함
        log_std = self.fc3_log_std(x)
        log_std = torch.clamp(log_std, min=-2, max=1)  # 표준편차가 너무 작거나 큰 값을 방지
        std = torch.exp(log_std)
        
        return mean, std
