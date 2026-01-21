import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class MyCustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 1. 공간 정의 (Manifold Definition)
        # 관측 공간: 28x28 이미지 (0:빈공간, 1:벽, 255:에이전트)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.center_pos = np.array([14, 14], dtype=np.float32)
        # 내부 상태 변수 선언
        self.agent_pos = None
        self.grid_state = None 
        # action: 0(상), 1(하), 2(좌), 3(우)
        self.move_lut = np.array([
            [-1, 0], # Action 0
            [1, 0],  # Action 1
            [0, -1], # Action 2
            [0, 1]   # Action 3
        ], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # agent 위치 초기화 (중앙에서 시작)
        self.agent_pos = np.array([14, 14])  # 벡터 연산을 위해 numpy array 사용 권장
        # 맵 텐서 초기화 (벽이 없는 상태)
        self.grid_state = np.zeros((28,28), dtype=np.uint8)
        
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        reward = 0.0
        # 벽 생성
        wall_mask = self.grid_state > 0
        self.grid_state[self.grid_state >= 235] = 0
        self.grid_state[self.grid_state > 0] += 20

        if np.random.rand() < 0.3:# 생성 확률 30%
            axis = np.random.choice([0, 1])
            if axis == 0: # 가로 벽 (Row update)
                row_idx = np.random.randint(0, 28)
                self.grid_state[row_idx, :] = 127
            else: # 세로 벽 (Col update)
                col_idx = np.random.randint(0, 28)
                self.grid_state[:, col_idx] = 127 

        diff = self.agent_pos - self.center_pos
        dist_sq = np.dot(diff, diff)
        distance_penalty = dist_sq * 0.005
        if isinstance(action, np.ndarray):
            action = action.item()

        # agent 이동
        self.agent_pos += self.move_lut[action]
        self.agent_pos = np.clip(self.agent_pos, 0, 27)
        current_pixel_val = self.grid_state[self.agent_pos[0], self.agent_pos[1]]

        # 5. 보상 및 종료 조건 (Reward & Termination)
        truncated = False
        terminated = False
        
        # 충돌 체크 (Collision Detection)
        # 현재 위치의 grid_state가 (벽)이면 충돌
        if current_pixel_val >= 207:
            reward = -100.0
            terminated = True
        elif current_pixel_val > 0:
            # (선택사항) 옅은 벽(127~167)을 밟았을 때 페널티
            reward -= -3.0 
            terminated = False
        else:
            # 빈 공간
            if distance_penalty == 0:
                reward += 10.0
            else:
                reward += 5.0 - distance_penalty
            
        observation = self._get_obs()
        
        # [수정 4] 5-Tuple 반환 (Gymnasium 표준)
        return observation, reward, terminated , truncated, {}

    def _get_obs(self):
        
        # 1. 현재 벽 상태 복사 (Base Layer)
        img = self.grid_state.copy()
        
        # 2. 에이전트 위치 오버레이 (Overlay Agent)
        y, x = self.agent_pos
        img[y, x] = 255 
        resized_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_NEAREST)
        
        # 3. 채널 차원 추가 (H, W) -> (H, W, C) for CNN Input
        return resized_img.astype(np.uint8)[:, :, np.newaxis]