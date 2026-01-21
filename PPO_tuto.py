import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from PPO_env import MyCustomEnv
log_dir = "./log/"

# 1. 환경 생성
env = MyCustomEnv()

# 2. Gymnasium 표준 규격 검사 (Sanity Check)
check_env(env)
print("Environment check passed!")

# 3. 모델 정의 
# CnnPolicy: 입력이 이미지이므로 자동으로 CNN Feature Extractor를 사용합니다.
model = PPO("CnnPolicy", env, verbose=1,tensorboard_log=log_dir)
#   tensorboard --logdir=./ppo_logs/

# 4. 학습 시작 (Optimization Loop)
# 10,000 스텝 동안 에이전트가 환경과 상호작용하며 그라디언트를 업데이트합니다.
model.learn(total_timesteps=100000)

obs, _ = env.reset()

for _ in range(1000):
    # 1. 모델 예측 (Inference)
    action, _states = model.predict(obs,deterministic=True)
    
    # 2. 환경 진행 (Dynamics)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 3. 시각화 (Rendering Pipeline)
    # obs가 (84, 84, 1)이라고 가정 시 resize 잘 동작함
    render_img = cv2.resize(obs, (280, 280), interpolation=cv2.INTER_NEAREST)
    render_img = cv2.cvtColor(render_img, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(render_img, f"R: {reward:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Env State", render_img)
    
    # [수정 1] 화면 갱신을 위해 매 프레임 호출 (조건문 밖으로 뺌)
    # 0.1초(100ms) 대기. 'q' 누르면 긴급 종료
    if cv2.waitKey(100) == ord('q'): 
        break
    
    # 4. 종료 처리 (Reset Logic)
    # terminated(성공/실패) 뿐만 아니라 truncated(시간초과)도 체크하는 것이 정석
    if terminated or truncated:
        obs, _ = env.reset()
        print("Episode Reset")

env.close()
cv2.destroyAllWindows()
