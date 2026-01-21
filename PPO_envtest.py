import cv2
import numpy as np
from PPO_env import MyCustomEnv

# 위에서 정의한 클래스 인스턴스화
env = MyCustomEnv()

# 1. 초기화 (t=0)
obs, info = env.reset()

# 렌더링을 위한 윈도우 생성
cv2.namedWindow("Env State", cv2.WINDOW_NORMAL)

# 시뮬레이션 루프 (Trajectory Rollout)
for t in range(1000):
    # 2. 행동 선택 (여기서는 랜덤 정책 \pi_{rand} 사용)
    action = env.action_space.sample()
    # 3. 환경 진행 (Step t -> t+1)
    next_obs, reward, terminated , truncated, info = env.step(action)

    # 4. 시각화 
    # 사람 눈에 잘 보이게 10배 확대 
    render_img = cv2.resize(next_obs, (280, 280), interpolation=cv2.INTER_NEAREST)
    
    # 컬러 텍스트로 상태 표시 
    render_img = cv2.cvtColor(render_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(render_img, f"R: {reward}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Env State", render_img)

    # 5. 종료 조건 처리
    if  terminated or truncated:
        print(f"Episode Finished at step {t}. Resetting...")
        obs, info = env.reset()
        # 시각적 확인을 위해 잠시 대기
        if cv2.waitKey(1000) == ord('q'): break
    else:
        obs = next_obs
        # 0.1초 대기 (프레임 속도 조절)
        if cv2.waitKey(100) == ord('q'): break

env.close()
cv2.destroyAllWindows()