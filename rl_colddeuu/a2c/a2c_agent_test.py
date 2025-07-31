import torch
import gym
import numpy as np
from a2c_learn import Agent  # 이미 학습한 Agent 클래스 import

def test():
    # 환경 생성 (render_mode를 반드시 human으로 설정)
    env = gym.make("Pendulum-v1", render_mode="human")

    # 에이전트 생성
    agent = Agent(env)

    # 저장된 모델 불러오기
    actor_weights_path = "./weights/pendulum_actor.pth"
    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.actor.eval()  # 평가 모드 전환

    # 초기 상태
    state, _ = env.reset()
    total_reward = 0

    for t in range(10000):  # 최대 타임스텝
        # 상태를 텐서로 변환
        state_tensor = torch.tensor(state.reshape(1, agent.state_dim), **agent.tensor_args)

        # 행동 선택 (no_grad로 그래프 생성 방지)
        with torch.no_grad():
            action = agent.get_action(state_tensor)

        # numpy 변환 및 차원 조정
        action_cpu = action.detach().cpu().numpy().reshape(agent.action_dim)

        # 환경에 행동 적용
        next_state, reward, terminated, truncated, _ = env.step(action_cpu)
        total_reward += reward

        # 다음 상태 업데이트
        state = next_state

        # 종료 조건
        if terminated or truncated:
            break

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test()