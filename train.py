"""
PPO 에이전트 학습
"""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import config
from trading_env import TradingEnv


class TradingCallback(BaseCallback):
    """학습 진행 상황 콜백"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.verbose > 0:
            mean_reward = self.model.ep_info_buffer[-1]["r"] if self.model.ep_info_buffer else 0
            print(f"Timestep: {self.num_timesteps}, Reward: {mean_reward:.2f}")


def create_env(symbol: str = "AAPL"):
    """학습 환경 생성"""
    def _init():
        return TradingEnv(symbol=symbol)
    return _init


def train(symbol: str = "AAPL", total_timesteps: int = None, save_path: str = None):
    """PPO 모델 학습"""

    if total_timesteps is None:
        total_timesteps = config.TRAINING_TIMESTEPS

    if save_path is None:
        save_path = f"models/ppo_{symbol}"

    # 환경 생성
    env = DummyVecEnv([create_env(symbol)])

    # 모델 생성
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/"
    )

    # 콜백 설정
    callback = TradingCallback(verbose=1)

    # 학습 실행
    print(f"Starting training for {symbol}...")
    print(f"Total timesteps: {total_timesteps}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # 모델 저장
    Path("models").mkdir(exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model


def load_model(path: str):
    """저장된 모델 로드"""
    return PPO.load(path)


if __name__ == "__main__":
    train()
