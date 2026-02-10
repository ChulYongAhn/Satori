"""
매일 자동 재학습 - AI 자가 성장 루프
"""
from datetime import datetime
from pathlib import Path
import shutil
import config
from data_collector import KISDataCollector
from train import train, load_model
from trading_env import TradingEnv


def backup_model(symbol: str):
    """기존 모델 백업"""
    model_path = Path(f"models/ppo_{symbol}.zip")
    if model_path.exists():
        backup_dir = Path("models/backup")
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"ppo_{symbol}_{timestamp}.zip"
        shutil.copy(model_path, backup_path)
        print(f"Backed up model to {backup_path}")


def evaluate_model(symbol: str, model_path: str = None) -> float:
    """모델 성능 평가"""
    if model_path is None:
        model_path = f"models/ppo_{symbol}"

    model = load_model(model_path)
    env = TradingEnv(symbol=symbol)

    total_rewards = []

    for _ in range(5):  # 5번 평가
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if truncated:
                break

        total_rewards.append(episode_reward)
        final_value = info["total_value"]

    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward


def retrain(symbol: str = "AAPL", min_improvement: float = 0.05):
    """재학습 실행"""
    print(f"\n{'='*50}")
    print(f"Retraining started: {datetime.now()}")
    print(f"Symbol: {symbol}")
    print(f"{'='*50}\n")

    # 1. 최신 데이터 수집
    print("[1/5] Collecting latest data...")
    collector = KISDataCollector()
    collector.collect_all_symbols([symbol])

    # 2. 기존 모델 성능 평가
    print("\n[2/5] Evaluating current model...")
    try:
        old_score = evaluate_model(symbol)
        print(f"Current model score: {old_score:.2f}")
    except:
        old_score = float("-inf")
        print("No existing model found")

    # 3. 기존 모델 백업
    print("\n[3/5] Backing up current model...")
    backup_model(symbol)

    # 4. 재학습
    print("\n[4/5] Training new model...")
    train(symbol=symbol, total_timesteps=config.TRAINING_TIMESTEPS)

    # 5. 새 모델 평가 및 비교
    print("\n[5/5] Evaluating new model...")
    new_score = evaluate_model(symbol)
    print(f"New model score: {new_score:.2f}")

    improvement = (new_score - old_score) / abs(old_score) if old_score != 0 else float("inf")

    if improvement > min_improvement:
        print(f"\nModel improved by {improvement*100:.1f}%! Keeping new model.")
    else:
        print(f"\nModel did not improve enough ({improvement*100:.1f}%). Keeping new model anyway for fresh learning.")

    print(f"\nRetraining completed: {datetime.now()}")


def daily_retrain():
    """모든 종목 일일 재학습"""
    for symbol in config.TRADING_SYMBOLS:
        try:
            retrain(symbol)
        except Exception as e:
            print(f"Error retraining {symbol}: {e}")


if __name__ == "__main__":
    retrain("AAPL")
