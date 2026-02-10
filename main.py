"""
Satori - 자가 학습 미국 주식 단타 AI
메인 실행 파일
"""
import argparse
import sys
from pathlib import Path

import config
from data_collector import KISDataCollector
from train import train, load_model
from trade import AutoTrader
from retrain import retrain, daily_retrain


def cmd_collect(args):
    """데이터 수집"""
    print("=== Data Collection ===")
    collector = KISDataCollector()

    symbols = args.symbols.split(",") if args.symbols else config.TRADING_SYMBOLS
    collector.collect_all_symbols(symbols)
    print("Data collection completed!")


def cmd_train(args):
    """모델 학습"""
    print("=== Model Training ===")
    symbol = args.symbol or "AAPL"
    timesteps = args.timesteps or config.TRAINING_TIMESTEPS

    train(symbol=symbol, total_timesteps=timesteps)
    print("Training completed!")


def cmd_trade(args):
    """자동 매매 실행"""
    print("=== Auto Trading ===")
    symbol = args.symbol or "AAPL"

    trader = AutoTrader(symbol=symbol)
    trader.load_ai_model()

    if args.loop:
        trader.run_loop(interval_minutes=args.interval)
    else:
        trader.run_once()


def cmd_retrain(args):
    """재학습"""
    print("=== Model Retraining ===")

    if args.all:
        daily_retrain()
    else:
        symbol = args.symbol or "AAPL"
        retrain(symbol=symbol)


def cmd_status(args):
    """현재 상태 확인"""
    print("=== Satori Status ===\n")

    # 데이터 확인
    data_dir = Path("data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        print(f"Data files: {len(csv_files)}")
        for f in csv_files:
            print(f"  - {f.name}")
    else:
        print("Data files: None")

    print()

    # 모델 확인
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.zip"))
        print(f"Trained models: {len(model_files)}")
        for f in model_files:
            print(f"  - {f.name}")
    else:
        print("Trained models: None")

    print()

    # 설정 확인
    print("Configuration:")
    print(f"  Paper Trading: {config.IS_PAPER_TRADING}")
    print(f"  Trading Symbols: {config.TRADING_SYMBOLS}")
    print(f"  Initial Balance: ${config.INITIAL_BALANCE}")
    print(f"  Training Timesteps: {config.TRAINING_TIMESTEPS}")


def main():
    parser = argparse.ArgumentParser(
        description="Satori - Self-learning US Stock Day Trading AI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # collect 명령
    collect_parser = subparsers.add_parser("collect", help="Collect stock data")
    collect_parser.add_argument(
        "-s", "--symbols",
        help="Comma-separated symbols (default: from config)"
    )

    # train 명령
    train_parser = subparsers.add_parser("train", help="Train AI model")
    train_parser.add_argument(
        "-s", "--symbol",
        default="AAPL",
        help="Stock symbol to train on"
    )
    train_parser.add_argument(
        "-t", "--timesteps",
        type=int,
        help="Training timesteps"
    )

    # trade 명령
    trade_parser = subparsers.add_parser("trade", help="Run auto trading")
    trade_parser.add_argument(
        "-s", "--symbol",
        default="AAPL",
        help="Stock symbol to trade"
    )
    trade_parser.add_argument(
        "-l", "--loop",
        action="store_true",
        help="Run in loop mode"
    )
    trade_parser.add_argument(
        "-i", "--interval",
        type=int,
        default=5,
        help="Loop interval in minutes"
    )

    # retrain 명령
    retrain_parser = subparsers.add_parser("retrain", help="Retrain model with new data")
    retrain_parser.add_argument(
        "-s", "--symbol",
        default="AAPL",
        help="Stock symbol to retrain"
    )
    retrain_parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Retrain all symbols"
    )

    # status 명령
    subparsers.add_parser("status", help="Show current status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "collect": cmd_collect,
        "train": cmd_train,
        "trade": cmd_trade,
        "retrain": cmd_retrain,
        "status": cmd_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
