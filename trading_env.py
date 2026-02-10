"""
강화학습을 위한 주식 거래 환경 (Gymnasium 기반)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import config


class TradingEnv(gym.Env):
    """주식 거래 강화학습 환경"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, symbol: str = "AAPL", window_size: int = 20):
        super().__init__()

        self.symbol = symbol
        self.window_size = window_size

        # 데이터 로드
        self.df = self._load_data()

        # 액션 공간: 0=홀드, 1=매수, 2=매도
        self.action_space = spaces.Discrete(3)

        # 관찰 공간: [window_size x 5 (OHLCV)] + [포지션, 수익률]
        obs_shape = (window_size * 5 + 2,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # 초기 상태
        self.initial_balance = config.INITIAL_BALANCE
        self.reset()

    def _load_data(self) -> pd.DataFrame:
        """주가 데이터 로드"""
        filepath = Path("data") / f"{self.symbol}.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["date"])
        return df

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관측값 반환"""
        # 윈도우 크기만큼의 OHLCV 데이터
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        window_data = self.df.iloc[start_idx:end_idx]

        # 정규화 (현재 가격 대비 비율)
        current_price = self.df.iloc[self.current_step]["close"]

        ohlcv = []
        for _, row in window_data.iterrows():
            ohlcv.extend([
                row["open"] / current_price - 1,
                row["high"] / current_price - 1,
                row["low"] / current_price - 1,
                row["close"] / current_price - 1,
                np.log1p(row["volume"]) / 20  # 볼륨 정규화
            ])

        # 포지션 정보
        position = 1.0 if self.shares_held > 0 else 0.0

        # 현재 수익률
        profit_ratio = (self.total_value - self.initial_balance) / self.initial_balance

        obs = np.array(ohlcv + [position, profit_ratio], dtype=np.float32)

        return obs

    @property
    def total_value(self) -> float:
        """현재 총 자산 가치"""
        current_price = self.df.iloc[self.current_step]["close"]
        return self.balance + self.shares_held * current_price

    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.current_step = self.window_size
        self.trades = []

        return self._get_observation(), {}

    def step(self, action: int):
        """한 스텝 실행"""
        current_price = self.df.iloc[self.current_step]["close"]

        reward = 0
        done = False

        # 액션 실행
        if action == 1:  # 매수
            if self.balance > 0 and self.shares_held == 0:
                # 전액 매수
                shares_to_buy = self.balance / current_price
                self.shares_held = shares_to_buy
                self.cost_basis = current_price
                self.balance = 0
                self.trades.append(("BUY", self.current_step, current_price))

        elif action == 2:  # 매도
            if self.shares_held > 0:
                # 전량 매도
                self.balance = self.shares_held * current_price
                profit = (current_price - self.cost_basis) / self.cost_basis
                reward = profit * 100  # 수익률을 보상으로
                self.shares_held = 0
                self.trades.append(("SELL", self.current_step, current_price))

        # 다음 스텝으로
        self.current_step += 1

        # 에피소드 종료 조건
        if self.current_step >= len(self.df) - 1:
            done = True
            # 마지막에 포지션 정리
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0

        # 최종 수익률 보상
        if done:
            final_return = (self.total_value - self.initial_balance) / self.initial_balance
            reward += final_return * 10

        obs = self._get_observation()
        truncated = False
        info = {
            "total_value": self.total_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "trades": len(self.trades)
        }

        return obs, reward, done, truncated, info

    def render(self):
        """현재 상태 출력"""
        current_price = self.df.iloc[self.current_step]["close"]
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares_held:.4f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Trades: {len(self.trades)}")
        print("-" * 40)
