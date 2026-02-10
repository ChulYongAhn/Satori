"""
학습된 AI를 이용한 실전 자동 매매
"""
import time
from datetime import datetime
import requests
import config
from data_collector import KISDataCollector
from train import load_model
from trading_env import TradingEnv


class AutoTrader:
    """자동 매매 실행기"""

    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol
        self.collector = KISDataCollector()
        self.model = None
        self.position = None  # None, "long"
        self.shares = 0

    def load_ai_model(self, model_path: str = None):
        """학습된 AI 모델 로드"""
        if model_path is None:
            model_path = f"models/ppo_{self.symbol}"

        self.model = load_model(model_path)
        print(f"Loaded model: {model_path}")

    def get_current_price(self) -> float:
        """현재가 조회"""
        data = self.collector.get_us_stock_price(self.symbol)
        price = float(data["output"]["last"])
        return price

    def place_order(self, side: str, quantity: float):
        """주문 실행 (매수/매도)"""
        url = f"{config.BASE_URL}/uapi/overseas-stock/v1/trading/order"

        # 모의투자 vs 실전투자 tr_id
        if config.IS_PAPER_TRADING:
            tr_id = "VTTT1002U" if side == "buy" else "VTTT1001U"
        else:
            tr_id = "JTTT1002U" if side == "buy" else "JTTT1006U"

        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.collector.get_access_token()}",
            "appkey": config.APP_KEY,
            "appsecret": config.APP_SECRET,
            "tr_id": tr_id
        }

        body = {
            "CANO": config.ACCOUNT_NO,
            "ACNT_PRDT_CD": config.ACCOUNT_PROD,
            "OVRS_EXCG_CD": "NASD",  # 나스닥
            "PDNO": self.symbol,
            "ORD_QTY": str(int(quantity)),
            "OVRS_ORD_UNPR": "0",  # 시장가
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00"  # 시장가 주문
        }

        response = requests.post(url, headers=headers, json=body)
        result = response.json()

        if result.get("rt_cd") == "0":
            print(f"Order placed: {side.upper()} {quantity} {self.symbol}")
            return True
        else:
            print(f"Order failed: {result.get('msg1')}")
            return False

    def get_ai_action(self, observation) -> int:
        """AI 모델로 행동 결정"""
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def run_once(self):
        """한 번의 매매 판단 실행"""
        try:
            # 최신 데이터 수집
            self.collector.collect_all_symbols([self.symbol])

            # 환경 생성 및 관측
            env = TradingEnv(symbol=self.symbol)
            obs, _ = env.reset()

            # 마지막 스텝으로 이동
            while env.current_step < len(env.df) - 2:
                obs, _, _, _, _ = env.step(0)  # 홀드

            # AI 판단
            action = self.get_ai_action(obs)

            current_price = self.get_current_price()
            action_names = ["HOLD", "BUY", "SELL"]

            print(f"\n[{datetime.now()}]")
            print(f"Symbol: {self.symbol}")
            print(f"Price: ${current_price:.2f}")
            print(f"AI Decision: {action_names[action]}")

            # 주문 실행
            if action == 1 and self.position is None:  # 매수
                quantity = config.INITIAL_BALANCE * config.MAX_POSITION_SIZE / current_price
                if self.place_order("buy", quantity):
                    self.position = "long"
                    self.shares = quantity

            elif action == 2 and self.position == "long":  # 매도
                if self.place_order("sell", self.shares):
                    self.position = None
                    self.shares = 0

        except Exception as e:
            print(f"Error: {e}")

    def run_loop(self, interval_minutes: int = 5):
        """주기적 매매 루프"""
        print(f"Starting auto trading loop (interval: {interval_minutes} min)")

        while True:
            self.run_once()
            time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    trader = AutoTrader(symbol="AAPL")
    trader.load_ai_model()
    trader.run_once()
