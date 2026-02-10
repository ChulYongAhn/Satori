"""
한국투자증권 API 설정
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 한국투자증권 API 키
APP_KEY = os.getenv("KIS_APP_KEY", "")
APP_SECRET = os.getenv("KIS_APP_SECRET", "")

# 계좌 정보
ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")  # 계좌번호 (8자리)
ACCOUNT_PROD = os.getenv("KIS_ACCOUNT_PROD", "01")  # 계좌상품코드

# API 환경 설정
IS_PAPER_TRADING = True  # True: 모의투자, False: 실전투자

# API URL
if IS_PAPER_TRADING:
    BASE_URL = "https://openapivts.koreainvestment.com:29443"  # 모의투자
else:
    BASE_URL = "https://openapi.koreainvestment.com:9443"  # 실전투자

# 학습 설정
TRAINING_TIMESTEPS = 100000  # PPO 학습 스텝 수
LEARNING_RATE = 0.0003

# 매매 설정
TRADING_SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]  # 거래 대상 종목
INITIAL_BALANCE = 10000  # 초기 자본 (USD)
MAX_POSITION_SIZE = 0.2  # 최대 포지션 비율 (20%)
