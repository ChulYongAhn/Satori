"""
한국투자증권 API를 사용한 S&P 500 기업 분봉 데이터 수집
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import logging
from dotenv import load_dotenv
from sp500_tickers import SP500_TICKERS, get_exchange

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KISDataCollector:
    """한국투자증권 API를 통한 미국 주식 데이터 수집"""

    def __init__(self, app_key: str = None, app_secret: str = None, account_no: str = None, is_real: bool = False):
        """
        Parameters:
        -----------
        app_key: API 앱 키 (None이면 환경변수에서 자동 로드)
        app_secret: API 앱 시크릿 (None이면 환경변수에서 자동 로드)
        account_no: 계좌번호 (None이면 환경변수에서 자동 로드)
        is_real: 실전투자(True) / 모의투자(False)
        """
        # 환경변수에서 자동 로드
        if is_real:
            self.app_key = app_key or os.getenv("KIS_APP_KEY")
            self.app_secret = app_secret or os.getenv("KIS_APP_SECRET")
            self.account_no = account_no or os.getenv("KIS_ACCOUNT_NO")
            self.base_url = "https://openapi.koreainvestment.com:9443"
        else:
            self.app_key = app_key or os.getenv("KIS_APP_KEY_VIRTUAL")
            self.app_secret = app_secret or os.getenv("KIS_APP_SECRET_VIRTUAL")
            self.account_no = account_no or os.getenv("KIS_ACCOUNT_NO_VIRTUAL")
            self.base_url = "https://openapivts.koreainvestment.com:29443"

        self.access_token = None
        self.token_expire_time = None

        # API 호출 제한 관리
        self.call_count = 0
        self.last_reset_time = datetime.now()
        self.calls_per_second = 10  # 초당 최대 호출 횟수

    def get_access_token(self) -> str:
        """접근 토큰 발급/갱신"""
        if self.access_token and self.token_expire_time and datetime.now() < self.token_expire_time:
            return self.access_token

        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()

        token_data = res.json()
        self.access_token = token_data["access_token"]
        # 토큰 만료 시간 설정 (보통 24시간)
        self.token_expire_time = datetime.now() + timedelta(hours=23)

        return self.access_token

    def make_hash(self, data: dict) -> str:
        """해시값 생성"""
        json_data = json.dumps(data, ensure_ascii=False).encode()
        return hashlib.sha256(json_data).hexdigest()

    def rate_limit(self):
        """API 호출 제한 관리"""
        current_time = datetime.now()

        # 1초가 지났으면 카운터 리셋
        if (current_time - self.last_reset_time).total_seconds() >= 1:
            self.call_count = 0
            self.last_reset_time = current_time

        # 초당 호출 제한 도달시 대기
        if self.call_count >= self.calls_per_second:
            sleep_time = 1 - (current_time - self.last_reset_time).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.call_count = 0
            self.last_reset_time = datetime.now()

        self.call_count += 1

    def get_minute_data(self, ticker: str, date: str) -> Optional[pd.DataFrame]:
        """
        특정 종목의 분봉 데이터 조회

        Parameters:
        -----------
        ticker: 종목 티커 (예: AAPL)
        date: 조회 날짜 (YYYYMMDD)

        Returns:
        --------
        DataFrame with columns: time, open, high, low, close, volume
        """
        self.rate_limit()

        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"

        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "HHDFS76950200"  # 해외주식 분/시간봉 조회
        }

        params = {
            "AUTH": "",
            "EXCD": "NAS",  # 나스닥 (NYSE 종목은 NYS)
            "SYMB": ticker,
            "NMIN": "1",  # 1분봉
            "PINC": "0",  # 0: 처음부터
            "NEXT": "",
            "NREC": "120",  # 최대 120개 (2시간치)
            "FILL": "",
            "KEYB": ""
        }

        try:
            res = requests.get(url, headers=headers, params=params)
            res.raise_for_status()

            data = res.json()

            if data.get("rt_cd") != "0":
                logger.error(f"Error fetching {ticker}: {data.get('msg1', 'Unknown error')}")
                return None

            output2 = data.get("output2", [])
            if not output2:
                logger.warning(f"No data for {ticker} on {date}")
                return None

            # DataFrame 생성
            df = pd.DataFrame(output2)
            df = df.rename(columns={
                'xymd': 'date',
                'xhms': 'time',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'clos': 'close',
                'tvol': 'volume'
            })

            # 필요한 컬럼만 선택
            df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]

            # 데이터 타입 변환
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')

            # datetime 컬럼 생성
            df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M%S')

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def get_full_day_minute_data(self, ticker: str, date: str, exchange: str = "NAS") -> Optional[pd.DataFrame]:
        """
        특정 종목의 하루 전체 분봉 데이터 조회 (여러 번 호출하여 결합)

        Parameters:
        -----------
        ticker: 종목 티커
        date: 조회 날짜 (YYYYMMDD)
        exchange: 거래소 (NAS: 나스닥, NYS: 뉴욕증권거래소)
        """
        all_data = []
        next_key = ""

        for i in range(10):  # 최대 10번 호출 (약 20시간치 데이터)
            self.rate_limit()

            url = f"{self.base_url}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"

            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self.get_access_token()}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "HHDFS76950200"
            }

            params = {
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": ticker,
                "NMIN": "1",
                "PINC": "0" if i == 0 else "1",  # 첫 호출은 0, 이후는 1
                "NEXT": next_key,
                "NREC": "120",
                "FILL": "",
                "KEYB": ""
            }

            try:
                res = requests.get(url, headers=headers, params=params)
                res.raise_for_status()

                data = res.json()

                if data.get("rt_cd") != "0":
                    logger.error(f"Error fetching {ticker}: {data.get('msg1', 'Unknown error')}")
                    break

                output2 = data.get("output2", [])
                if not output2:
                    break

                # 첫 번째 데이터의 키 확인 (디버깅용)
                if i == 0 and output2:
                    logger.info(f"Available columns for {ticker}: {list(output2[0].keys())}")

                all_data.extend(output2)

                # 다음 키가 없으면 종료
                if not data.get("output1", {}).get("nkey"):
                    break

                next_key = data["output1"]["nkey"]

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                break

        if not all_data:
            return None

        # DataFrame 생성
        df = pd.DataFrame(all_data)

        # 실제 컬럼명 확인 후 매핑
        logger.info(f"Original columns: {df.columns.tolist()}")

        # 컬럼 이름 변경 - 실제 API 응답에 맞춰 수정
        rename_mapping = {}
        if 'xymd' in df.columns:
            rename_mapping['xymd'] = 'date'
        if 'xhms' in df.columns:
            rename_mapping['xhms'] = 'time'
        if 'clos' in df.columns:
            rename_mapping['clos'] = 'close'
        if 'tvol' in df.columns:
            rename_mapping['tvol'] = 'volume'

        df = df.rename(columns=rename_mapping)

        # 필요한 컬럼 확인 후 선택
        required_cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]

        if len(available_cols) < len(required_cols):
            logger.warning(f"Missing columns: {set(required_cols) - set(available_cols)}")
            logger.warning(f"Available columns: {df.columns.tolist()}")

        df = df[available_cols]

        # 데이터 타입 변환 (컬럼이 존재하는 경우만)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')

        # datetime 컬럼 생성 (date와 time 컬럼이 모두 있는 경우만)
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M%S')

        # 중복 제거 및 정렬 (datetime 컬럼이 있는 경우만)
        if 'datetime' in df.columns:
            df = df.drop_duplicates(subset=['datetime'])
            df = df.sort_values('datetime')

        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 계산 추가

        Parameters:
        -----------
        df: OHLCV 데이터프레임

        Returns:
        --------
        기술적 지표가 추가된 DataFrame
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # 1. RSI (Relative Strength Index) - 14분 기준
        def calculate_rsi(data, period=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        if 'close' in df.columns:
            df['rsi'] = calculate_rsi(df['close'])

        # 2. 이동평균선 (Moving Averages)
        if 'close' in df.columns:
            df['sma_5'] = df['close'].rolling(window=5).mean()    # 5분 이동평균
            df['sma_20'] = df['close'].rolling(window=20).mean()  # 20분 이동평균
            df['sma_60'] = df['close'].rolling(window=60).mean()  # 60분 이동평균

            # EMA (Exponential Moving Average)
            df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # 3. MACD (Moving Average Convergence Divergence)
        if 'close' in df.columns:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 4. 볼린저 밴드 (Bollinger Bands)
        if 'close' in df.columns:
            period = 20
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 5. ATR (Average True Range) - 변동성 지표
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=14).mean()

        # 6. 스토캐스틱 (Stochastic Oscillator)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            period = 14
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # 7. OBV (On Balance Volume) - 거래량 지표
        if 'volume' in df.columns and 'close' in df.columns:
            obv = []
            obv_val = 0
            for i in range(len(df)):
                if i == 0:
                    obv.append(0)
                else:
                    if df['close'].iloc[i] > df['close'].iloc[i-1]:
                        obv_val += df['volume'].iloc[i]
                    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                        obv_val -= df['volume'].iloc[i]
                    obv.append(obv_val)
            df['obv'] = obv

        # 8. VWAP (Volume Weighted Average Price)
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # 9. 변화율 (Price Rate of Change)
        if 'close' in df.columns:
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(periods=5)  # 5분 변화율
            df['price_change_20'] = df['close'].pct_change(periods=20)  # 20분 변화율

        # 10. 거래량 이동평균
        if 'volume' in df.columns:
            df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']  # 거래량 비율

        return df

    def collect_sp500_minute_data(self, date: str = None, ticker_list: List[str] = None):
        """
        S&P 500 전체 종목의 분봉 데이터 수집 및 저장

        Parameters:
        -----------
        date: 수집할 날짜 (YYYYMMDD), None이면 어제 날짜
        ticker_list: 수집할 티커 리스트, None이면 S&P 500 전체
        """
        if date is None:
            # 어제 날짜 계산 (주말 고려)
            yesterday = datetime.now() - timedelta(days=1)
            # 토요일이면 금요일로, 일요일이면 금요일로
            if yesterday.weekday() == 5:  # Saturday
                yesterday -= timedelta(days=1)
            elif yesterday.weekday() == 6:  # Sunday
                yesterday -= timedelta(days=2)
            date = yesterday.strftime("%Y%m%d")

        # 저장 경로 생성
        save_dir = f"StockHistory/{date}"
        os.makedirs(save_dir, exist_ok=True)

        # S&P 500 티커 리스트 (주요 종목만 예시로 포함)
        if ticker_list is None:
            ticker_list = get_sp500_tickers()

        logger.info(f"Starting data collection for {len(ticker_list)} tickers on {date}")

        success_count = 0
        fail_count = 0

        for i, ticker in enumerate(ticker_list, 1):
            logger.info(f"Processing {i}/{len(ticker_list)}: {ticker}")

            # 거래소 판별 (sp500_tickers 모듈의 get_exchange 함수 사용)
            exchange = get_exchange(ticker)

            # 데이터 수집
            df = self.get_full_day_minute_data(ticker, date, exchange)

            if df is not None and not df.empty:
                # 기술적 지표 계산 추가
                df = self.calculate_technical_indicators(df)

                # 파일 저장
                file_path = os.path.join(save_dir, f"{ticker}_{date}.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {ticker} data to {file_path} with technical indicators")
                success_count += 1
            else:
                logger.warning(f"No data available for {ticker}")
                fail_count += 1

            # API 제한 고려하여 잠시 대기
            time.sleep(0.5)

        logger.info(f"Data collection completed: Success={success_count}, Failed={fail_count}")

        # 수집 요약 정보 저장
        summary = {
            "date": date,
            "total_tickers": len(ticker_list),
            "success": success_count,
            "failed": fail_count,
            "timestamp": datetime.now().isoformat()
        }

        summary_path = os.path.join(save_dir, "collection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


def get_sp500_tickers() -> List[str]:
    """S&P 500 전체 티커 리스트 반환"""
    # sp500_tickers 모듈에서 전체 리스트 가져오기
    return SP500_TICKERS.copy()


# 사용 예시
if __name__ == "__main__":
    # 데이터 수집기 초기화 (환경변수에서 자동으로 로드)
    collector = KISDataCollector(
        is_real=False  # 모의투자 (False면 KIS_APP_KEY_VIRTUAL 사용)
    )

    # S&P 500 전체 데이터 수집 (어제 날짜)
    # 주의: 전체 500개 종목 수집시 시간이 오래 걸리고 API 제한에 걸릴 수 있음

    # 테스트용 (5개 종목만)
    # test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    # collector.collect_sp500_minute_data(ticker_list=test_tickers)

    # 전체 S&P 500 수집 (약 500개 종목)
    print(f"Starting S&P 500 data collection for {len(SP500_TICKERS)} tickers...")
    print("This will take approximately 10-15 minutes...")
    collector.collect_sp500_minute_data()  # ticker_list 파라미터 없이 호출하면 전체 S&P 500 수집