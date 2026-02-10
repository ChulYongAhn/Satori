"""
한국투자증권 API를 통한 미국 주식 데이터 수집
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import config


class KISDataCollector:
    """한국투자증권 OpenAPI 데이터 수집기"""

    def __init__(self):
        self.base_url = config.BASE_URL
        self.app_key = config.APP_KEY
        self.app_secret = config.APP_SECRET
        self.access_token = None
        self.token_expires = None

    def get_access_token(self) -> str:
        """접근 토큰 발급"""
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return self.access_token

        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()

        data = response.json()
        self.access_token = data["access_token"]
        self.token_expires = datetime.now() + timedelta(hours=23)

        return self.access_token

    def _get_headers(self, tr_id: str) -> dict:
        """API 요청 헤더 생성"""
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_access_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }

    def get_us_stock_price(self, symbol: str) -> dict:
        """미국 주식 현재가 조회"""
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"

        # 모의투자 vs 실전투자 tr_id
        tr_id = "HHDFS00000300"

        headers = self._get_headers(tr_id)
        params = {
            "AUTH": "",
            "EXCD": "NAS",  # NAS: 나스닥, NYS: 뉴욕, AMS: 아멕스
            "SYMB": symbol
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        return response.json()

    def get_us_stock_daily(self, symbol: str, period: str = "D", count: int = 100) -> pd.DataFrame:
        """미국 주식 일봉 데이터 조회"""
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"

        tr_id = "HHDFS76240000"
        headers = self._get_headers(tr_id)

        # 거래소 코드 결정
        excd = "NAS"  # 기본값 나스닥

        params = {
            "AUTH": "",
            "EXCD": excd,
            "SYMB": symbol,
            "GUBN": "0",  # 0: 일, 1: 주, 2: 월
            "BYMD": "",   # 기준일자 (빈값: 최근)
            "MODP": "1"   # 수정주가 반영
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("rt_cd") != "0":
            raise Exception(f"API Error: {data.get('msg1')}")

        # DataFrame 변환
        records = data.get("output2", [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # 컬럼명 변환 및 타입 변환
        df = df.rename(columns={
            "xymd": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "clos": "close",
            "tvol": "volume"
        })

        # 필요한 컬럼만 선택
        df = df[["date", "open", "high", "low", "close", "volume"]]

        # 타입 변환
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])

        # 날짜순 정렬
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def save_stock_data(self, symbol: str, df: pd.DataFrame):
        """주가 데이터 저장"""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        filepath = data_dir / f"{symbol}.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")

    def collect_all_symbols(self, symbols: list = None):
        """모든 종목 데이터 수집"""
        if symbols is None:
            symbols = config.TRADING_SYMBOLS

        for symbol in symbols:
            try:
                print(f"Collecting {symbol}...")
                df = self.get_us_stock_daily(symbol)
                self.save_stock_data(symbol, df)
                time.sleep(0.5)  # API 호출 제한 대응
            except Exception as e:
                print(f"Error collecting {symbol}: {e}")


if __name__ == "__main__":
    collector = KISDataCollector()
    collector.collect_all_symbols()
