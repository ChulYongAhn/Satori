# 🤖 Stock AI — 자가 학습 미국 주식 단타 AI

## 프로젝트 개요

스스로 학습하고, 매매 방법을 바꾸며, 수익률을 점점 개선해나가는 **미국 주식 단타 전용 AI**.
사람이 규칙을 정해주지 않아도 AI가 수만 번의 시행착오를 통해 **알아서 최적의 매매법을 찾아간다.**

## 기술 스택

| 구분 | 기술 | 역할 |
|------|------|------|
| 언어 | Python | 전체 개발 |
| AI 엔진 | Stable-Baselines3 (PPO) | 강화학습으로 매매 전략 자가 학습 |
| 딥러닝 | PyTorch | PPO 내부 신경망 |
| 증권사 | 한국투자증권 OpenAPI | 미국 주식 시세 조회 + 자동 주문 |
| 데이터 | pandas, numpy | 주가 데이터 처리 |

## 핵심 원리

```
PPO (Proximal Policy Optimization)
= 시행착오를 통해 스스로 배우는 강화학습 알고리즘

1) 과거 주가 데이터로 가상 시장(시뮬레이터)을 만든다
2) AI가 그 안에서 수만 번 매매를 연습한다
3) 돈 벌면 +보상, 잃으면 -벌점 → 점점 나은 전략을 스스로 찾아간다
4) 충분히 학습된 AI를 실전에 투입한다
5) 매일 새 데이터로 재학습 → 시장 변화에 맞춰 계속 성장한다
```

## 프로젝트 구조

```
stock_ai/
├── data/                  # 주가 데이터 저장
├── models/                # 학습된 AI 모델 저장
├── logs/                  # 학습 로그
├── config.py              # 한투 API 키 설정
├── data_collector.py      # 한투 API로 미국 주식 데이터 수집
├── trading_env.py         # AI 매매 연습 환경 (가상 시장)
├── train.py               # AI 학습 (PPO)
├── trade.py               # 실전 자동 매매
├── retrain.py             # 매일 자동 재학습 (자가 성장)
├── requirements.txt       # 필요한 라이브러리
└── README.md
```

## 각 파일 역할

- **config.py** — 한투 API 키(APP_KEY, APP_SECRET) 설정
- **data_collector.py** — 한투 OpenAPI로 미국 주식 분봉/일봉 데이터를 가져와 data/ 폴더에 저장
- **trading_env.py** — 주가 데이터를 기반으로 AI가 매수/매도/홀드를 연습할 수 있는 가상 시장 환경
- **train.py** — PPO 알고리즘으로 AI를 학습시키고, 학습된 모델을 models/ 폴더에 저장
- **trade.py** — 학습된 AI가 실시간 시세를 보고 자동으로 매매 주문을 실행
- **retrain.py** — 매일 장 마감 후 최신 데이터를 반영하여 AI를 재학습 (자동 성장 루프)

## 동작 흐름

```
[1단계] 데이터 수집
  한투 API → 미국 주식 분봉 데이터 → data/ 저장

[2단계] AI 학습
  data/ 데이터 → 가상 시장 환경 → PPO가 수만 번 연습 → models/ 저장

[3단계] 실전 매매
  실시간 시세 → 학습된 AI가 판단 → 매수/매도/홀드 → 한투 API로 주문

[4단계] 자동 성장
  매일 장 마감 → 오늘 데이터 추가 → 재학습 → 모델 업데이트
  → AI가 최신 시장 패턴에 맞춰 계속 진화
```

## 설치 방법

```bash
# 프로젝트 폴더 생성
mkdir stock_ai
cd stock_ai

# 가상환경 생성 및 활성화
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # Mac/Linux

# 라이브러리 설치
pip install -r requirements.txt
```

## requirements.txt

```
stable-baselines3
gymnasium
torch
numpy
pandas
matplotlib
requests
websocket-client
```

## 거래 조건

- **시장**: 미국 주식 (NYSE, NASDAQ)
- **증권사**: 한국투자증권 OpenAPI
- **수수료**: 거의 0 (미국 주식 거래세 없음)
- **거래 시간**: 한국 시간 23:30 ~ 06:00 (서머타임 22:30 ~ 05:00)
- **AI 비용**: 0원 (오픈소스, 본인 PC에서 학습)

## 개발 로드맵

- [ ] 1단계: 한투 API 연동 + 미국 주식 데이터 수집
- [ ] 2단계: 피처 엔지니어링 (기술적 지표 계산)
- [ ] 3단계: 가상 매매 환경(trading_env) 구축
- [ ] 4단계: PPO 에이전트 학습
- [ ] 5단계: 한투 모의투자로 페이퍼 트레이딩 검증
- [ ] 6단계: 자동 재학습 루프 구현
- [ ] 7단계: 소액 실전 투입

## 주의사항

- 이 프로젝트는 학습 및 실험 목적입니다
- 실전 투입 전 반드시 모의투자로 충분한 검증이 필요합니다
- 투자 손실에 대한 책임은 본인에게 있습니다
- 백테스트 수익률 ≠ 실전 수익률 (과적합 주의)
