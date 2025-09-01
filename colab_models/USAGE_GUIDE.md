# 🚀 AI 자동 트레이딩 봇 사용 가이드

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [설치 및 설정](#설치-및-설정)
3. [Colab에서 사용하기](#colab에서-사용하기)
4. [로컬에서 사용하기](#로컬에서-사용하기)
5. [실행 예시](#실행-예시)
6. [결과 해석](#결과-해석)
7. [문제 해결](#문제-해결)

## 🎯 시스템 개요

이 시스템은 다음과 같은 기능을 제공합니다:

- **🤖 자동 하이퍼파라미터 튜닝**: Optuna를 사용한 최적화
- **📊 다중 모델 지원**: LightGBM, XGBoost
- **🔄 실시간 성능 모니터링**: 학습 과정 추적
- **💾 자동 저장**: 모델과 결과 자동 저장
- **📈 시각화**: 성능 비교 및 분석 차트
- **🎯 트레이딩 신호**: 실시간 매매 신호 생성

## 🔧 설치 및 설정

### 1. 필수 요구사항
- Python 3.8 이상
- Supabase 계정 및 프로젝트
- 충분한 저장 공간 (최소 1GB)

### 2. 패키지 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 xgboost==1.7.6 lightgbm==4.0.0 pandas-ta==0.3.14b0 supabase optuna plotly
```

### 3. Supabase 설정

1. [Supabase](https://supabase.com)에서 계정 생성
2. 새 프로젝트 생성
3. 다음 테이블 생성:

```sql
-- 암호화폐 OHLCV 데이터 테이블
CREATE TABLE crypto_ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp BIGINT NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 인덱스 생성
CREATE INDEX idx_crypto_ohlcv_symbol_timeframe ON crypto_ohlcv(symbol, timeframe);
CREATE INDEX idx_crypto_ohlcv_timestamp ON crypto_ohlcv(timestamp);
```

4. API 키 복사:
   - Settings > API
   - Project URL과 anon public key 복사

## 🚀 Colab에서 사용하기

### 1단계: 노트북 업로드
1. Google Colab 열기
2. `colab_auto_training_template.ipynb` 파일 업로드
3. 런타임 > 런타임 유형 변경 > GPU 선택 (선택사항)

### 2단계: 설정 입력
```python
# Supabase 설정
SUPABASE_URL = "https://your-project.supabase.co"  # 실제 URL로 변경
SUPABASE_KEY = "your-anon-key"  # 실제 키로 변경

# 최적화 설정
SYMBOLS = ['ADAUSDT', 'BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['1m', '5m', '15m']
MODEL_TYPES = ['lightgbm', 'xgboost']
N_TRIALS = 100  # 하이퍼파라미터 최적화 시도 횟수
```

### 3단계: 실행
1. 모든 셀을 순서대로 실행
2. 자동으로 모든 최적화 완료
3. Google Drive에 결과 저장

## 💻 로컬에서 사용하기

### 1단계: 스크립트 실행

```bash
# 전체 시스템 실행
python run_auto_training.py \
    --supabase-url "https://your-project.supabase.co" \
    --supabase-key "your-anon-key" \
    --mode all \
    --symbols ADAUSDT BTCUSDT ETHUSDT \
    --timeframes 1m 5m 15m \
    --model-types lightgbm xgboost \
    --n-trials 100 \
    --output-dir output
```

### 2단계: 개별 모드 실행

```bash
# 훈련만 실행
python run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --mode train

# 평가만 실행
python run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --mode evaluate

# 신호 생성만 실행
python run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --mode signal
```

## 📊 실행 예시

### 예시 1: 빠른 테스트
```bash
python run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --symbols ADAUSDT \
    --timeframes 1m \
    --model-types lightgbm \
    --n-trials 20
```

### 예시 2: 전체 최적화
```bash
python run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --symbols ADAUSDT BTCUSDT ETHUSDT \
    --timeframes 1m 5m 15m 1h \
    --model-types lightgbm xgboost \
    --n-trials 200
```

### 예시 3: Python 코드로 실행
```python
from auto_hyperparameter_tuner import AutoTradingBot

# 봇 생성
bot = AutoTradingBot("your-url", "your-key")

# 단일 모델 최적화
model_name, metrics, best_params = bot.run_optimization(
    symbol='ADAUSDT',
    timeframe='1m',
    model_type='lightgbm',
    n_trials=100
)

print(f"모델: {model_name}")
print(f"성능: {metrics}")
```

## 📈 결과 해석

### 1. 성능 지표
- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (가중 평균)
- **Recall**: 재현율 (가중 평균)
- **F1-Score**: F1 점수 (가중 평균)
- **ROC-AUC**: ROC 곡선 아래 면적

### 2. 신호 강도
- **STRONG_BUY**: 강한 매수 신호 (신호 강도 ≥ 0.6)
- **BUY**: 매수 신호 (신호 강도 ≥ 0.4)
- **WEAK_BUY**: 약한 매수 신호 (신호 강도 ≥ 0.2)
- **NEUTRAL**: 중립 (신호 강도 ≥ 0.1)
- **SELL**: 매도 신호 (신호 강도 < 0.1)

### 3. 출력 파일
```
output/
├── models/                    # 훈련된 모델들
│   ├── ADAUSDT_1m_lightgbm_20241201_143022.pkl
│   └── ...
├── results/                   # 훈련 결과
│   ├── ADAUSDT_1m_lightgbm_20241201_143022_results.json
│   └── ...
├── evaluation_report.html     # 성능 평가 리포트
├── signal_dashboard.html      # 신호 대시보드
└── trading_signals.json       # 트레이딩 신호
```

## 🔍 문제 해결

### 1. 데이터 연결 오류
```
❌ 데이터를 찾을 수 없습니다: ADAUSDT 1m
```
**해결방법:**
- Supabase URL과 키 확인
- 데이터베이스에 데이터 존재 여부 확인
- 심볼명과 타임프레임 형식 확인

### 2. 메모리 부족 오류
```
❌ 메모리 부족
```
**해결방법:**
- 데이터 수집량 줄이기 (`limit` 파라미터 조정)
- 배치 크기 줄이기
- 더 작은 모델 사용

### 3. 하이퍼파라미터 최적화 실패
```
❌ 최적화 실패
```
**해결방법:**
- 시도 횟수 늘리기 (`n_trials` 증가)
- 파라미터 범위 조정
- 더 많은 데이터 사용

### 4. 모델 저장 실패
```
❌ 모델 저장 실패
```
**해결방법:**
- 디렉토리 권한 확인
- 저장 공간 확인
- 파일명 충돌 확인

## ⚡ 성능 최적화 팁

### 1. 데이터 최적화
- 충분한 데이터 확보 (최소 10,000개)
- 데이터 품질 확인
- 결측값 처리

### 2. 하이퍼파라미터 최적화
- 시도 횟수 늘리기 (100-500회)
- 다양한 파라미터 범위 시도
- 조기 종료 사용

### 3. 모델 앙상블
- 여러 모델 조합 사용
- 가중 평균으로 신호 생성
- 성능 기반 가중치 적용

### 4. 실시간 적용
- 정기적인 모델 업데이트
- 성능 모니터링
- 드리프트 감지

## 📞 지원

문제가 발생하거나 개선 사항이 있으면:

1. **로그 확인**: 실행 로그에서 오류 메시지 확인
2. **설정 검증**: Supabase 설정 및 파라미터 확인
3. **데이터 확인**: 데이터베이스 연결 및 데이터 품질 확인
4. **문서 참조**: README.md 및 코드 주석 참조

## 🔄 업데이트

시스템을 최신 버전으로 업데이트하려면:

```bash
# 코드 업데이트
git pull origin main

# 패키지 업데이트
pip install -r requirements.txt --upgrade
```

---

**마지막 업데이트**: 2024년 12월 1일
**버전**: 2.0.0
**작성자**: AI 트레이딩 봇 개발팀

