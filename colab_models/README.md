# 🤖 AI 자동 트레이딩 봇 시스템

## 📁 폴더 구조

```
colab_models/
├── README.md                           # 이 파일
├── auto_hyperparameter_tuner.py        # 자동 하이퍼파라미터 튜닝 시스템
├── colab_auto_training_template.ipynb  # Colab 자동 실행 템플릿
├── model_evaluator.py                  # 모델 성능 평가 도구
├── trading_signal_generator.py         # 트레이딩 신호 생성기
└── legacy/                             # 기존 노트북들 (백업)
    ├── binary_classification_*.ipynb
    ├── regression_*.ipynb
    └── *_training.ipynb
```

## 🚀 새로운 자동화 시스템

### 1. 자동 하이퍼파라미터 튜닝 (`auto_hyperparameter_tuner.py`)

**주요 기능:**
- 🔍 Optuna를 사용한 자동 하이퍼파라미터 최적화
- 📊 LightGBM, XGBoost 모델 지원
- 🎯 이진 분류 및 회귀 문제 해결
- 💾 자동 모델 저장 및 로드
- 📈 실시간 성능 모니터링

**사용법:**
```python
from auto_hyperparameter_tuner import AutoTradingBot

# Supabase 설정
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-key"

# 자동 트레이딩 봇 생성
bot = AutoTradingBot(SUPABASE_URL, SUPABASE_KEY)

# 최적화 실행
model_name, metrics, best_params = bot.run_optimization(
    symbol='ADAUSDT',
    timeframe='1m',
    model_type='lightgbm',
    n_trials=100
)
```

### 2. Colab 자동 실행 템플릿 (`colab_auto_training_template.ipynb`)

**주요 기능:**
- 🚀 Google Colab에서 자동 실행
- 📦 필요한 라이브러리 자동 설치
- 🔄 다중 심볼/타임프레임/모델 자동 최적화
- 📊 Plotly를 사용한 인터랙티브 시각화
- 💾 Google Drive 자동 저장

**사용법:**
1. Colab에서 노트북 업로드
2. Supabase 설정 입력
3. 실행 버튼 클릭
4. 자동으로 모든 최적화 완료

## 📊 기술적 지표

시스템에서 사용하는 기술적 지표들:

### 기본 지표
- **이동평균**: SMA(5,10,20,50), EMA(5,10,20,50)
- **MACD**: 빠른/느린 설정 (6,13,4) / (12,26,9)
- **RSI**: 다중 기간 (9,14,21)
- **볼린저 밴드**: 단기/장기 (10,20 기간)

### 고급 지표
- **스토캐스틱**: %K, %D
- **ATR**: Average True Range
- **ADX**: Average Directional Index
- **거래량 지표**: 거래량 SMA, EMA, 비율

### 시간 특성
- **세션 구분**: 아시아, 런던, 뉴욕 세션
- **주말 여부**: 주말/평일 구분
- **시간대**: 시간별 특성

## 🎯 목표 변수 생성

### 이진 분류
```python
# 미래 수익률 기반 이진 분류
df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
df['target'] = np.where(df['future_return'] > threshold, 1, 0)
```

**파라미터:**
- `lookforward`: 미래 예측 기간 (기본값: 5)
- `threshold`: 상승 임계값 (기본값: 0.002 = 0.2%)

## 🔧 하이퍼파라미터 최적화

### LightGBM 최적화 범위
```python
params = {
    'n_estimators': [100, 2000],
    'learning_rate': [0.01, 0.3],
    'max_depth': [3, 12],
    'num_leaves': [10, 100],
    'min_child_samples': [10, 100],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'reg_alpha': [1e-8, 10.0],
    'reg_lambda': [1e-8, 10.0]
}
```

### XGBoost 최적화 범위
```python
params = {
    'n_estimators': [100, 2000],
    'learning_rate': [0.01, 0.3],
    'max_depth': [3, 12],
    'min_child_weight': [1, 10],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'reg_alpha': [1e-8, 10.0],
    'reg_lambda': [1e-8, 10.0]
}
```

## 📈 성능 평가 지표

### 분류 성능
- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (가중 평균)
- **Recall**: 재현율 (가중 평균)
- **F1-Score**: F1 점수 (가중 평균)
- **ROC-AUC**: ROC 곡선 아래 면적

### 교차 검증
- **StratifiedKFold**: 5-fold 교차 검증
- **시계열 고려**: 시간 순서 유지

## 💾 모델 저장 및 로드

### 저장 형식
```python
# 모델 파일
models/{symbol}_{timeframe}_{model_type}_{timestamp}.pkl

# 결과 파일
results/{symbol}_{timeframe}_{model_type}_{timestamp}_results.json
```

### 결과 JSON 구조
```json
{
  "model_name": "ADAUSDT_1m_lightgbm_20241201_143022",
  "symbol": "ADAUSDT",
  "timeframe": "1m",
  "model_type": "lightgbm",
  "timestamp": "20241201_143022",
  "metrics": {
    "accuracy": 0.8234,
    "precision": 0.8156,
    "recall": 0.8234,
    "f1": 0.8195,
    "roc_auc": 0.7891
  },
  "best_params": {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 8,
    ...
  },
  "feature_columns": ["sma_5", "ema_9", ...]
}
```

## 🔄 자동화 워크플로우

### 1. 데이터 수집
```python
df = bot.collect_data(symbol='ADAUSDT', timeframe='1m', limit=100000)
```

### 2. 특성 계산
```python
df = bot.calculate_features(df)
```

### 3. 목표 변수 생성
```python
df = bot.create_target(df, lookforward=5, threshold=0.002)
```

### 4. 특성 준비
```python
X, y, feature_columns = bot.prepare_features(df)
```

### 5. 하이퍼파라미터 최적화
```python
best_params, best_score = bot.optimize_hyperparameters(X, y, 'lightgbm', 100)
```

### 6. 최종 모델 훈련
```python
model, metrics, test_results = bot.train_final_model(X, y, best_params, 'lightgbm')
```

### 7. 모델 저장
```python
model_name = bot.save_model(model, metrics, best_params, feature_columns, symbol, timeframe, model_type)
```

## 📊 시각화

### 성능 비교 차트
- **산점도**: Accuracy vs F1 Score
- **박스플롯**: 심볼별/타임프레임별 성능 분포
- **히트맵**: 모델별 성능 매트릭스

### 최적화 과정
- **학습 곡선**: 훈련/검증 손실
- **하이퍼파라미터 중요도**: Optuna 시각화
- **특성 중요도**: 모델별 중요 특성

## 🚨 주의사항

### 1. 데이터 품질
- 충분한 데이터 확보 (최소 10,000개)
- 결측값 처리
- 이상치 제거

### 2. 과적합 방지
- 교차 검증 사용
- 정규화 파라미터 조정
- 조기 종료 (Early Stopping)

### 3. 실시간 적용
- 모델 업데이트 주기 설정
- 성능 모니터링
- 드리프트 감지

## 🔧 설정 가이드

### Supabase 설정
```python
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

### 최적화 설정
```python
SYMBOLS = ['ADAUSDT', 'BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['1m', '5m', '15m']
MODEL_TYPES = ['lightgbm', 'xgboost']
N_TRIALS = 100  # 하이퍼파라미터 최적화 시도 횟수
```

### 성능 임계값
```python
LOOKFORWARD = 5      # 미래 예측 기간
THRESHOLD = 0.002    # 상승 임계값 (0.2%)
MIN_ACCURACY = 0.6   # 최소 정확도
MIN_F1 = 0.5         # 최소 F1 점수
```

## 📞 지원

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.

---

**마지막 업데이트**: 2024년 12월 1일
**버전**: 2.0.0

