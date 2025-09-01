# 🤖 AI 자동 트레이딩 봇 시스템

암호화폐 시장을 위한 머신러닝 기반 자동 트레이딩 시스템입니다.

## 🚀 주요 기능

- **🤖 자동 하이퍼파라미터 튜닝**: Optuna를 사용한 최적화
- **📊 다중 모델 지원**: LightGBM, XGBoost, LSTM
- **🔄 실시간 데이터 수집**: Supabase 연동
- **🎯 트레이딩 신호 생성**: 실시간 매매 신호
- **📈 성능 모니터링**: 자동 성능 평가 및 시각화
- **💾 자동 저장**: 모델 및 결과 자동 저장

## 📁 프로젝트 구조

```
crypto-currency-datacollect/
├── README.md                           # 프로젝트 개요
├── requirements.txt                    # 필수 패키지
├── .gitignore                         # Git 제외 파일
├── create_tables.sql                  # 데이터베이스 스키마
├── colab_models/                      # 🆕 정리된 AI 모델 시스템
│   ├── README.md                      # AI 시스템 가이드
│   ├── USAGE_GUIDE.md                 # 상세 사용 가이드
│   ├── requirements.txt               # AI 시스템 패키지
│   ├── run_auto_training.py          # 메인 실행 스크립트
│   ├── auto_hyperparameter_tuner.py  # 자동 하이퍼파라미터 튜닝
│   ├── model_evaluator.py            # 모델 성능 평가
│   ├── trading_signal_generator.py   # 트레이딩 신호 생성
│   ├── colab_auto_training_template.ipynb # Colab 템플릿
│   └── legacy/                       # 기존 노트북들 (백업)
├── main.py                           # 메인 실행 파일
├── collect_3years_data.py            # 3년 데이터 수집
├── check_data.py                     # 데이터 확인
├── trading_signals.py                # 트레이딩 신호
├── realtime_trading_signals.py       # 실시간 트레이딩 신호
├── futures_trading_signals.py        # 선물 트레이딩 신호
├── realtime_data_collector.py        # 실시간 데이터 수집
├── ml_trading_optimizer.py           # ML 트레이딩 최적화
├── continuous_optimization_system.py # 연속 최적화 시스템
├── colab_training_system.py          # Colab 훈련 시스템
├── colab_api_controller.py           # Colab API 컨트롤러
├── real_colab_monitor.py             # Colab 모니터링
├── training_progress_monitor.py      # 훈련 진행 모니터링
├── start_training.py                 # 훈련 시작
└── test_main.py                      # 테스트 파일
```

## 🆕 새로운 AI 시스템 (colab_models/)

### 주요 개선사항
- ✅ **자동화**: 수동 작업을 자동화하여 시간 절약
- ✅ **최적화**: Optuna를 사용한 하이퍼파라미터 최적화
- ✅ **실용성**: 실제 트레이딩 신호 생성 기능
- ✅ **시각화**: Plotly를 사용한 인터랙티브 차트
- ✅ **관리**: 체계적인 모델 및 결과 저장
- ✅ **문서화**: 상세한 사용 가이드

### 사용 방법

#### 1. Colab에서 사용하기
```python
# colab_auto_training_template.ipynb 업로드 후 실행
SUPABASE_URL = "your-url"
SUPABASE_KEY = "your-key"
```

#### 2. 로컬에서 사용하기
```bash
python colab_models/run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --mode all
```

## 🔧 설치 및 설정

### 1. 필수 요구사항
- Python 3.8 이상
- Supabase 계정 및 프로젝트
- 충분한 저장 공간 (최소 1GB)

### 2. 패키지 설치
```bash
# 전체 시스템
pip install -r requirements.txt

# AI 시스템만
pip install -r colab_models/requirements.txt
```

### 3. 데이터베이스 설정
```sql
-- create_tables.sql 실행
-- Supabase에서 테이블 생성
```

## 🚀 빠른 시작

### 1. 데이터 수집
```bash
python collect_3years_data.py
```

### 2. AI 모델 훈련
```bash
python colab_models/run_auto_training.py \
    --supabase-url "your-url" \
    --supabase-key "your-key" \
    --symbols ADAUSDT BTCUSDT \
    --timeframes 1m 5m \
    --model-types lightgbm xgboost \
    --n-trials 50
```

### 3. 트레이딩 신호 생성
```bash
python colab_models/trading_signal_generator.py
```

## 📊 성능 지표

- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (가중 평균)
- **Recall**: 재현율 (가중 평균)
- **F1-Score**: F1 점수 (가중 평균)
- **ROC-AUC**: ROC 곡선 아래 면적

## 🎯 트레이딩 신호

- **STRONG_BUY**: 강한 매수 신호 (신호 강도 ≥ 0.6)
- **BUY**: 매수 신호 (신호 강도 ≥ 0.4)
- **WEAK_BUY**: 약한 매수 신호 (신호 강도 ≥ 0.2)
- **NEUTRAL**: 중립 (신호 강도 ≥ 0.1)
- **SELL**: 매도 신호 (신호 강도 < 0.1)

## 🔍 문제 해결

### 일반적인 문제들
1. **데이터 연결 오류**: Supabase 설정 확인
2. **메모리 부족**: 데이터 수집량 줄이기
3. **모델 저장 실패**: 디렉토리 권한 확인
4. **Colab 연결 오류**: 확장 프로그램 재설치

### 지원
- 📚 [사용 가이드](colab_models/USAGE_GUIDE.md)
- 📖 [AI 시스템 가이드](colab_models/README.md)
- 🔧 [문제 해결](colab_models/USAGE_GUIDE.md#문제-해결)

## 📈 업데이트 내역

### v2.0.0 (2024-12-01)
- 🆕 AI 자동화 시스템 구축
- 🔧 기존 노트북 정리 및 백업
- 📊 성능 평가 도구 추가
- 🎯 트레이딩 신호 생성기 추가
- 📚 상세한 문서화

### v1.0.0 (이전)
- 기본 데이터 수집 시스템
- 기본 트레이딩 신호 생성
- Colab 연동 시스템

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**개발자**: AI 트레이딩 봇 개발팀  
**마지막 업데이트**: 2024년 12월 1일  
**버전**: 2.0.0
