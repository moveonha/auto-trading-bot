# 🤖 자동 하이퍼파라미터 튜닝 시스템
# 암호화폐 AI 트레이딩 봇 최적화

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import optuna
import joblib
import json
import warnings
from datetime import datetime
import os
from supabase import create_client

warnings.filterwarnings('ignore')

class AutoTradingBot:
    def __init__(self, supabase_url, supabase_key):
        """자동 트레이딩 봇 초기화"""
        self.supabase = create_client(supabase_url, supabase_key)
        self.models = {}
        self.results = {}
        self.best_params = {}

    def collect_data(self, symbol='ADAUSDT', timeframe='1m', limit=100000):
        """데이터 수집"""
        print(f'🔄 {symbol} {timeframe} 데이터 수집 중...')

        response = self.supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol.upper()).eq('timeframe', timeframe).order('timestamp', desc=True).limit(limit).execute()

        if not response.data:
            raise ValueError(f'데이터를 찾을 수 없습니다: {symbol} {timeframe}')

        df = pd.DataFrame(response.data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('datetime').reset_index(drop=True)

        print(f'✅ 데이터 수집 완료: {len(df):,}개')
        print(f'📅 기간: {df["datetime"].min()} ~ {df["datetime"].max()}')

        return df

    def calculate_features(self, df):
        """고급 기술적 지표 계산"""
        print('🧮 기술적 지표 계산 중...')

        # 기본 이동평균
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)

        # MACD (다양한 설정)
        macd_fast = ta.macd(df['close'], fast=6, slow=13, signal=4)
        macd_slow = ta.macd(df['close'], fast=12, slow=26, signal=9)

        df['macd_fast'] = macd_fast['MACD_6_13_4']
        df['macd_signal_fast'] = macd_fast['MACDs_6_13_4']
        df['macd_hist_fast'] = macd_fast['MACDh_6_13_4']

        df['macd_slow'] = macd_slow['MACD_12_26_9']
        df['macd_signal_slow'] = macd_slow['MACDs_12_26_9']
        df['macd_hist_slow'] = macd_slow['MACDh_12_26_9']

        # RSI (다양한 기간)
        for period in [9, 14, 21]:
            df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)

        # 볼린저 밴드
        bb_short = ta.bbands(df['close'], length=10, std=2)
        bb_long = ta.bbands(df['close'], length=20, std=2)

        df['bb_upper_short'] = bb_short['BBU_10_2.0']
        df['bb_lower_short'] = bb_short['BBL_10_2.0']
        df['bb_width_short'] = (df['bb_upper_short'] - df['bb_lower_short']) / df['close']
        df['bb_position_short'] = (df['close'] - df['bb_lower_short']) / (df['bb_upper_short'] - df['bb_lower_short'])

        df['bb_upper_long'] = bb_long['BBU_20_2.0']
        df['bb_lower_long'] = bb_long['BBL_20_2.0']
        df['bb_width_long'] = (df['bb_upper_long'] - df['bb_lower_long']) / df['close']
        df['bb_position_long'] = (df['close'] - df['bb_lower_long']) / (df['bb_upper_long'] - df['bb_lower_long'])

        # 스토캐스틱
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        # 가격 변화율
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # 거래량 지표
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_ema'] = ta.ema(df['volume'], length=20)

        # 시간 특성
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)

        # 추가 지표
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])

        print('✅ 기술적 지표 계산 완료')
        return df

    def create_target(self, df, lookforward=5, threshold=0.002):
        """목표 변수 생성 (이진 분류)"""
        print(f'🎯 목표 변수 생성 (lookforward={lookforward}, threshold={threshold})')

        # 미래 수익률 계산
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1

        # 이진 분류: 상승(1) vs 하락(0)
        df['target'] = np.where(df['future_return'] > threshold, 1, 0)

        # 클래스 분포 확인
        class_dist = df['target'].value_counts()
        print(f'📊 클래스 분포: {dict(class_dist)}')
        print(f'📈 상승 비율: {df["target"].mean()*100:.1f}%')

        return df

    def prepare_features(self, df):
        """특성 준비 및 전처리"""
        print('🔧 특성 준비 중...')

        # 사용할 특성들
        feature_columns = [
            # 이동평균
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',

            # MACD
            'macd_fast', 'macd_signal_fast', 'macd_hist_fast',
            'macd_slow', 'macd_signal_slow', 'macd_hist_slow',

            # RSI
            'rsi_9', 'rsi_14', 'rsi_21',

            # 볼린저 밴드
            'bb_upper_short', 'bb_lower_short', 'bb_width_short', 'bb_position_short',
            'bb_upper_long', 'bb_lower_long', 'bb_width_long', 'bb_position_long',

            # 스토캐스틱
            'stoch_k', 'stoch_d',

            # 수익률
            'return_1', 'return_3', 'return_5', 'return_10', 'return_20',

            # 거래량
            'volume_sma', 'volume_ratio', 'volume_ema',

            # 시간 특성
            'hour', 'day_of_week', 'is_weekend',
            'is_asia_session', 'is_london_session', 'is_ny_session',

            # 추가 지표
            'atr', 'adx'
        ]

        # 결측값 제거
        all_columns = feature_columns + ['target']
        df_clean = df[all_columns].dropna()

        X = df_clean[feature_columns]
        y = df_clean['target']

        print(f'✅ 특성 준비 완료: {X.shape}')
        return X, y, feature_columns

    def objective_lightgbm(self, trial, X, y):
        """LightGBM 하이퍼파라미터 최적화 목적 함수"""
        # 하이퍼파라미터 범위 정의
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # 교차 검증
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    def objective_xgboost(self, trial, X, y):
        """XGBoost 하이퍼파라미터 최적화 목적 함수"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    def optimize_hyperparameters(self, X, y, model_type='lightgbm', n_trials=100):
        """하이퍼파라미터 최적화"""
        print(f'🔍 {model_type.upper()} 하이퍼파라미터 최적화 시작 (n_trials={n_trials})')

        if model_type == 'lightgbm':
            objective = lambda trial: self.objective_lightgbm(trial, X, y)
        elif model_type == 'xgboost':
            objective = lambda trial: self.objective_xgboost(trial, X, y)
        else:
            raise ValueError(f'지원하지 않는 모델 타입: {model_type}')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        print(f'✅ 최적화 완료')
        print(f'🏆 최고 점수: {study.best_value:.4f}')
        print(f'🔧 최적 파라미터: {study.best_params}')

        return study.best_params, study.best_value

    def train_final_model(self, X, y, best_params, model_type='lightgbm'):
        """최적 파라미터로 최종 모델 훈련"""
        print(f'🚀 최종 {model_type.upper()} 모델 훈련 중...')

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 모델 생성 및 훈련
        if model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**best_params)
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(**best_params)
        else:
            raise ValueError(f'지원하지 않는 모델 타입: {model_type}')

        model.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 성능 지표 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        print('📊 최종 모델 성능:')
        for metric, value in metrics.items():
            print(f'  {metric}: {value:.4f}')

        return model, metrics, (X_test, y_test, y_pred, y_pred_proba)

    def save_model(self, model, metrics, best_params, feature_columns, symbol, timeframe, model_type):
        """모델 및 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'{symbol}_{timeframe}_{model_type}_{timestamp}'

        # 모델 저장
        model_path = f'models/{model_name}.pkl'
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)

        # 결과 저장
        results = {
            'model_name': model_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'best_params': best_params,
            'feature_columns': feature_columns
        }

        results_path = f'results/{model_name}_results.json'
        os.makedirs('results', exist_ok=True)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f'💾 모델 저장 완료: {model_path}')
        print(f'💾 결과 저장 완료: {results_path}')

        return model_name

    def run_optimization(self, symbol='ADAUSDT', timeframe='1m', model_type='lightgbm', n_trials=100):
        """전체 최적화 프로세스 실행"""
        print(f'🚀 {symbol} {timeframe} {model_type.upper()} 최적화 시작')
        print('=' * 60)

        try:
            # 1. 데이터 수집
            df = self.collect_data(symbol, timeframe)

            # 2. 특성 계산
            df = self.calculate_features(df)

            # 3. 목표 변수 생성
            df = self.create_target(df)

            # 4. 특성 준비
            X, y, feature_columns = self.prepare_features(df)

            # 5. 하이퍼파라미터 최적화
            best_params, best_score = self.optimize_hyperparameters(X, y, model_type, n_trials)

            # 6. 최종 모델 훈련
            model, metrics, test_results = self.train_final_model(X, y, best_params, model_type)

            # 7. 모델 저장
            model_name = self.save_model(model, metrics, best_params, feature_columns, symbol, timeframe, model_type)

            print('=' * 60)
            print(f'✅ {symbol} {timeframe} {model_type.upper()} 최적화 완료!')
            print(f'🏆 최고 F1 점수: {best_score:.4f}')
            print(f'📊 최종 정확도: {metrics["accuracy"]:.4f}')

            return model_name, metrics, best_params

        except Exception as e:
            print(f'❌ 오류 발생: {e}')
            return None, None, None

# 사용 예시
if __name__ == "__main__":
    # Supabase 설정
    SUPABASE_URL = "https://your-project.supabase.co"
    SUPABASE_KEY = "your-anon-key"

    # 자동 트레이딩 봇 생성
    bot = AutoTradingBot(SUPABASE_URL, SUPABASE_KEY)

    # 최적화 실행
    symbols = ['ADAUSDT', 'BTCUSDT', 'ETHUSDT']
    timeframes = ['1m', '5m', '15m']
    model_types = ['lightgbm', 'xgboost']

    for symbol in symbols:
        for timeframe in timeframes:
            for model_type in model_types:
                print(f'\n🔄 {symbol} {timeframe} {model_type.upper()} 최적화 중...')
                model_name, metrics, best_params = bot.run_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    n_trials=50  # 빠른 테스트용
                )

                if model_name:
                    print(f'✅ {model_name} 완료!')
                else:
                    print(f'❌ {symbol} {timeframe} {model_type} 실패')

