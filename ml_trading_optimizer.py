import os
import json
import asyncio
import websockets
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from supabase import create_client
import pandas_ta as ta
import time
import logging
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trading.log'),
        logging.StreamHandler()
    ]
)

class MLTradingOptimizer:
    def __init__(self):
        self.load_env_file()
        self.supabase = self.get_supabase_client()
        self.websocket = None
        self.is_running = False
        
        # 실시간 데이터 저장소
        self.realtime_data = {}
        self.signal_history = []
        self.trade_history = []
        
        # 수집할 심볼과 타임프레임
        self.symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt']
        self.timeframes = ['1m', '5m', '15m', '1h']
        
        # ML 모델들
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        
        # 최적화된 파라미터
        self.optimized_params = {
            'signal_threshold': 2.5,
            'position_size': 0.02,  # 계좌의 2%
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'max_holding_time': 24,  # 시간
            'min_confidence': 0.7
        }
        
        # 백테스팅 결과
        self.backtest_results = {}
        
        # 모델 저장 경로
        self.model_path = Path('models')
        self.model_path.mkdir(exist_ok=True)
        
    def load_env_file(self):
        """환경변수 로드"""
        config_file = Path('.env')
        if not config_file.exists():
            raise FileNotFoundError(".env 파일을 찾을 수 없습니다.")
        
        with open(config_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    def get_supabase_client(self):
        """Supabase 클라이언트 생성"""
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        if not url or not key:
            raise ValueError("Supabase URL 또는 Key가 설정되지 않았습니다.")
        return create_client(url, key)
    
    def get_historical_data(self, symbol, timeframe, limit=1000):
        """과거 데이터 수집"""
        try:
            # Supabase에서 데이터 가져오기
            response = self.supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol.upper()).eq('timeframe', timeframe).order('timestamp', desc=True).limit(limit).execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime')
                return df
            else:
                logging.warning(f"데이터가 없습니다: {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logging.error(f"❌ 과거 데이터 수집 오류: {str(e)}")
            return None
    
    def calculate_advanced_features(self, df):
        """고급 기술적 지표 계산"""
        try:
            # 기본 지표들
            df['sma_5'] = ta.sma(df['close'], length=5)
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            
            # MACD
            macd = ta.macd(df['close'], fast=6, slow=13, signal=4)
            df['macd'] = macd['MACD_6_13_4']
            df['macd_signal'] = macd['MACDs_6_13_4']
            df['macd_histogram'] = macd['MACDh_6_13_4']
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=9)
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            
            # 볼린저 밴드
            bb = ta.bbands(df['close'], length=10, std=2)
            df['bb_upper'] = bb['BBU_10_2.0']
            df['bb_lower'] = bb['BBL_10_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 스토캐스틱
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3)
            df['stoch_k'] = stoch['STOCHk_5_3_3']
            df['stoch_d'] = stoch['STOCHd_5_3_3']
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=7)
            df['atr_ratio'] = df['atr'] / df['close']
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'], length=7)
            df['adx'] = adx['ADX_7']
            df['di_plus'] = adx['DMP_7']
            df['di_minus'] = adx['DMN_7']
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=9)
            
            # Momentum
            df['momentum'] = ta.mom(df['close'], length=10)
            df['momentum_5'] = ta.mom(df['close'], length=5)
            
            # OBV (On Balance Volume)
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['obv_sma'] = ta.sma(df['obv'], length=20)
            
            # CCI (Commodity Channel Index)
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=9)
            
            # Parabolic SAR
            df['psar'] = ta.psar(df['high'], df['low'], df['close'])
            
            # 가격 변화율
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # 볼륨 지표
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 변동성 지표
            df['volatility'] = df['close'].rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['close']
            
            # 추세 지표
            df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['close']
            df['trend_direction'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
            
            # 크로스오버 지표
            df['ema_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, 0)
            df['ema_cross_change'] = df['ema_cross'].diff()
            
            df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
            df['macd_cross_change'] = df['macd_cross'].diff()
            
            # 시간 기반 특성
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            return df
            
        except Exception as e:
            logging.error(f"❌ 고급 지표 계산 오류: {str(e)}")
            return df
    
    def create_target_variable(self, df, lookforward=5):
        """목표 변수 생성 (미래 수익률 기반)"""
        try:
            # 미래 가격 변화율 계산
            df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
            
            # 목표 변수 생성
            df['target'] = np.where(df['future_return'] > 0.01, 1,  # 1% 이상 상승 시 LONG
                          np.where(df['future_return'] < -0.01, -1, 0))  # 1% 이상 하락 시 SHORT
            
            # 신호 강도 계산
            df['signal_strength'] = abs(df['future_return']) * 100
            
            return df
            
        except Exception as e:
            logging.error(f"❌ 목표 변수 생성 오류: {str(e)}")
            return df
    
    def prepare_features(self, df):
        """특성 준비"""
        try:
            # 사용할 특성 컬럼들
            feature_columns = [
                'sma_5', 'sma_10', 'sma_20', 'ema_9', 'ema_21',
                'macd', 'macd_signal', 'macd_histogram',
                'rsi', 'rsi_14',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'stoch_k', 'stoch_d',
                'atr', 'atr_ratio',
                'adx', 'di_plus', 'di_minus',
                'williams_r', 'momentum', 'momentum_5',
                'obv', 'obv_sma', 'cci', 'psar',
                'price_change', 'price_change_5', 'price_change_10',
                'volume_ratio', 'volatility_ratio',
                'trend_strength', 'trend_direction',
                'ema_cross', 'ema_cross_change',
                'macd_cross', 'macd_cross_change',
                'hour', 'day_of_week', 'is_weekend'
            ]
            
            # NaN 값 처리
            df = df.dropna()
            
            # 특성과 타겟 분리
            X = df[feature_columns]
            y = df['target']
            
            return X, y, feature_columns
            
        except Exception as e:
            logging.error(f"❌ 특성 준비 오류: {str(e)}")
            return None, None, None
    
    def train_models(self, symbol, timeframe):
        """모델 훈련"""
        try:
            logging.info(f"🔄 {symbol} {timeframe} 모델 훈련 시작...")
            
            # 데이터 수집
            df = self.get_historical_data(symbol, timeframe, limit=2000)
            if df is None or len(df) < 500:
                logging.warning(f"충분한 데이터가 없습니다: {symbol} {timeframe}")
                return False
            
            # 고급 지표 계산
            df = self.calculate_advanced_features(df)
            
            # 목표 변수 생성
            df = self.create_target_variable(df)
            
            # 특성 준비
            X, y, feature_columns = self.prepare_features(df)
            if X is None:
                return False
            
            self.feature_columns = feature_columns
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # 특성 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 라벨 인코딩
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # 모델들 정의
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # 모델 훈련 및 평가
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                logging.info(f"훈련 중: {name}")
                
                # 훈련
                model.fit(X_train_scaled, y_train_encoded)
                
                # 예측
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test_encoded, y_pred)
                
                logging.info(f"{name} 정확도: {accuracy:.4f}")
                
                # 교차 검증
                cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5)
                cv_mean = cv_scores.mean()
                logging.info(f"{name} 교차 검증 평균: {cv_mean:.4f}")
                
                # 최고 모델 선택
                if cv_mean > best_score:
                    best_score = cv_mean
                    best_model = model
                    best_model_name = name
            
            # 최고 모델 저장
            key = f"{symbol}_{timeframe}"
            self.models[key] = best_model
            self.scalers[key] = scaler
            self.label_encoders[key] = label_encoder
            
            # 모델 파일 저장
            model_file = self.model_path / f"{key}_model.pkl"
            scaler_file = self.model_path / f"{key}_scaler.pkl"
            encoder_file = self.model_path / f"{key}_encoder.pkl"
            
            joblib.dump(best_model, model_file)
            joblib.dump(scaler, scaler_file)
            joblib.dump(label_encoder, encoder_file)
            
            logging.info(f"✅ {symbol} {timeframe} 모델 훈련 완료! 최고 모델: {best_model_name} (정확도: {best_score:.4f})")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ 모델 훈련 오류: {str(e)}")
            return False
    
    def train_lstm_model(self, symbol, timeframe):
        """LSTM 모델 훈련"""
        try:
            logging.info(f"🔄 {symbol} {timeframe} LSTM 모델 훈련 시작...")
            
            # 데이터 수집
            df = self.get_historical_data(symbol, timeframe, limit=3000)
            if df is None or len(df) < 1000:
                return False
            
            # 고급 지표 계산
            df = self.calculate_advanced_features(df)
            df = self.create_target_variable(df)
            
            # 특성 준비
            X, y, feature_columns = self.prepare_features(df)
            if X is None:
                return False
            
            # 시계열 데이터로 변환 (시퀀스 길이: 60)
            sequence_length = 60
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X.iloc[i-sequence_length:i].values)
                y_sequences.append(y.iloc[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # 데이터 분할
            split_idx = int(len(X_sequences) * 0.8)
            X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # 특성 스케일링
            scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_test_scaled = scaler.transform(X_test_reshaped)
            
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # 라벨 인코딩
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # 원-핫 인코딩
            y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=3)
            y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=3)
            
            # LSTM 모델 구축
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(sequence_length, len(feature_columns))),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 콜백 설정
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # 모델 훈련
            history = model.fit(
                X_train_scaled, y_train_onehot,
                validation_data=(X_test_scaled, y_test_onehot),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # 모델 평가
            y_pred = model.predict(X_test_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test_encoded, y_pred_classes)
            
            logging.info(f"LSTM 정확도: {accuracy:.4f}")
            
            # 모델 저장
            key = f"{symbol}_{timeframe}_lstm"
            self.models[key] = model
            self.scalers[key] = scaler
            self.label_encoders[key] = label_encoder
            
            model.save(self.model_path / f"{key}_model.h5")
            joblib.dump(scaler, self.model_path / f"{key}_scaler.pkl")
            joblib.dump(label_encoder, self.model_path / f"{key}_encoder.pkl")
            
            logging.info(f"✅ {symbol} {timeframe} LSTM 모델 훈련 완료!")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ LSTM 모델 훈련 오류: {str(e)}")
            return False
    
    def predict_signal(self, symbol, timeframe, current_data):
        """실시간 신호 예측"""
        try:
            key = f"{symbol}_{timeframe}"
            
            if key not in self.models:
                logging.warning(f"모델이 없습니다: {key}")
                return None
            
            model = self.models[key]
            scaler = self.scalers[key]
            label_encoder = self.label_encoders[key]
            
            # 특성 계산
            features = self.calculate_advanced_features(current_data)
            if features is None or len(features) < 50:
                return None
            
            # 최신 데이터 추출
            latest_features = features[self.feature_columns].iloc[-1:].values
            
            # 스케일링
            scaled_features = scaler.transform(latest_features)
            
            # 예측
            if 'lstm' in key:
                # LSTM 모델의 경우 시퀀스 데이터 필요
                return None  # 실시간 LSTM 예측은 복잡하므로 일단 제외
            else:
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
                
                # 신호 생성
                signal = {
                    'symbol': symbol.upper(),
                    'timeframe': timeframe,
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'confidence': max(probabilities),
                    'action': label_encoder.inverse_transform([prediction])[0],
                    'datetime': datetime.now()
                }
                
                return signal
                
        except Exception as e:
            logging.error(f"❌ 신호 예측 오류: {str(e)}")
            return None
    
    def optimize_parameters(self, symbol, timeframe):
        """파라미터 최적화"""
        try:
            logging.info(f"🔄 {symbol} {timeframe} 파라미터 최적화 시작...")
            
            # 백테스팅을 통한 파라미터 최적화
            best_params = self.optimized_params.copy()
            best_sharpe = -999
            
            # 파라미터 그리드
            param_grid = {
                'signal_threshold': [1.5, 2.0, 2.5, 3.0, 3.5],
                'stop_loss_atr': [1.0, 1.5, 2.0, 2.5],
                'take_profit_atr': [2.0, 3.0, 4.0, 5.0],
                'min_confidence': [0.6, 0.7, 0.8, 0.9]
            }
            
            # 그리드 서치
            for threshold in param_grid['signal_threshold']:
                for stop_loss in param_grid['stop_loss_atr']:
                    for take_profit in param_grid['take_profit_atr']:
                        for confidence in param_grid['min_confidence']:
                            
                            params = {
                                'signal_threshold': threshold,
                                'stop_loss_atr': stop_loss,
                                'take_profit_atr': take_profit,
                                'min_confidence': confidence
                            }
                            
                            # 백테스팅 실행
                            results = self.run_backtest(symbol, timeframe, params)
                            
                            if results and results['sharpe_ratio'] > best_sharpe:
                                best_sharpe = results['sharpe_ratio']
                                best_params.update(params)
                                logging.info(f"새로운 최고 파라미터 발견! Sharpe: {best_sharpe:.4f}")
            
            self.optimized_params = best_params
            logging.info(f"✅ 파라미터 최적화 완료! 최고 Sharpe: {best_sharpe:.4f}")
            
            return best_params
            
        except Exception as e:
            logging.error(f"❌ 파라미터 최적화 오류: {str(e)}")
            return self.optimized_params
    
    def run_backtest(self, symbol, timeframe, params=None):
        """백테스팅 실행"""
        try:
            if params is None:
                params = self.optimized_params
            
            # 과거 데이터 수집
            df = self.get_historical_data(symbol, timeframe, limit=1000)
            if df is None:
                return None
            
            # 고급 지표 계산
            df = self.calculate_advanced_features(df)
            
            # 백테스팅 결과
            trades = []
            position = None
            entry_price = 0
            entry_time = None
            
            for i in range(100, len(df)):
                current_data = df.iloc[:i+1]
                
                # 신호 예측
                signal = self.predict_signal(symbol, timeframe, current_data)
                
                if signal is None:
                    continue
                
                current_price = df.iloc[i]['close']
                current_time = df.iloc[i]['datetime']
                
                # 포지션 진입 조건
                if position is None and signal['confidence'] >= params['min_confidence']:
                    if signal['action'] in [1, -1]:  # LONG 또는 SHORT
                        position = signal['action']
                        entry_price = current_price
                        entry_time = current_time
                        
                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'position': 'LONG' if position == 1 else 'SHORT',
                            'confidence': signal['confidence']
                        })
                
                # 포지션 청산 조건
                elif position is not None:
                    # ATR 계산
                    atr = df.iloc[i]['atr']
                    
                    # 손절/익절 계산
                    if position == 1:  # LONG
                        stop_loss = entry_price - (atr * params['stop_loss_atr'])
                        take_profit = entry_price + (atr * params['take_profit_atr'])
                        
                        if current_price <= stop_loss or current_price >= take_profit:
                            # 청산
                            exit_price = current_price
                            pnl = (exit_price - entry_price) / entry_price
                            
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'exit_reason': 'stop_loss' if current_price <= stop_loss else 'take_profit'
                            })
                            
                            position = None
                    
                    elif position == -1:  # SHORT
                        stop_loss = entry_price + (atr * params['stop_loss_atr'])
                        take_profit = entry_price - (atr * params['take_profit_atr'])
                        
                        if current_price >= stop_loss or current_price <= take_profit:
                            # 청산
                            exit_price = current_price
                            pnl = (entry_price - exit_price) / entry_price
                            
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'exit_reason': 'stop_loss' if current_price >= stop_loss else 'take_profit'
                            })
                            
                            position = None
                
                # 최대 보유 시간 체크
                if position is not None and entry_time:
                    holding_time = (current_time - entry_time).total_seconds() / 3600
                    if holding_time > params.get('max_holding_time', 24):
                        # 강제 청산
                        exit_price = current_price
                        if position == 1:
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
                        trades[-1].update({
                            'exit_time': current_time,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_reason': 'timeout'
                        })
                        
                        position = None
            
            # 결과 계산
            if not trades:
                return None
            
            completed_trades = [t for t in trades if 'exit_price' in t]
            
            if not completed_trades:
                return None
            
            total_return = sum(t['pnl'] for t in completed_trades)
            win_trades = [t for t in completed_trades if t['pnl'] > 0]
            win_rate = len(win_trades) / len(completed_trades)
            
            returns = [t['pnl'] for t in completed_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            results = {
                'total_trades': len(completed_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'avg_return': np.mean(returns),
                'max_drawdown': min(returns) if returns else 0,
                'trades': completed_trades
            }
            
            return results
            
        except Exception as e:
            logging.error(f"❌ 백테스팅 오류: {str(e)}")
            return None
    
    def train_all_models(self):
        """모든 심볼과 타임프레임에 대해 모델 훈련"""
        logging.info("🚀 전체 모델 훈련 시작...")
        
        success_count = 0
        total_count = len(self.symbols) * len(self.timeframes)
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    # 기본 모델 훈련
                    if self.train_models(symbol, timeframe):
                        success_count += 1
                    
                    # LSTM 모델 훈련
                    if self.train_lstm_model(symbol, timeframe):
                        success_count += 1
                    
                    # 파라미터 최적화
                    self.optimize_parameters(symbol, timeframe)
                    
                except Exception as e:
                    logging.error(f"❌ {symbol} {timeframe} 모델 훈련 실패: {str(e)}")
        
        logging.info(f"✅ 모델 훈련 완료! 성공: {success_count}/{total_count * 2}")
    
    def generate_ml_signals(self):
        """ML 기반 실시간 신호 생성"""
        logging.info("🤖 ML 기반 실시간 신호 생성 시작...")
        
        while self.is_running:
            try:
                # 각 심볼과 타임프레임에 대해 신호 생성
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        key = f"{symbol}_{timeframe}"
                        
                        if key in self.realtime_data and len(self.realtime_data[key]) > 50:
                            # 최신 데이터로 신호 예측
                            current_data = pd.DataFrame(self.realtime_data[key])
                            signal = self.predict_signal(symbol, timeframe, current_data)
                            
                            if signal and signal['confidence'] >= self.optimized_params['min_confidence']:
                                # 신호 처리
                                self.process_ml_signal(signal)
                
                time.sleep(10)  # 10초마다 체크
                
            except Exception as e:
                logging.error(f"❌ ML 신호 생성 오류: {str(e)}")
                time.sleep(30)
    
    def process_ml_signal(self, signal):
        """ML 신호 처리"""
        try:
            action_emoji = "🟢" if signal['action'] == 1 else "🔴" if signal['action'] == -1 else "🟡"
            confidence_emoji = "🔥" if signal['confidence'] >= 0.9 else "⚡" if signal['confidence'] >= 0.8 else "💤"
            
            logging.info(f"🤖 {action_emoji} ML {signal['action']} {confidence_emoji} - {signal['symbol']} {signal['timeframe']}")
            logging.info(f"   💰 신뢰도: {signal['confidence']:.4f}")
            logging.info(f"   📊 확률: {signal['probabilities']}")
            
            # 강한 신호인 경우 즉시 알림
            if signal['confidence'] >= 0.9:
                logging.warning(f"🔥 강한 ML 신호 감지! {signal['symbol']} {signal['timeframe']} {signal['action']}")
                
                # 여기에 자동매매 로직 추가 가능
                # await self.execute_ml_trade(signal)
            
        except Exception as e:
            logging.error(f"❌ ML 신호 처리 오류: {str(e)}")
    
    def start(self):
        """ML 트레이딩 시스템 시작"""
        self.is_running = True
        logging.info("🤖 ML 기반 트레이딩 시스템 시작...")
        
        try:
            # 모델 훈련
            self.train_all_models()
            
            # 실시간 신호 생성 시작
            self.generate_ml_signals()
            
        except KeyboardInterrupt:
            logging.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            logging.error(f"❌ 실행 오류: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """ML 트레이딩 시스템 중지"""
        self.is_running = False
        logging.info("ML 기반 트레이딩 시스템이 중지되었습니다.")

def main():
    """메인 실행 함수"""
    print("🤖 ML 기반 암호화폐 트레이딩 최적화 시스템")
    print("=" * 60)
    
    try:
        ml_optimizer = MLTradingOptimizer()
        ml_optimizer.start()
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
