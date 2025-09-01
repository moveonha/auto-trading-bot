import os
import json
import asyncio
import time
import logging
import schedule
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from supabase import create_client
import pandas_ta as ta
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_optimization.log'),
        logging.StreamHandler()
    ]
)

class ContinuousOptimizationSystem:
    def __init__(self):
        self.load_env_file()
        self.supabase = self.get_supabase_client()
        self.is_running = False
        
        # 최적화 설정
        self.optimization_config = {
            'symbols': ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt'],
            'timeframes': ['1m', '5m', '15m', '1h'],
            'models': ['random_forest', 'xgboost', 'lightgbm'],
            'data_limit': 3000,
            'retrain_interval_hours': 24,  # 24시간마다 재훈련
            'optimization_interval_hours': 6,  # 6시간마다 파라미터 최적화
            'backtest_days': 30,  # 30일 백테스팅
            'min_accuracy_threshold': 0.75,  # 최소 정확도 임계값
            'min_sharpe_threshold': 1.0,  # 최소 Sharpe Ratio 임계값
            'max_drawdown_threshold': -0.1  # 최대 손실 임계값
        }
        
        # 모델 저장소
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        
        # 최적화된 파라미터
        self.optimized_params = {
            'signal_threshold': 2.5,
            'position_size': 0.02,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'max_holding_time': 24,
            'min_confidence': 0.7
        }
        
        # 성능 추적
        self.performance_history = {}
        self.optimization_history = []
        
        # 모델 저장 경로
        self.model_path = Path('optimized_models')
        self.model_path.mkdir(exist_ok=True)
        
        # 성능 데이터베이스
        self.performance_db = Path('performance_data')
        self.performance_db.mkdir(exist_ok=True)
        
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
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['obv_sma'] = ta.sma(df['obv'], length=20)
            
            # CCI
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
        """목표 변수 생성"""
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
    
    def train_model_with_hyperparameter_tuning(self, symbol, timeframe, model_type):
        """하이퍼파라미터 튜닝과 함께 모델 훈련"""
        try:
            logging.info(f"🔄 {symbol} {timeframe} {model_type} 하이퍼파라미터 튜닝 시작...")
            
            # 데이터 수집
            df = self.get_historical_data(symbol, timeframe, limit=self.optimization_config['data_limit'])
            if df is None or len(df) < 500:
                return False
            
            # 고급 지표 계산
            df = self.calculate_advanced_features(df)
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
            
            # 하이퍼파라미터 그리드 정의
            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                base_model = RandomForestClassifier(random_state=42)
            elif model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
                base_model = xgb.XGBClassifier(random_state=42)
            elif model_type == 'lightgbm':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 62, 127]
                }
                base_model = lgb.LGBMClassifier(random_state=42)
            else:
                return False
            
            # 그리드 서치로 최적 하이퍼파라미터 찾기
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train_encoded)
            
            # 최적 모델
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # 테스트 세트에서 평가
            y_pred = best_model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # 교차 검증
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train_encoded, cv=5)
            
            logging.info(f"✅ {model_type} 하이퍼파라미터 튜닝 완료!")
            logging.info(f"📊 최적 파라미터: {best_params}")
            logging.info(f"📊 CV 평균 정확도: {best_score:.4f}")
            logging.info(f"📊 테스트 정확도: {test_accuracy:.4f}")
            logging.info(f"📊 CV 표준편차: {cv_scores.std():.4f}")
            
            # 모델 저장
            key = f"{symbol}_{timeframe}_{model_type}"
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
            
            # 성능 기록
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': model_type,
                'best_params': best_params,
                'cv_accuracy': best_score,
                'test_accuracy': test_accuracy,
                'cv_std': cv_scores.std(),
                'best_score': best_score
            }
            
            self.save_performance_record(performance_record)
            
            return True
            
        except Exception as e:
            logging.error(f"❌ 하이퍼파라미터 튜닝 오류: {str(e)}")
            return False
    
    def run_comprehensive_backtest(self, symbol, timeframe, params=None):
        """종합 백테스팅 실행"""
        try:
            if params is None:
                params = self.optimized_params
            
            # 과거 데이터 수집 (더 긴 기간)
            days_back = self.optimization_config['backtest_days']
            limit = days_back * 24 * 60  # 1분봉 기준
            
            df = self.get_historical_data(symbol, timeframe, limit=limit)
            if df is None:
                return None
            
            # 고급 지표 계산
            df = self.calculate_advanced_features(df)
            
            # 백테스팅 결과
            trades = []
            position = None
            entry_price = 0
            entry_time = None
            account_balance = 10000  # 초기 자본
            current_balance = account_balance
            
            for i in range(100, len(df)):
                current_data = df.iloc[:i+1]
                
                # 신호 예측 (여러 모델의 앙상블)
                ensemble_signal = self.get_ensemble_signal(symbol, timeframe, current_data)
                
                if ensemble_signal is None:
                    continue
                
                current_price = df.iloc[i]['close']
                current_time = df.iloc[i]['datetime']
                
                # 포지션 진입 조건
                if position is None and ensemble_signal['confidence'] >= params['min_confidence']:
                    if ensemble_signal['action'] in [1, -1]:  # LONG 또는 SHORT
                        position = ensemble_signal['action']
                        entry_price = current_price
                        entry_time = current_time
                        
                        # 포지션 크기 계산
                        position_size = current_balance * params['position_size']
                        quantity = position_size / current_price
                        
                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'position': 'LONG' if position == 1 else 'SHORT',
                            'confidence': ensemble_signal['confidence'],
                            'quantity': quantity,
                            'position_size': position_size
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
                            
                            # 수익/손실 계산
                            trade_pnl = trades[-1]['position_size'] * pnl
                            current_balance += trade_pnl
                            
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'trade_pnl': trade_pnl,
                                'current_balance': current_balance,
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
                            
                            # 수익/손실 계산
                            trade_pnl = trades[-1]['position_size'] * pnl
                            current_balance += trade_pnl
                            
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'trade_pnl': trade_pnl,
                                'current_balance': current_balance,
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
                        
                        trade_pnl = trades[-1]['position_size'] * pnl
                        current_balance += trade_pnl
                        
                        trades[-1].update({
                            'exit_time': current_time,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'trade_pnl': trade_pnl,
                            'current_balance': current_balance,
                            'exit_reason': 'timeout'
                        })
                        
                        position = None
            
            # 결과 계산
            if not trades:
                return None
            
            completed_trades = [t for t in trades if 'exit_price' in t]
            
            if not completed_trades:
                return None
            
            # 기본 통계
            total_return = (current_balance - account_balance) / account_balance
            win_trades = [t for t in completed_trades if t['pnl'] > 0]
            win_rate = len(win_trades) / len(completed_trades)
            
            returns = [t['pnl'] for t in completed_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # 최대 낙폭 계산
            balance_history = [t['current_balance'] for t in completed_trades]
            peak = account_balance
            max_drawdown = 0
            
            for balance in balance_history:
                if balance > peak:
                    peak = balance
                drawdown = (balance - peak) / peak
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
            
            # 추가 지표
            avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t['pnl'] for t in completed_trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in completed_trades) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            results = {
                'total_trades': len(completed_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'final_balance': current_balance,
                'initial_balance': account_balance,
                'trades': completed_trades,
                'params': params
            }
            
            return results
            
        except Exception as e:
            logging.error(f"❌ 종합 백테스팅 오류: {str(e)}")
            return None
    
    def get_ensemble_signal(self, symbol, timeframe, current_data):
        """앙상블 신호 생성"""
        try:
            signals = []
            weights = []
            
            # 각 모델에서 신호 가져오기
            for model_type in self.optimization_config['models']:
                key = f"{symbol}_{timeframe}_{model_type}"
                
                if key in self.models:
                    signal = self.predict_signal(symbol, timeframe, current_data, model_type)
                    if signal:
                        signals.append(signal)
                        # 모델 성능에 따른 가중치
                        weight = self.get_model_weight(key)
                        weights.append(weight)
            
            if not signals:
                return None
            
            # 가중 평균으로 앙상블 신호 생성
            ensemble_action = 0
            ensemble_confidence = 0
            total_weight = sum(weights)
            
            for signal, weight in zip(signals, weights):
                ensemble_action += signal['action'] * (weight / total_weight)
                ensemble_confidence += signal['confidence'] * (weight / total_weight)
            
            # 앙상블 액션 결정
            if ensemble_action > 0.5:
                final_action = 1
            elif ensemble_action < -0.5:
                final_action = -1
            else:
                final_action = 0
            
            return {
                'symbol': symbol.upper(),
                'timeframe': timeframe,
                'action': final_action,
                'confidence': ensemble_confidence,
                'ensemble_action': ensemble_action,
                'datetime': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"❌ 앙상블 신호 생성 오류: {str(e)}")
            return None
    
    def predict_signal(self, symbol, timeframe, current_data, model_type):
        """개별 모델 신호 예측"""
        try:
            key = f"{symbol}_{timeframe}_{model_type}"
            
            if key not in self.models:
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
    
    def get_model_weight(self, model_key):
        """모델 가중치 계산"""
        try:
            # 성능 기록에서 모델 가중치 계산
            performance_file = self.performance_db / f"{model_key}_performance.json"
            
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
                
                # 최근 성능 기반 가중치
                recent_performance = performance_data.get('recent_accuracy', 0.5)
                return recent_performance
            else:
                return 1.0  # 기본 가중치
                
        except Exception as e:
            logging.error(f"❌ 모델 가중치 계산 오류: {str(e)}")
            return 1.0
    
    def continuous_parameter_optimization(self):
        """지속적인 파라미터 최적화"""
        try:
            logging.info("🔄 지속적인 파라미터 최적화 시작...")
            
            best_overall_params = self.optimized_params.copy()
            best_overall_sharpe = -999
            
            # 파라미터 그리드 (더 세밀하게)
            param_grid = {
                'signal_threshold': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                'stop_loss_atr': [1.0, 1.5, 2.0, 2.5, 3.0],
                'take_profit_atr': [2.0, 3.0, 4.0, 5.0, 6.0],
                'min_confidence': [0.6, 0.7, 0.8, 0.85, 0.9],
                'position_size': [0.01, 0.02, 0.03, 0.05]
            }
            
            total_combinations = len(param_grid['signal_threshold']) * len(param_grid['stop_loss_atr']) * len(param_grid['take_profit_atr']) * len(param_grid['min_confidence']) * len(param_grid['position_size'])
            
            logging.info(f"📊 총 {total_combinations}개 파라미터 조합 테스트 중...")
            
            combination_count = 0
            
            for threshold in param_grid['signal_threshold']:
                for stop_loss in param_grid['stop_loss_atr']:
                    for take_profit in param_grid['take_profit_atr']:
                        for confidence in param_grid['min_confidence']:
                            for position_size in param_grid['position_size']:
                                
                                combination_count += 1
                                logging.info(f"🔄 진행률: {combination_count}/{total_combinations}")
                                
                                params = {
                                    'signal_threshold': threshold,
                                    'stop_loss_atr': stop_loss,
                                    'take_profit_atr': take_profit,
                                    'min_confidence': confidence,
                                    'position_size': position_size
                                }
                                
                                # 모든 심볼과 타임프레임에 대해 백테스팅
                                total_sharpe = 0
                                valid_results = 0
                                
                                for symbol in self.optimization_config['symbols']:
                                    for timeframe in self.optimization_config['timeframes']:
                                        results = self.run_comprehensive_backtest(symbol, timeframe, params)
                                        
                                        if results and results['sharpe_ratio'] > 0:
                                            total_sharpe += results['sharpe_ratio']
                                            valid_results += 1
                                
                                # 평균 Sharpe Ratio 계산
                                if valid_results > 0:
                                    avg_sharpe = total_sharpe / valid_results
                                    
                                    if avg_sharpe > best_overall_sharpe:
                                        best_overall_sharpe = avg_sharpe
                                        best_overall_params = params.copy()
                                        
                                        logging.info(f"🔥 새로운 최고 파라미터 발견!")
                                        logging.info(f"📊 평균 Sharpe: {avg_sharpe:.4f}")
                                        logging.info(f"⚙️ 파라미터: {params}")
            
            self.optimized_params = best_overall_params
            
            # 최적화 기록 저장
            optimization_record = {
                'timestamp': datetime.now().isoformat(),
                'best_params': best_overall_params,
                'best_sharpe': best_overall_sharpe,
                'total_combinations_tested': total_combinations
            }
            
            self.save_optimization_record(optimization_record)
            
            logging.info(f"✅ 지속적인 파라미터 최적화 완료!")
            logging.info(f"📊 최고 Sharpe Ratio: {best_overall_sharpe:.4f}")
            logging.info(f"⚙️ 최적 파라미터: {best_overall_params}")
            
            return best_overall_params
            
        except Exception as e:
            logging.error(f"❌ 지속적인 파라미터 최적화 오류: {str(e)}")
            return self.optimized_params
    
    def save_performance_record(self, record):
        """성능 기록 저장"""
        try:
            key = f"{record['symbol']}_{record['timeframe']}_{record['model_type']}"
            performance_file = self.performance_db / f"{key}_performance.json"
            
            # 기존 기록 로드
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {'history': []}
            
            # 새 기록 추가
            existing_data['history'].append(record)
            
            # 최근 10개만 유지
            if len(existing_data['history']) > 10:
                existing_data['history'] = existing_data['history'][-10:]
            
            # 최근 성능 계산
            recent_performances = [r['test_accuracy'] for r in existing_data['history']]
            existing_data['recent_accuracy'] = np.mean(recent_performances)
            
            # 파일 저장
            with open(performance_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"❌ 성능 기록 저장 오류: {str(e)}")
    
    def save_optimization_record(self, record):
        """최적화 기록 저장"""
        try:
            optimization_file = self.performance_db / 'optimization_history.json'
            
            # 기존 기록 로드
            if optimization_file.exists():
                with open(optimization_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {'history': []}
            
            # 새 기록 추가
            existing_data['history'].append(record)
            
            # 최근 50개만 유지
            if len(existing_data['history']) > 50:
                existing_data['history'] = existing_data['history'][-50:]
            
            # 파일 저장
            with open(optimization_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"❌ 최적화 기록 저장 오류: {str(e)}")
    
    def check_performance_degradation(self):
        """성능 저하 확인"""
        try:
            logging.info("🔍 성능 저하 확인 중...")
            
            for symbol in self.optimization_config['symbols']:
                for timeframe in self.optimization_config['timeframes']:
                    for model_type in self.optimization_config['models']:
                        key = f"{symbol}_{timeframe}_{model_type}"
                        performance_file = self.performance_db / f"{key}_performance.json"
                        
                        if performance_file.exists():
                            with open(performance_file, 'r') as f:
                                data = json.load(f)
                            
                            if 'history' in data and len(data['history']) >= 2:
                                recent_accuracy = data['history'][-1]['test_accuracy']
                                previous_accuracy = data['history'][-2]['test_accuracy']
                                
                                # 성능 저하 확인 (5% 이상 하락)
                                if recent_accuracy < previous_accuracy * 0.95:
                                    logging.warning(f"⚠️ 성능 저하 감지: {key}")
                                    logging.warning(f"📊 이전: {previous_accuracy:.4f} → 현재: {recent_accuracy:.4f}")
                                    
                                    # 재훈련 필요 표시
                                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"❌ 성능 저하 확인 오류: {str(e)}")
            return False
    
    def schedule_optimization_tasks(self):
        """최적화 작업 스케줄링"""
        try:
            # 24시간마다 모델 재훈련
            schedule.every(self.optimization_config['retrain_interval_hours']).hours.do(self.retrain_all_models)
            
            # 6시간마다 파라미터 최적화
            schedule.every(self.optimization_config['optimization_interval_hours']).hours.do(self.continuous_parameter_optimization)
            
            # 1시간마다 성능 모니터링
            schedule.every().hour.do(self.check_performance_degradation)
            
            logging.info("📅 최적화 작업 스케줄링 완료!")
            
        except Exception as e:
            logging.error(f"❌ 스케줄링 오류: {str(e)}")
    
    def retrain_all_models(self):
        """모든 모델 재훈련"""
        try:
            logging.info("🔄 모든 모델 재훈련 시작...")
            
            success_count = 0
            total_count = len(self.optimization_config['symbols']) * len(self.optimization_config['timeframes']) * len(self.optimization_config['models'])
            
            for symbol in self.optimization_config['symbols']:
                for timeframe in self.optimization_config['timeframes']:
                    for model_type in self.optimization_config['models']:
                        try:
                            if self.train_model_with_hyperparameter_tuning(symbol, timeframe, model_type):
                                success_count += 1
                                logging.info(f"✅ {symbol} {timeframe} {model_type} 재훈련 성공!")
                            else:
                                logging.error(f"❌ {symbol} {timeframe} {model_type} 재훈련 실패!")
                        except Exception as e:
                            logging.error(f"❌ {symbol} {timeframe} {model_type} 재훈련 오류: {str(e)}")
            
            logging.info(f"🎉 모델 재훈련 완료! 성공: {success_count}/{total_count}")
            
        except Exception as e:
            logging.error(f"❌ 모델 재훈련 오류: {str(e)}")
    
    def run_continuous_optimization(self):
        """지속적인 최적화 실행"""
        try:
            logging.info("🚀 지속적인 최적화 시스템 시작...")
            
            # 초기 모델 훈련
            self.retrain_all_models()
            
            # 초기 파라미터 최적화
            self.continuous_parameter_optimization()
            
            # 스케줄링 설정
            self.schedule_optimization_tasks()
            
            # 지속적인 실행
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # 1분마다 체크
                    
                except KeyboardInterrupt:
                    logging.info("사용자에 의해 중단되었습니다.")
                    break
                except Exception as e:
                    logging.error(f"❌ 실행 오류: {str(e)}")
                    time.sleep(300)  # 5분 대기 후 재시도
            
        except Exception as e:
            logging.error(f"❌ 지속적인 최적화 오류: {str(e)}")
    
    def start(self):
        """지속적인 최적화 시스템 시작"""
        self.is_running = True
        logging.info("🚀 지속적인 파인튜닝 시스템 시작...")
        
        try:
            self.run_continuous_optimization()
        except KeyboardInterrupt:
            logging.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            logging.error(f"❌ 실행 오류: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """지속적인 최적화 시스템 중지"""
        self.is_running = False
        logging.info("지속적인 파인튜닝 시스템이 중지되었습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 지속적인 파인튜닝 및 백테스팅 시스템")
    print("=" * 60)
    
    try:
        optimization_system = ContinuousOptimizationSystem()
        optimization_system.start()
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
