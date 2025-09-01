# 🚀 트레이딩 신호 생성기
# 훈련된 AI 모델들을 사용한 실시간 트레이딩 신호 생성

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from datetime import datetime, timedelta
import pandas_ta as ta
from supabase import create_client
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class TradingSignalGenerator:
    def __init__(self, supabase_url, supabase_key, models_dir='models', results_dir='results'):
        """트레이딩 신호 생성기 초기화"""
        self.supabase = create_client(supabase_url, supabase_key)
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.model_configs = {}
        self.current_signals = {}

    def load_models(self):
        """저장된 모델들 로드"""
        print('🔄 트레이딩 모델 로드 중...')

        if not os.path.exists(self.models_dir):
            print(f'❌ 모델 디렉토리가 없습니다: {self.models_dir}')
            return

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]

        for model_file in model_files:
            try:
                model_path = os.path.join(self.models_dir, model_file)
                model = joblib.load(model_path)

                # 모델 이름에서 정보 추출
                model_name = model_file.replace('.pkl', '')
                parts = model_name.split('_')

                if len(parts) >= 4:
                    symbol = parts[0]
                    timeframe = parts[1]
                    model_type = parts[2]
                    timestamp = '_'.join(parts[3:])

                    # 결과 파일에서 설정 정보 로드
                    result_file = os.path.join(self.results_dir, f"{model_name}_results.json")
                    config = {}

                    if os.path.exists(result_file):
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            config = {
                                'feature_columns': result_data.get('feature_columns', []),
                                'metrics': result_data.get('metrics', {}),
                                'best_params': result_data.get('best_params', {}),
                                'model_name': model_name
                            }

                    key = f"{symbol}_{timeframe}_{model_type}"
                    self.models[key] = {
                        'model': model,
                        'config': config,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'model_type': model_type,
                        'timestamp': timestamp
                    }

                    print(f'✅ {model_name} 로드 완료 (F1: {config.get("metrics", {}).get("f1", 0):.4f})')

            except Exception as e:
                print(f'❌ {model_file} 로드 실패: {e}')

        print(f'📊 총 {len(self.models)}개 트레이딩 모델 로드 완료')

    def get_latest_data(self, symbol, timeframe, limit=1000):
        """최신 데이터 수집"""
        try:
            response = self.supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol.upper()).eq('timeframe', timeframe).order('timestamp', desc=True).limit(limit).execute()

            if not response.data:
                raise ValueError(f'데이터를 찾을 수 없습니다: {symbol} {timeframe}')

            df = pd.DataFrame(response.data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('datetime').reset_index(drop=True)

            return df

        except Exception as e:
            print(f'❌ 데이터 수집 실패: {e}')
            return None

    def calculate_features(self, df):
        """기술적 지표 계산"""
        try:
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

            return df

        except Exception as e:
            print(f'❌ 특성 계산 실패: {e}')
            return None

    def generate_signal(self, symbol, timeframe, model_type='lightgbm'):
        """단일 모델로 트레이딩 신호 생성"""
        key = f"{symbol}_{timeframe}_{model_type}"

        if key not in self.models:
            print(f'❌ 모델을 찾을 수 없습니다: {key}')
            return None

        model_info = self.models[key]
        model = model_info['model']
        config = model_info['config']
        feature_columns = config.get('feature_columns', [])

        # 최신 데이터 수집
        df = self.get_latest_data(symbol, timeframe, limit=1000)
        if df is None:
            return None

        # 특성 계산
        df = self.calculate_features(df)
        if df is None:
            return None

        # 최신 데이터에서 특성 추출
        latest_features = df[feature_columns].iloc[-1:].dropna()

        if len(latest_features) == 0:
            print(f'❌ 특성 데이터가 부족합니다: {symbol} {timeframe}')
            return None

        try:
            # 예측 수행
            prediction = model.predict(latest_features)[0]
            prediction_proba = model.predict_proba(latest_features)[0]

            # 신호 생성
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': model_type,
                'model_name': config.get('model_name', ''),
                'timestamp': datetime.now().isoformat(),
                'current_price': df['close'].iloc[-1],
                'prediction': int(prediction),
                'confidence': float(max(prediction_proba)),
                'buy_probability': float(prediction_proba[1]),
                'sell_probability': float(prediction_proba[0]),
                'model_metrics': config.get('metrics', {}),
                'signal_strength': self._calculate_signal_strength(prediction_proba[1], config.get('metrics', {}).get('f1', 0))
            }

            return signal

        except Exception as e:
            print(f'❌ 신호 생성 실패: {e}')
            return None

    def _calculate_signal_strength(self, buy_probability, model_f1_score):
        """신호 강도 계산"""
        # 신호 강도 = 구매 확률 * 모델 성능
        signal_strength = buy_probability * model_f1_score

        # 강도 레벨 분류
        if signal_strength >= 0.6:
            return 'STRONG_BUY'
        elif signal_strength >= 0.4:
            return 'BUY'
        elif signal_strength >= 0.2:
            return 'WEAK_BUY'
        elif signal_strength >= 0.1:
            return 'NEUTRAL'
        else:
            return 'SELL'

    def generate_multi_model_signals(self, symbol, timeframe):
        """다중 모델 신호 생성 및 앙상블"""
        print(f'🔄 {symbol} {timeframe} 다중 모델 신호 생성 중...')

        signals = []
        available_models = [key for key in self.models.keys() if key.startswith(f"{symbol}_{timeframe}_")]

        for model_key in available_models:
            model_type = model_key.split('_')[-1]
            signal = self.generate_signal(symbol, timeframe, model_type)

            if signal:
                signals.append(signal)
                print(f'✅ {model_type.upper()} 신호 생성: {signal["signal_strength"]} (확신도: {signal["confidence"]:.3f})')

        if signals:
            # 앙상블 신호 생성
            ensemble_signal = self._create_ensemble_signal(signals)
            return ensemble_signal, signals

        return None, []

    def _create_ensemble_signal(self, signals):
        """앙상블 신호 생성"""
        if not signals:
            return None

        # 가중 평균 계산 (모델 성능 기반)
        total_weight = 0
        weighted_buy_prob = 0
        weighted_confidence = 0

        for signal in signals:
            weight = signal['model_metrics'].get('f1', 0.5)  # F1 점수를 가중치로 사용
            total_weight += weight
            weighted_buy_prob += signal['buy_probability'] * weight
            weighted_confidence += signal['confidence'] * weight

        if total_weight > 0:
            avg_buy_prob = weighted_buy_prob / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            avg_buy_prob = np.mean([s['buy_probability'] for s in signals])
            avg_confidence = np.mean([s['confidence'] for s in signals])

        # 앙상블 신호 생성
        ensemble_signal = {
            'symbol': signals[0]['symbol'],
            'timeframe': signals[0]['timeframe'],
            'timestamp': datetime.now().isoformat(),
            'signal_type': 'ENSEMBLE',
            'current_price': signals[0]['current_price'],
            'ensemble_buy_probability': avg_buy_prob,
            'ensemble_confidence': avg_confidence,
            'ensemble_signal_strength': self._calculate_signal_strength(avg_buy_prob, avg_confidence),
            'model_count': len(signals),
            'individual_signals': signals
        }

        return ensemble_signal

    def generate_all_signals(self):
        """모든 심볼/타임프레임에 대한 신호 생성"""
        print('🚀 전체 트레이딩 신호 생성 시작')

        # 사용 가능한 심볼/타임프레임 조합 찾기
        combinations = set()
        for key in self.models.keys():
            parts = key.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                combinations.add((symbol, timeframe))

        all_signals = {}

        for symbol, timeframe in combinations:
            print(f'\n🔄 {symbol} {timeframe} 신호 생성 중...')

            ensemble_signal, individual_signals = self.generate_multi_model_signals(symbol, timeframe)

            if ensemble_signal:
                all_signals[f"{symbol}_{timeframe}"] = {
                    'ensemble': ensemble_signal,
                    'individual': individual_signals
                }
                print(f'✅ {symbol} {timeframe} 앙상블 신호: {ensemble_signal["ensemble_signal_strength"]}')

        self.current_signals = all_signals
        return all_signals

    def save_signals(self, output_path='trading_signals.json'):
        """신호 저장"""
        if not self.current_signals:
            print('❌ 저장할 신호가 없습니다')
            return

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_signals, f, indent=2, ensure_ascii=False)

            print(f'💾 신호 저장 완료: {output_path}')

        except Exception as e:
            print(f'❌ 신호 저장 실패: {e}')

    def create_signal_dashboard(self, output_path='signal_dashboard.html'):
        """신호 대시보드 생성"""
        if not self.current_signals:
            print('❌ 시각화할 신호가 없습니다')
            return

        print('📊 신호 대시보드 생성 중...')

        # HTML 대시보드 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI 트레이딩 신호 대시보드</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
                .signal-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .signal-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .signal-strength {{ font-size: 18px; font-weight: bold; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0; }}
                .strong-buy {{ background-color: #d4edda; color: #155724; }}
                .buy {{ background-color: #d1ecf1; color: #0c5460; }}
                .weak-buy {{ background-color: #fff3cd; color: #856404; }}
                .neutral {{ background-color: #f8f9fa; color: #6c757d; }}
                .sell {{ background-color: #f8d7da; color: #721c24; }}
                .metric {{ display: inline-block; margin: 5px; padding: 5px; background-color: #e9ecef; border-radius: 3px; }}
                .timestamp {{ color: #6c757d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 AI 트레이딩 신호 대시보드</h1>
                <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="signal-grid">
        """

        for key, signal_data in self.current_signals.items():
            ensemble = signal_data['ensemble']
            individual = signal_data['individual']

            # 신호 강도에 따른 CSS 클래스
            strength_class = ensemble['ensemble_signal_strength'].lower().replace('_', '-')

            html_content += f"""
                <div class="signal-card">
                    <h3>{ensemble['symbol']} ({ensemble['timeframe']})</h3>
                    <div class="signal-strength {strength_class}">
                        {ensemble['ensemble_signal_strength']}
                    </div>
                    <div class="metric">
                        <strong>현재가:</strong> ${ensemble['current_price']:,.4f}
                    </div>
                    <div class="metric">
                        <strong>매수 확률:</strong> {ensemble['ensemble_buy_probability']:.1%}
                    </div>
                    <div class="metric">
                        <strong>확신도:</strong> {ensemble['ensemble_confidence']:.1%}
                    </div>
                    <div class="metric">
                        <strong>모델 수:</strong> {ensemble['model_count']}개
                    </div>
                    <div class="timestamp">
                        {ensemble['timestamp']}
                    </div>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # HTML 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f'✅ 신호 대시보드 생성 완료: {output_path}')
        return output_path

    def run_signal_generation(self):
        """전체 신호 생성 프로세스 실행"""
        print('🚀 트레이딩 신호 생성 시작')
        print('=' * 60)

        # 1. 모델 로드
        self.load_models()

        if not self.models:
            print('❌ 로드된 모델이 없습니다')
            return None

        # 2. 모든 신호 생성
        all_signals = self.generate_all_signals()

        if not all_signals:
            print('❌ 생성된 신호가 없습니다')
            return None

        # 3. 신호 저장
        self.save_signals()

        # 4. 대시보드 생성
        self.create_signal_dashboard()

        print('=' * 60)
        print('✅ 트레이딩 신호 생성 완료!')

        # 요약 출력
        print('\n📊 신호 요약:')
        for key, signal_data in all_signals.items():
            ensemble = signal_data['ensemble']
            print(f"{key}: {ensemble['ensemble_signal_strength']} (매수확률: {ensemble['ensemble_buy_probability']:.1%})")

        return all_signals

# 사용 예시
if __name__ == "__main__":
    # Supabase 설정
    SUPABASE_URL = "https://your-project.supabase.co"
    SUPABASE_KEY = "your-anon-key"

    # 트레이딩 신호 생성기 생성
    signal_generator = TradingSignalGenerator(SUPABASE_URL, SUPABASE_KEY)

    # 신호 생성 실행
    signals = signal_generator.run_signal_generation()

    if signals:
        print(f'\n🎉 총 {len(signals)}개 조합의 신호 생성 완료!')

