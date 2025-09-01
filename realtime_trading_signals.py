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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_signals.log'),
        logging.StreamHandler()
    ]
)

class RealtimeTradingSignals:
    def __init__(self):
        self.load_env_file()
        self.supabase = self.get_supabase_client()
        self.websocket = None
        self.is_running = False

        # 실시간 데이터 저장소 (메모리)
        self.realtime_data = {}
        self.signal_history = []

        # 수집할 심볼과 타임프레임
        self.symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt']
        self.timeframes = ['1m', '5m', '15m', '1h']

        # 신호 생성 임계값
        self.signal_threshold = 2.5

        # 최근 신호 추적 (중복 방지)
        self.last_signals = {}

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

    async def connect_websocket(self):
        """WebSocket 연결"""
        try:
            ws_url = "wss://stream.binance.com:9443/ws"

            # 구독할 스트림들 생성
            streams = []
            for symbol in self.symbols:
                streams.append(f"{symbol}@kline_1m")
                streams.append(f"{symbol}@kline_5m")
                streams.append(f"{symbol}@kline_15m")
                streams.append(f"{symbol}@kline_1h")

            stream_url = f"{ws_url}/{'/'.join(streams)}"

            logging.info(f"WebSocket 연결 시도: {stream_url}")

            self.websocket = await websockets.connect(stream_url)
            logging.info("✅ WebSocket 연결 성공!")

            return True

        except Exception as e:
            logging.error(f"❌ WebSocket 연결 실패: {str(e)}")
            return False

    def update_realtime_data(self, data):
        """실시간 데이터 업데이트"""
        try:
            kline = data['k']

            symbol = data['s'].lower()
            timeframe = kline['i']
            is_closed = kline['x']

            key = f"{symbol}_{timeframe}"

            if key not in self.realtime_data:
                self.realtime_data[key] = []

            kline_data = {
                'symbol': symbol.upper(),
                'timeframe': timeframe,
                'timestamp': kline['t'],
                'datetime': datetime.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': is_closed
            }

            # 최신 데이터로 업데이트
            self.realtime_data[key].append(kline_data)

            # 최대 500개 데이터만 유지 (메모리 절약)
            if len(self.realtime_data[key]) > 500:
                self.realtime_data[key] = self.realtime_data[key][-500:]

            return kline_data

        except Exception as e:
            logging.error(f"❌ 실시간 데이터 업데이트 오류: {str(e)}")
            return None

    def calculate_technical_indicators(self, symbol, timeframe):
        """기술적 지표 계산"""
        try:
            key = f"{symbol}_{timeframe}"
            if key not in self.realtime_data or len(self.realtime_data[key]) < 50:
                return None

            df = pd.DataFrame(self.realtime_data[key])

            # 1분봉에 최적화된 지표들
            df['sma_5'] = ta.sma(df['close'], length=5)
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)

            # MACD (빠른 설정)
            macd = ta.macd(df['close'], fast=6, slow=13, signal=4)
            df['macd'] = macd['MACD_6_13_4']
            df['macd_signal'] = macd['MACDs_6_13_4']

            # RSI (단기)
            df['rsi'] = ta.rsi(df['close'], length=9)

            # 볼린저 밴드 (단기)
            bb = ta.bbands(df['close'], length=10, std=2)
            df['bb_upper'] = bb['BBU_10_2.0']
            df['bb_lower'] = bb['BBL_10_2.0']

            # 스토캐스틱 (빠른 설정)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3)
            df['stoch_k'] = stoch['STOCHk_5_3_3']
            df['stoch_d'] = stoch['STOCHd_5_3_3']

            # ATR (단기)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=7)

            # ADX (단기)
            adx = ta.adx(df['high'], df['low'], df['close'], length=7)
            df['adx'] = adx['ADX_7']

            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=9)

            # Momentum
            df['momentum'] = ta.mom(df['close'], length=10)

            return df

        except Exception as e:
            logging.error(f"❌ 기술적 지표 계산 오류: {str(e)}")
            return None

    def generate_realtime_signal(self, symbol, timeframe):
        """실시간 트레이딩 신호 생성"""
        try:
            df = self.calculate_technical_indicators(symbol, timeframe)
            if df is None or len(df) < 50:
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None

            if prev is None:
                return None

            signal = {
                'symbol': symbol.upper(),
                'timeframe': timeframe,
                'datetime': latest['datetime'],
                'current_price': latest['close'],
                'signals': [],
                'strength': 0,
                'action': 'HOLD',
                'confidence': 'LOW'
            }

            # 1. EMA 크로스오버 (빠른 신호)
            if pd.notna(latest['ema_9']) and pd.notna(latest['ema_21']) and pd.notna(prev['ema_9']) and pd.notna(prev['ema_21']):
                if latest['ema_9'] > latest['ema_21'] and prev['ema_9'] <= prev['ema_21']:
                    signal['signals'].append('EMA_BULLISH_CROSS')
                    signal['strength'] += 2
                elif latest['ema_9'] < latest['ema_21'] and prev['ema_9'] >= prev['ema_21']:
                    signal['signals'].append('EMA_BEARISH_CROSS')
                    signal['strength'] -= 2

            # 2. MACD 신호
            if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']) and pd.notna(prev['macd']) and pd.notna(prev['macd_signal']):
                if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                    signal['signals'].append('MACD_BULLISH_CROSS')
                    signal['strength'] += 2
                elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                    signal['signals'].append('MACD_BEARISH_CROSS')
                    signal['strength'] -= 2

            # 3. RSI 신호 (단기)
            if pd.notna(latest['rsi']):
                if latest['rsi'] < 20:
                    signal['signals'].append('RSI_OVERSOLD')
                    signal['strength'] += 1
                elif latest['rsi'] > 80:
                    signal['signals'].append('RSI_OVERBOUGHT')
                    signal['strength'] -= 1

            # 4. 볼린저 밴드 신호
            if pd.notna(latest['bb_lower']) and pd.notna(latest['bb_upper']):
                if latest['close'] < latest['bb_lower']:
                    signal['signals'].append('BB_LOWER_BREAK')
                    signal['strength'] += 1
                elif latest['close'] > latest['bb_upper']:
                    signal['signals'].append('BB_UPPER_BREAK')
                    signal['strength'] -= 1

            # 5. 스토캐스틱 신호
            if pd.notna(latest['stoch_k']) and pd.notna(latest['stoch_d']):
                if latest['stoch_k'] < 15 and latest['stoch_d'] < 15:
                    signal['signals'].append('STOCH_OVERSOLD')
                    signal['strength'] += 1
                elif latest['stoch_k'] > 85 and latest['stoch_d'] > 85:
                    signal['signals'].append('STOCH_OVERBOUGHT')
                    signal['strength'] -= 1

            # 6. Williams %R
            if pd.notna(latest['williams_r']):
                if latest['williams_r'] < -80:
                    signal['signals'].append('WILLIAMS_OVERSOLD')
                    signal['strength'] += 1
                elif latest['williams_r'] > -20:
                    signal['signals'].append('WILLIAMS_OVERBOUGHT')
                    signal['strength'] -= 1

            # 7. Momentum
            if pd.notna(latest['momentum']):
                if latest['momentum'] > 0:
                    signal['signals'].append('POSITIVE_MOMENTUM')
                    signal['strength'] += 0.5
                else:
                    signal['signals'].append('NEGATIVE_MOMENTUM')
                    signal['strength'] -= 0.5

            # 8. ADX (추세 강도)
            if pd.notna(latest['adx']):
                if latest['adx'] > 20:
                    signal['signals'].append('STRONG_TREND')
                    signal['strength'] = signal['strength'] * 1.3

            # 최종 액션 결정
            if signal['strength'] >= self.signal_threshold:
                signal['action'] = 'LONG'
                signal['confidence'] = 'HIGH' if signal['strength'] >= 4 else 'MEDIUM'
            elif signal['strength'] <= -self.signal_threshold:
                signal['action'] = 'SHORT'
                signal['confidence'] = 'HIGH' if signal['strength'] <= -4 else 'MEDIUM'
            else:
                signal['action'] = 'HOLD'
                signal['confidence'] = 'LOW'

            return signal

        except Exception as e:
            logging.error(f"❌ 신호 생성 오류: {str(e)}")
            return None

    def check_signal_change(self, signal):
        """신호 변화 확인 (중복 방지)"""
        if signal is None:
            return False

        key = f"{signal['symbol']}_{signal['timeframe']}"

        if key not in self.last_signals:
            self.last_signals[key] = signal
            return True

        last_signal = self.last_signals[key]

        # 액션이 변경되었거나 강한 신호인 경우
        if (last_signal['action'] != signal['action'] or
            signal['confidence'] == 'HIGH' or
            abs(signal['strength']) >= 4):

            self.last_signals[key] = signal
            return True

        return False

    def process_realtime_signal(self, signal):
        """실시간 신호 처리"""
        if signal is None:
            return

        # 신호 변화 확인
        if not self.check_signal_change(signal):
            return

        # 신호 히스토리에 추가
        self.signal_history.append(signal)

        # 최대 100개 신호만 유지
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

        # 신호 출력
        action_emoji = "🟢" if signal['action'] == 'LONG' else "🔴" if signal['action'] == 'SHORT' else "🟡"
        confidence_emoji = "🔥" if signal['confidence'] == 'HIGH' else "⚡" if signal['confidence'] == 'MEDIUM' else "💤"

        logging.info(f"🚨 {action_emoji} {signal['action']} {confidence_emoji} - {signal['symbol']} {signal['timeframe']}")
        logging.info(f"   💰 가격: ${signal['current_price']:.8f}")
        logging.info(f"   📊 강도: {signal['strength']:.1f}")
        logging.info(f"   📈 신호: {', '.join(signal['signals'][:3])}")

        # 강한 신호인 경우 즉시 알림
        if signal['confidence'] == 'HIGH':
            logging.warning(f"🔥 강한 신호 감지! {signal['symbol']} {signal['timeframe']} {signal['action']}")

            # 여기에 자동매매 로직 추가 가능
            # await self.execute_trade(signal)

    async def receive_messages(self):
        """WebSocket 메시지 수신 및 실시간 신호 생성"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # K라인 데이터 처리
                    if 'k' in data:
                        kline_data = self.update_realtime_data(data)

                        if kline_data and kline_data['is_closed']:
                            # 캔들이 완료되면 즉시 신호 생성
                            symbol = kline_data['symbol'].lower()
                            timeframe = kline_data['timeframe']

                            # 실시간 신호 생성
                            signal = self.generate_realtime_signal(symbol, timeframe)

                            # 신호 처리
                            self.process_realtime_signal(signal)

                            # 간단한 로그
                            logging.info(f"📊 {symbol.upper()} {timeframe}: ${kline_data['close']:.8f}")

                except json.JSONDecodeError as e:
                    logging.error(f"❌ JSON 파싱 오류: {str(e)}")
                except Exception as e:
                    logging.error(f"❌ 메시지 처리 오류: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logging.warning("⚠️ WebSocket 연결이 끊어졌습니다.")
        except Exception as e:
            logging.error(f"❌ 메시지 수신 오류: {str(e)}")

    async def start_realtime_signals(self):
        """실시간 신호 생성 시작"""
        logging.info("🚀 실시간 트레이딩 신호 생성 시작...")

        while self.is_running:
            try:
                # WebSocket 연결
                if not await self.connect_websocket():
                    logging.error("WebSocket 연결 실패. 10초 후 재시도...")
                    await asyncio.sleep(10)
                    continue

                # 메시지 수신 및 신호 생성
                await self.receive_messages()

            except Exception as e:
                logging.error(f"❌ 실시간 신호 생성 오류: {str(e)}")
                logging.info("10초 후 재시도...")
                await asyncio.sleep(10)

    def start(self):
        """실시간 신호 생성 시작"""
        self.is_running = True
        logging.info("실시간 트레이딩 신호 생성을 시작합니다...")
        logging.info(f"수집 대상: {', '.join(self.symbols)}")
        logging.info(f"타임프레임: {', '.join(self.timeframes)}")
        logging.info(f"신호 임계값: {self.signal_threshold}")

        try:
            asyncio.run(self.start_realtime_signals())
        except KeyboardInterrupt:
            logging.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            logging.error(f"❌ 실행 오류: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """실시간 신호 생성 중지"""
        self.is_running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        logging.info("실시간 트레이딩 신호 생성이 중지되었습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 실시간 암호화폐 트레이딩 신호 생성기")
    print("=" * 60)

    try:
        signal_generator = RealtimeTradingSignals()
        signal_generator.start()
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
