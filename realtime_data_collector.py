import os
import json
import asyncio
import websockets
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from supabase import create_client
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_data.log'),
        logging.StreamHandler()
    ]
)

class RealtimeDataCollector:
    def __init__(self):
        self.load_env_file()
        self.supabase = self.get_supabase_client()
        self.websocket = None
        self.is_running = False
        self.data_buffer = {}
        self.last_save_time = 0  # 수정: 딕셔너리가 아닌 단일 값으로 변경

        # 수집할 심볼과 타임프레임
        self.symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt']
        self.timeframes = ['1m', '5m', '15m', '1h']

        # 데이터 저장 간격 (초)
        self.save_interval = 60  # 1분마다 저장

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
            # Binance WebSocket URL (실시간 1분봉 데이터)
            ws_url = "wss://stream.binance.com:9443/ws"

            # 구독할 스트림들 생성
            streams = []
            for symbol in self.symbols:
                streams.append(f"{symbol}@kline_1m")
                streams.append(f"{symbol}@kline_5m")
                streams.append(f"{symbol}@kline_15m")
                streams.append(f"{symbol}@kline_1h")

            # 스트림 URL 생성
            stream_url = f"{ws_url}/{'/'.join(streams)}"

            logging.info(f"WebSocket 연결 시도: {stream_url}")

            self.websocket = await websockets.connect(stream_url)
            logging.info("✅ WebSocket 연결 성공!")

            return True

        except Exception as e:
            logging.error(f"❌ WebSocket 연결 실패: {str(e)}")
            return False

    def process_kline_data(self, data):
        """K라인 데이터 처리"""
        try:
            kline = data['k']

            # 데이터 추출
            symbol = data['s'].lower()
            timeframe = kline['i']
            is_closed = kline['x']  # 캔들이 완료되었는지

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

            # 버퍼에 저장
            key = f"{symbol}_{timeframe}"
            if key not in self.data_buffer:
                self.data_buffer[key] = []

            self.data_buffer[key].append(kline_data)

            # 캔들이 완료되었을 때만 로그 출력
            if is_closed:
                logging.info(f"📊 {symbol.upper()} {timeframe}: ${kline_data['close']:.8f} (완료)")

            return kline_data

        except Exception as e:
            logging.error(f"❌ 데이터 처리 오류: {str(e)}")
            return None

    async def save_data_to_supabase(self):
        """버퍼의 데이터를 Supabase에 저장"""
        try:
            for key, data_list in self.data_buffer.items():
                if not data_list:
                    continue

                # 완료된 캔들만 필터링
                completed_data = [d for d in data_list if d['is_closed']]

                if not completed_data:
                    continue

                # DataFrame 생성
                df = pd.DataFrame(completed_data)

                # 중복 제거 (timestamp 기준)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')

                # Supabase에 저장
                for _, row in df.iterrows():
                    try:
                        # upsert로 저장 (중복 방지)
                        result = self.supabase.table('crypto_ohlcv').upsert({
                            'symbol': row['symbol'],
                            'timeframe': row['timeframe'],
                            'timestamp': row['timestamp'],
                            'datetime': row['datetime'].isoformat(),
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        }).execute()

                        logging.info(f"💾 저장 완료: {row['symbol']} {row['timeframe']} {row['datetime']}")

                    except Exception as e:
                        logging.error(f"❌ 데이터 저장 오류 ({row['symbol']} {row['timeframe']}): {str(e)}")

                # 저장된 데이터는 버퍼에서 제거
                self.data_buffer[key] = [d for d in data_list if not d['is_closed']]

        except Exception as e:
            logging.error(f"❌ 데이터 저장 중 오류: {str(e)}")

    async def receive_messages(self):
        """WebSocket 메시지 수신"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # K라인 데이터 처리
                    if 'k' in data:
                        self.process_kline_data(data)

                    # 주기적으로 데이터 저장
                    current_time = time.time()
                    if current_time - self.last_save_time >= self.save_interval:
                        await self.save_data_to_supabase()
                        self.last_save_time = current_time

                except json.JSONDecodeError as e:
                    logging.error(f"❌ JSON 파싱 오류: {str(e)}")
                except Exception as e:
                    logging.error(f"❌ 메시지 처리 오류: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logging.warning("⚠️ WebSocket 연결이 끊어졌습니다.")
        except Exception as e:
            logging.error(f"❌ 메시지 수신 오류: {str(e)}")

    async def start_collection(self):
        """실시간 데이터 수집 시작"""
        logging.info("🚀 실시간 데이터 수집 시작...")

        while self.is_running:
            try:
                # WebSocket 연결
                if not await self.connect_websocket():
                    logging.error("WebSocket 연결 실패. 10초 후 재시도...")
                    await asyncio.sleep(10)
                    continue

                # 메시지 수신
                await self.receive_messages()

            except Exception as e:
                logging.error(f"❌ 데이터 수집 오류: {str(e)}")
                logging.info("10초 후 재시도...")
                await asyncio.sleep(10)

    def start(self):
        """수집 시작"""
        self.is_running = True
        logging.info("실시간 데이터 수집을 시작합니다...")
        logging.info(f"수집 대상: {', '.join(self.symbols)}")
        logging.info(f"타임프레임: {', '.join(self.timeframes)}")

        try:
            asyncio.run(self.start_collection())
        except KeyboardInterrupt:
            logging.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            logging.error(f"❌ 실행 오류: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """수집 중지"""
        self.is_running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        logging.info("실시간 데이터 수집이 중지되었습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 실시간 암호화폐 데이터 수집기")
    print("=" * 60)

    try:
        collector = RealtimeDataCollector()
        collector.start()
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
