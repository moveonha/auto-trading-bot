

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from supabase import create_client
import json
from tqdm import tqdm

def load_env_file():
    """환경변수 로드"""
    config_file = Path('.env')
    if not config_file.exists():
        return False

    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    return True

def get_supabase_client():
    """Supabase 클라이언트 생성"""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    return create_client(url, key)

def get_binance_klines_batch(symbol, interval, start_time, end_time):
    """
    Binance API에서 배치로 OHLCV 데이터 가져오기
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return None
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # 데이터 타입 변환
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        return df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"❌ Binance API 오류 ({symbol} {interval}): {str(e)}")
        return None

def save_to_supabase_batch(df, symbol, timeframe, supabase):
    """
    데이터를 Supabase에 배치 저장
    """
    try:
        if df is None or df.empty:
            return 0
        
        # 데이터 준비
        records = []
        for _, row in df.iterrows():
            record = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': int(row['timestamp']),
                'datetime': row['datetime'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            records.append(record)
        
        # 배치 삽입 (upsert로 중복 방지)
        if records:
            result = supabase.table('crypto_ohlcv').upsert(records).execute()
            return len(records)
            
    except Exception as e:
        print(f"❌ Supabase 저장 오류: {str(e)}")
        return 0

def collect_3years_data(symbols, timeframes, years=3):
    """
    3년치 데이터 수집
    """
    print(f"📊 {years}년치 데이터 수집 시작")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    # 시간 범위 계산 (3년 전부터 현재까지)
    end_time = int(time.time() * 1000)
    start_time = end_time - (years * 365 * 24 * 60 * 60 * 1000)
    
    print(f"📅 수집 기간: {datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d')} ~ {datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d')}")
    print(f"📊 수집 대상: {len(symbols)}개 심볼 × {len(timeframes)}개 타임프레임 = {len(symbols) * len(timeframes)}개 조합")
    
    total_requests = 0
    total_saved = 0
    
    # 각 타임프레임별 배치 크기 (3년치 데이터를 효율적으로 수집)
    batch_sizes = {
        '1m': 1000,      # 1000분 = 약 16.7시간
        '5m': 1000,      # 5000분 = 약 3.5일
        '15m': 1000,     # 15000분 = 약 10.4일
        '1h': 1000,      # 1000시간 = 약 41.7일
        '4h': 1000,      # 4000시간 = 약 166.7일
        '1d': 1000       # 1000일 = 약 2.7년
    }
    
    # 전체 진행률을 위한 총 작업 수 계산
    total_tasks = len(symbols) * len(timeframes)
    current_task = 0
    
    for symbol in symbols:
        print(f"\n🔍 {symbol} 데이터 수집 중...")
        
        for timeframe in timeframes:
            current_task += 1
            print(f"   ⏱️  [{current_task}/{total_tasks}] {timeframe} 타임프레임 처리 중...")
            
            current_start = start_time
            batch_count = 0
            symbol_saved = 0
            
            # 진행률 표시를 위한 전체 배치 수 계산
            total_batches = 0
            temp_start = start_time
            while temp_start < end_time:
                temp_end = min(temp_start + (batch_sizes[timeframe] * get_interval_ms(timeframe)), end_time)
                total_batches += 1
                temp_start = temp_end
            
            print(f"      📊 예상 배치 수: {total_batches}개")
            
            with tqdm(total=total_batches, desc=f"      {symbol} {timeframe}", unit="batch") as pbar:
                while current_start < end_time:
                    current_end = min(current_start + (batch_sizes[timeframe] * get_interval_ms(timeframe)), end_time)
                    
                    # 데이터 가져오기
                    df = get_binance_klines_batch(symbol, timeframe, current_start, current_end)
                    
                    if df is not None and not df.empty:
                        # 데이터 저장
                        saved_count = save_to_supabase_batch(df, symbol, timeframe, supabase)
                        symbol_saved += saved_count
                        total_saved += saved_count
                    
                    batch_count += 1
                    total_requests += 1
                    pbar.update(1)
                    pbar.set_postfix({'saved': symbol_saved, 'total': total_saved})
                    
                    # API 제한 방지
                    time.sleep(0.1)
                    
                    current_start = current_end
            
            print(f"   ✅ {symbol} {timeframe}: {symbol_saved:,}개 데이터 저장 완료")
    
    print(f"\n🎉 {years}년치 데이터 수집 완료!")
    print(f"📊 총 요청 수: {total_requests:,}개")
    print(f"📊 총 저장된 데이터: {total_saved:,}개")
    
    return total_saved

def get_interval_ms(timeframe):
    """타임프레임을 밀리초로 변환"""
    intervals = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }
    return intervals.get(timeframe, 60 * 1000)

def estimate_collection_time(symbols, timeframes, years=3):
    """수집 시간 예상"""
    print(f"⏱️  수집 시간 예상")
    print("=" * 40)
    
    # 각 타임프레임별 예상 배치 수 계산
    total_batches = 0
    
    for timeframe in timeframes:
        batch_sizes = {
            '1m': 1000,
            '5m': 1000,
            '15m': 1000,
            '1h': 1000,
            '4h': 1000,
            '1d': 1000
        }
        
        start_time = int(time.time() * 1000) - (years * 365 * 24 * 60 * 60 * 1000)
        end_time = int(time.time() * 1000)
        
        temp_start = start_time
        timeframe_batches = 0
        
        while temp_start < end_time:
            temp_end = min(temp_start + (batch_sizes[timeframe] * get_interval_ms(timeframe)), end_time)
            timeframe_batches += 1
            temp_start = temp_end
        
        total_batches += timeframe_batches * len(symbols)
        
        # 각 타임프레임별 예상 데이터 개수
        expected_data_points = {
            '1m': years * 365 * 24 * 60,      # 3년 × 365일 × 24시간 × 60분
            '5m': years * 365 * 24 * 12,      # 3년 × 365일 × 24시간 × 12
            '15m': years * 365 * 24 * 4,      # 3년 × 365일 × 24시간 × 4
            '1h': years * 365 * 24,           # 3년 × 365일 × 24시간
            '4h': years * 365 * 6,            # 3년 × 365일 × 6
            '1d': years * 365                 # 3년 × 365일
        }
        
        print(f"   {timeframe}: {timeframe_batches} 배치 × {len(symbols)} 심볼 = {timeframe_batches * len(symbols)} 요청 (예상 {expected_data_points[timeframe]:,}개 데이터)")
    
    # 예상 소요 시간 (각 요청당 0.1초 + 0.1초 대기)
    estimated_seconds = total_batches * 0.2
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    
    print(f"\n📊 총 예상 요청 수: {total_batches:,}개")
    print(f"⏱️  예상 소요 시간: {estimated_hours:.1f}시간 ({estimated_minutes:.1f}분)")
    
    # 예상 총 데이터 개수
    total_expected_data = sum([
        years * 365 * 24 * 60,      # 1m
        years * 365 * 24 * 12,      # 5m
        years * 365 * 24 * 4,       # 15m
        years * 365 * 24,           # 1h
        years * 365 * 6,            # 4h
        years * 365                 # 1d
    ]) * len(symbols)
    
    print(f"📊 예상 총 데이터 개수: {total_expected_data:,}개")
    
    return total_batches, estimated_hours, total_expected_data

def main():
    """메인 실행 함수"""
    print("🚀 3년치 암호화폐 데이터 수집기")
    print("=" * 60)
    
    # 환경변수 로드
    load_env_file()
    
    # 수집할 심볼과 타임프레임 설정
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    print(f"📊 수집 대상: {', '.join(symbols)}")
    print(f"⏱️  타임프레임: {', '.join(timeframes)}")
    
    # 수집 시간 예상
    total_batches, estimated_hours, total_expected_data = estimate_collection_time(symbols, timeframes, 3)
    
    # 사용자 확인
    print(f"\n⚠️  주의사항:")
    print("   - 3년치 데이터는 매우 많은 양입니다")
    print("   - 예상 소요 시간: {:.1f}시간".format(estimated_hours))
    print("   - 예상 데이터 개수: {:,}개".format(total_expected_data))
    print("   - 중간에 중단하면 Ctrl+C로 중단할 수 있습니다")
    
    confirm = input(f"\n계속 진행하시겠습니까? (y/n): ").strip().lower()
    
    if confirm == 'y':
        print(f"\n🚀 3년치 데이터 수집 시작!")
        start_time = time.time()
        
        try:
            total_saved = collect_3years_data(symbols, timeframes, 3)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n🎉 수집 완료!")
            print(f"⏱️  실제 소요 시간: {duration/3600:.1f}시간")
            print(f"📊 총 저장된 데이터: {total_saved:,}개")
            print(f"📊 예상 대비: {total_saved/total_expected_data*100:.1f}%")
            
        except KeyboardInterrupt:
            print(f"\n👋 사용자에 의해 중단되었습니다.")
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
    else:
        print("❌ 수집이 취소되었습니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
