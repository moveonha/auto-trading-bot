import os
from pathlib import Path
from supabase import create_client
import pandas as pd

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

def check_data_summary():
    """데이터 요약 확인"""
    print("📊 데이터 수집 현황 확인")
    print("=" * 50)

    supabase = get_supabase_client()

    try:
        # 전체 데이터 개수 확인
        result = supabase.table('crypto_ohlcv').select('*', count='exact').execute()
        total_count = result.count
        print(f"📈 전체 데이터 개수: {total_count:,}개")

        # 심볼별 데이터 개수
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

        print(f"\n🔍 심볼별 데이터 현황:")
        print("-" * 40)

        for symbol in symbols:
            print(f"\n📊 {symbol}:")
            for timeframe in timeframes:
                result = supabase.table('crypto_ohlcv').select('*', count='exact').eq('symbol', symbol).eq('timeframe', timeframe).execute()
                count = result.count
                print(f"   {timeframe}: {count:,}개")

        # 최신 데이터 확인
        print(f"\n🕐 최신 데이터 확인:")
        print("-" * 40)

        for symbol in symbols:
            result = supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol).order('datetime', desc=True).limit(1).execute()
            if result.data:
                latest = result.data[0]
                print(f"📊 {symbol}: {latest['close']:.8f} ({latest['datetime']})")

        # 데이터 범위 확인
        print(f"\n📅 데이터 범위:")
        print("-" * 40)

        # 가장 오래된 데이터
        result = supabase.table('crypto_ohlcv').select('datetime').order('datetime', desc=False).limit(1).execute()
        if result.data:
            oldest = result.data[0]['datetime']
            print(f"📅 가장 오래된 데이터: {oldest}")

        # 가장 최신 데이터
        result = supabase.table('crypto_ohlcv').select('datetime').order('datetime', desc=True).limit(1).execute()
        if result.data:
            newest = result.data[0]['datetime']
            print(f"📅 가장 최신 데이터: {newest}")

        return True

    except Exception as e:
        print(f"❌ 데이터 확인 오류: {str(e)}")
        return False

def check_sample_data():
    """샘플 데이터 확인"""
    print(f"\n🔍 샘플 데이터 확인")
    print("=" * 50)

    supabase = get_supabase_client()

    try:
        # BTCUSDT 1시간 데이터 샘플
        result = supabase.table('crypto_ohlcv').select('*').eq('symbol', 'BTCUSDT').eq('timeframe', '1h').order('datetime', desc=True).limit(5).execute()

        if result.data:
            print("📊 BTCUSDT 1시간 데이터 (최근 5개):")
            print("-" * 80)
            for i, row in enumerate(result.data, 1):
                print(f"{i}. {row['datetime']} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.2f}")

        # ETHUSDT 1분 데이터 샘플
        result = supabase.table('crypto_ohlcv').select('*').eq('symbol', 'ETHUSDT').eq('timeframe', '1m').order('datetime', desc=True).limit(5).execute()

        if result.data:
            print(f"\n📊 ETHUSDT 1분 데이터 (최근 5개):")
            print("-" * 80)
            for i, row in enumerate(result.data, 1):
                print(f"{i}. {row['datetime']} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.2f}")

        return True

    except Exception as e:
        print(f"❌ 샘플 데이터 확인 오류: {str(e)}")
        return False

def check_data_quality():
    """데이터 품질 확인"""
    print(f"\n🔍 데이터 품질 확인")
    print("=" * 50)

    supabase = get_supabase_client()

    try:
        # 중복 데이터 확인
        result = supabase.table('crypto_ohlcv').select('symbol, timeframe, timestamp').execute()

        if result.data:
            # 중복 체크
            seen = set()
            duplicates = []

            for row in result.data:
                key = (row['symbol'], row['timeframe'], row['timestamp'])
                if key in seen:
                    duplicates.append(key)
                else:
                    seen.add(key)

            if duplicates:
                print(f"⚠️  중복 데이터 발견: {len(duplicates)}개")
                for dup in duplicates[:5]:  # 처음 5개만 표시
                    print(f"   {dup[0]} {dup[1]} {dup[2]}")
            else:
                print("✅ 중복 데이터 없음")

        # NULL 값 확인
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']

        for symbol in symbols:
            result = supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol).is_('close', 'null').execute()
            if result.data:
                print(f"⚠️  {symbol}에 NULL close 값 발견: {len(result.data)}개")
            else:
                print(f"✅ {symbol} close 값 정상")

        return True

    except Exception as e:
        print(f"❌ 데이터 품질 확인 오류: {str(e)}")
        return False

def main():
    """메인 실행 함수"""
    print("🔍 암호화폐 데이터 수집 현황 확인")
    print("=" * 60)

    # 환경변수 로드
    load_env_file()

    # 데이터 요약 확인
    check_data_summary()

    # 샘플 데이터 확인
    check_sample_data()

    # 데이터 품질 확인
    check_data_quality()

    print(f"\n✅ 데이터 확인 완료!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
