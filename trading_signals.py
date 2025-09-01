import os
import pandas as pd
import numpy as np
from pathlib import Path
from supabase import create_client
from datetime import datetime, timedelta
import pandas_ta as ta

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

def get_market_data(symbol, timeframe, limit=1000):
    """시장 데이터 가져오기"""
    supabase = get_supabase_client()

    try:
        result = supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol).eq('timeframe', timeframe).order('datetime', desc=True).limit(limit).execute()

        if result.data:
            df = pd.DataFrame(result.data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            return df
        return None

    except Exception as e:
        print(f"❌ 데이터 가져오기 오류 ({symbol} {timeframe}): {str(e)}")
        return None

def calculate_technical_indicators(df):
    """기술적 지표 계산"""
    if df is None or len(df) < 50:
        return None

    try:
        # pandas-ta를 사용한 기술적 지표 계산
        # 이동평균선
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)

        # 지수이동평균선
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)

        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)

        # 볼린저 밴드
        bb = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']

        # 스토캐스틱
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        # ATR (Average True Range)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # OBV (On Balance Volume)
        df['obv'] = ta.obv(df['close'], df['volume'])

        # ADX (Average Directional Index)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']

        # CCI (Commodity Channel Index)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)

        return df

    except Exception as e:
        print(f"❌ 기술적 지표 계산 오류: {str(e)}")
        return None

def generate_trading_signals(df):
    """트레이딩 신호 생성"""
    if df is None or len(df) < 50:
        return None

    try:
        # 최신 데이터 (가장 마지막 행)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None

        if prev is None:
            return None

        signal = {
            'symbol': latest['symbol'],
            'timeframe': latest['timeframe'],
            'datetime': latest['datetime'],
            'price': latest['close'],
            'signals': [],
            'strength': 0,
            'action': 'HOLD'
        }

        # 1. 이동평균선 크로스오버
        if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']) and pd.notna(prev['sma_20']) and pd.notna(prev['sma_50']):
            if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                signal['signals'].append('SMA_20_50_CROSS_UP')
                signal['strength'] += 1
            elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                signal['signals'].append('SMA_20_50_CROSS_DOWN')
                signal['strength'] -= 1

        # 2. MACD 신호
        if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']) and pd.notna(prev['macd']) and pd.notna(prev['macd_signal']):
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                signal['signals'].append('MACD_BULLISH_CROSS')
                signal['strength'] += 2
            elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                signal['signals'].append('MACD_BEARISH_CROSS')
                signal['strength'] -= 2

        # 3. RSI 신호
        if pd.notna(latest['rsi']):
            if latest['rsi'] < 30:
                signal['signals'].append('RSI_OVERSOLD')
                signal['strength'] += 1
            elif latest['rsi'] > 70:
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
            if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                signal['signals'].append('STOCH_OVERSOLD')
                signal['strength'] += 1
            elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                signal['signals'].append('STOCH_OVERBOUGHT')
                signal['strength'] -= 1

        # 6. 가격 위치
        if pd.notna(latest['sma_200']):
            if latest['close'] > latest['sma_200']:
                signal['signals'].append('ABOVE_200_SMA')
                signal['strength'] += 1
            else:
                signal['signals'].append('BELOW_200_SMA')
                signal['strength'] -= 1

        # 7. ADX (추세 강도)
        if pd.notna(latest['adx']):
            if latest['adx'] > 25:
                signal['signals'].append('STRONG_TREND')
                signal['strength'] = signal['strength'] * 1.2  # 추세가 강할 때 신호 강화

        # 8. CCI 신호
        if pd.notna(latest['cci']):
            if latest['cci'] < -100:
                signal['signals'].append('CCI_OVERSOLD')
                signal['strength'] += 1
            elif latest['cci'] > 100:
                signal['signals'].append('CCI_OVERBOUGHT')
                signal['strength'] -= 1

        # 최종 액션 결정
        if signal['strength'] >= 3:
            signal['action'] = 'BUY'
            signal['confidence'] = 'HIGH' if signal['strength'] >= 5 else 'MEDIUM'
        elif signal['strength'] <= -3:
            signal['action'] = 'SELL'
            signal['confidence'] = 'HIGH' if signal['strength'] <= -5 else 'MEDIUM'
        else:
            signal['action'] = 'HOLD'
            signal['confidence'] = 'LOW'

        return signal

    except Exception as e:
        print(f"❌ 신호 생성 오류: {str(e)}")
        return None

def analyze_market_sentiment(symbols, timeframes):
    """시장 심리 분석"""
    print("🔍 시장 심리 분석 중...")
    print("=" * 60)

    all_signals = []

    for symbol in symbols:
        print(f"\n📊 {symbol} 분석 중...")

        for timeframe in timeframes:
            try:
                # 데이터 가져오기
                df = get_market_data(symbol, timeframe, limit=500)

                if df is not None:
                    # 기술적 지표 계산
                    df = calculate_technical_indicators(df)

                    if df is not None:
                        # 트레이딩 신호 생성
                        signal = generate_trading_signals(df)

                        if signal:
                            all_signals.append(signal)

                            # 신호 출력
                            action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴" if signal['action'] == 'SELL' else "🟡"
                            confidence_emoji = "🔥" if signal['confidence'] == 'HIGH' else "⚡" if signal['confidence'] == 'MEDIUM' else "💤"

                            print(f"   {timeframe}: {action_emoji} {signal['action']} {confidence_emoji} (강도: {signal['strength']:.1f})")

                            if signal['signals']:
                                print(f"      신호: {', '.join(signal['signals'][:3])}...")

            except Exception as e:
                print(f"   ❌ {timeframe} 분석 오류: {str(e)}")

    return all_signals

def generate_summary_report(signals):
    """요약 리포트 생성"""
    print(f"\n📋 트레이딩 신호 요약 리포트")
    print("=" * 60)

    if not signals:
        print("❌ 분석된 신호가 없습니다.")
        return

    # 액션별 분류
    buy_signals = [s for s in signals if s['action'] == 'BUY']
    sell_signals = [s for s in signals if s['action'] == 'SELL']
    hold_signals = [s for s in signals if s['action'] == 'HOLD']

    print(f"📊 총 신호: {len(signals)}개")
    print(f"🟢 매수 신호: {len(buy_signals)}개")
    print(f"🔴 매도 신호: {len(sell_signals)}개")
    print(f"🟡 보유 신호: {len(hold_signals)}개")

    # 강한 신호만 필터링
    strong_buy = [s for s in buy_signals if s['confidence'] == 'HIGH']
    strong_sell = [s for s in sell_signals if s['confidence'] == 'HIGH']

    print(f"\n🔥 강한 매수 신호: {len(strong_buy)}개")
    for signal in strong_buy[:3]:  # 상위 3개만 표시
        print(f"   {signal['symbol']} {signal['timeframe']}: ${signal['price']:.8f}")

    print(f"\n🔥 강한 매도 신호: {len(strong_sell)}개")
    for signal in strong_sell[:3]:  # 상위 3개만 표시
        print(f"   {signal['symbol']} {signal['timeframe']}: ${signal['price']:.8f}")

    # 심볼별 요약
    print(f"\n📈 심볼별 요약:")
    symbols = list(set([s['symbol'] for s in signals]))

    for symbol in symbols:
        symbol_signals = [s for s in signals if s['symbol'] == symbol]
        buy_count = len([s for s in symbol_signals if s['action'] == 'BUY'])
        sell_count = len([s for s in symbol_signals if s['action'] == 'SELL'])

        if buy_count > sell_count:
            sentiment = "🟢 강세"
        elif sell_count > buy_count:
            sentiment = "🔴 약세"
        else:
            sentiment = "🟡 중립"

        print(f"   {symbol}: {sentiment} (매수: {buy_count}, 매도: {sell_count})")

def main():
    """메인 실행 함수"""
    print("🚀 암호화폐 트레이딩 신호 분석기")
    print("=" * 60)

    # 환경변수 로드
    load_env_file()

    # 분석할 심볼과 타임프레임
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    timeframes = ['1h', '4h', '1d']  # 1시간, 4시간, 1일

    print(f"📊 분석 대상: {', '.join(symbols)}")
    print(f"⏱️  타임프레임: {', '.join(timeframes)}")

    # 시장 심리 분석
    signals = analyze_market_sentiment(symbols, timeframes)

    # 요약 리포트 생성
    generate_summary_report(signals)

    print(f"\n🎉 분석 완료!")
    print("💡 이 신호들은 참고용이며, 실제 투자는 신중하게 결정하세요.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
