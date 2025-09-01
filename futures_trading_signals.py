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
    """기술적 지표 계산 (1분봉 최적화)"""
    if df is None or len(df) < 50:
        return None

    try:
        # 1분봉에 최적화된 지표들
        # 이동평균선 (단기 중심)
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)

        # 지수이동평균선
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)

        # MACD (빠른 설정)
        macd = ta.macd(df['close'], fast=6, slow=13, signal=4)
        df['macd'] = macd['MACD_6_13_4']
        df['macd_signal'] = macd['MACDs_6_13_4']
        df['macd_hist'] = macd['MACDh_6_13_4']

        # RSI (단기)
        df['rsi'] = ta.rsi(df['close'], length=9)

        # 볼린저 밴드 (단기)
        bb = ta.bbands(df['close'], length=10, std=2)
        df['bb_upper'] = bb['BBU_10_2.0']
        df['bb_middle'] = bb['BBM_10_2.0']
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

        # CCI (단기)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=9)

        # 추가 지표들
        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=9)

        # Momentum
        df['momentum'] = ta.mom(df['close'], length=10)

        return df

    except Exception as e:
        print(f"❌ 기술적 지표 계산 오류: {str(e)}")
        return None

def calculate_position_sizes(price, leverage=20, risk_percent=2):
    """포지션 크기 계산 (레버리지 20배 기준)"""
    # 계좌 잔고 (가정: $10,000)
    account_balance = 10000

    # 리스크 금액 (계좌의 2%)
    risk_amount = account_balance * (risk_percent / 100)

    # 레버리지 20배 기준 포지션 크기
    position_value = risk_amount * leverage

    # 수량 계산
    quantity = position_value / price

    return {
        'position_value': position_value,
        'quantity': quantity,
        'risk_amount': risk_amount
    }

def calculate_entry_exit_prices(signal_type, current_price, atr, timeframe):
    """진입가/청산가 계산"""
    # ATR 기반 스탑로스 및 타겟 계산
    atr_multiplier = {
        '1m': 1.5,   # 1분봉: 빠른 진입/청산
        '5m': 2.0,   # 5분봉
        '15m': 2.5,  # 15분봉
        '1h': 3.0,   # 1시간봉
        '4h': 4.0,   # 4시간봉
        '1d': 5.0    # 1일봉
    }

    multiplier = atr_multiplier.get(timeframe, 2.0)
    atr_value = atr if pd.notna(atr) else current_price * 0.01  # ATR이 없으면 가격의 1%

    if signal_type == 'LONG':
        # 롱 포지션
        entry_price = current_price + (atr_value * 0.5)  # 약간 위에서 진입
        stop_loss = current_price - (atr_value * multiplier)
        take_profit = current_price + (atr_value * multiplier * 2)  # 2:1 리스크/리워드

    else:  # SHORT
        # 숏 포지션
        entry_price = current_price - (atr_value * 0.5)  # 약간 아래에서 진입
        stop_loss = current_price + (atr_value * multiplier)
        take_profit = current_price - (atr_value * multiplier * 2)  # 2:1 리스크/리워드

    return {
        'entry_price': round(entry_price, 8),
        'stop_loss': round(stop_loss, 8),
        'take_profit': round(take_profit, 8),
        'risk_reward_ratio': 2.0
    }

def generate_futures_signals(df):
    """선물 거래 신호 생성 (1분봉 최적화)"""
    if df is None or len(df) < 50:
        return None

    try:
        # 최신 데이터
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None

        if prev is None:
            return None

        signal = {
            'symbol': latest['symbol'],
            'timeframe': latest['timeframe'],
            'datetime': latest['datetime'],
            'current_price': latest['close'],
            'signals': [],
            'strength': 0,
            'action': 'HOLD',
            'position_type': None,
            'entry_exit_prices': None,
            'position_sizes': None
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
                signal['strength'] = signal['strength'] * 1.3  # 추세가 강할 때 신호 강화

        # 최종 액션 결정 (선물 거래용 임계값)
        if signal['strength'] >= 2.5:
            signal['action'] = 'LONG'
            signal['position_type'] = 'LONG'
            signal['confidence'] = 'HIGH' if signal['strength'] >= 4 else 'MEDIUM'
        elif signal['strength'] <= -2.5:
            signal['action'] = 'SHORT'
            signal['position_type'] = 'SHORT'
            signal['confidence'] = 'HIGH' if signal['strength'] <= -4 else 'MEDIUM'
        else:
            signal['action'] = 'HOLD'
            signal['confidence'] = 'LOW'

        # 진입가/청산가 계산
        if signal['action'] != 'HOLD':
            entry_exit = calculate_entry_exit_prices(
                signal['position_type'],
                latest['close'],
                latest['atr'],
                latest['timeframe']
            )
            signal['entry_exit_prices'] = entry_exit

            # 포지션 크기 계산
            position_sizes = calculate_position_sizes(latest['close'], leverage=20)
            signal['position_sizes'] = position_sizes

        return signal

    except Exception as e:
        print(f"❌ 선물 신호 생성 오류: {str(e)}")
        return None

def analyze_futures_market(symbols, timeframes):
    """선물 시장 분석"""
    print("🔍 선물 시장 분석 중...")
    print("=" * 80)

    all_signals = []

    for symbol in symbols:
        print(f"\n📊 {symbol} 선물 분석 중...")

        for timeframe in timeframes:
            try:
                # 데이터 가져오기 (1분봉은 더 많은 데이터 필요)
                limit = 1000 if timeframe == '1m' else 500
                df = get_market_data(symbol, timeframe, limit=limit)

                if df is not None:
                    # 기술적 지표 계산
                    df = calculate_technical_indicators(df)

                    if df is not None:
                        # 선물 거래 신호 생성
                        signal = generate_futures_signals(df)

                        if signal:
                            all_signals.append(signal)

                            # 신호 출력
                            if signal['action'] == 'LONG':
                                action_emoji = "🟢"
                                action_text = "LONG"
                            elif signal['action'] == 'SHORT':
                                action_emoji = "🔴"
                                action_text = "SHORT"
                            else:
                                action_emoji = "🟡"
                                action_text = "HOLD"

                            confidence_emoji = "🔥" if signal['confidence'] == 'HIGH' else "⚡" if signal['confidence'] == 'MEDIUM' else "💤"

                            print(f"   {timeframe}: {action_emoji} {action_text} {confidence_emoji} (강도: {signal['strength']:.1f})")
                            print(f"      현재가: ${signal['current_price']:.8f}")

                            if signal['action'] != 'HOLD':
                                prices = signal['entry_exit_prices']
                                sizes = signal['position_sizes']
                                print(f"      진입가: ${prices['entry_price']:.8f}")
                                print(f"      스탑로스: ${prices['stop_loss']:.8f}")
                                print(f"      타겟가: ${prices['take_profit']:.8f}")
                                print(f"      포지션 크기: {sizes['quantity']:.4f} ({sizes['position_value']:.2f} USDT)")
                                print(f"      리스크/리워드: 1:{prices['risk_reward_ratio']:.1f}")

                            if signal['signals']:
                                print(f"      신호: {', '.join(signal['signals'][:3])}...")

            except Exception as e:
                print(f"   ❌ {timeframe} 분석 오류: {str(e)}")

    return all_signals

def generate_futures_report(signals):
    """선물 거래 리포트 생성"""
    print(f"\n📋 선물 거래 신호 리포트")
    print("=" * 80)

    if not signals:
        print("❌ 분석된 신호가 없습니다.")
        return

    # 액션별 분류
    long_signals = [s for s in signals if s['action'] == 'LONG']
    short_signals = [s for s in signals if s['action'] == 'SHORT']
    hold_signals = [s for s in signals if s['action'] == 'HOLD']

    print(f"📊 총 신호: {len(signals)}개")
    print(f"🟢 롱 신호: {len(long_signals)}개")
    print(f"🔴 숏 신호: {len(short_signals)}개")
    print(f"🟡 보유 신호: {len(hold_signals)}개")

    # 강한 신호만 필터링
    strong_long = [s for s in long_signals if s['confidence'] == 'HIGH']
    strong_short = [s for s in short_signals if s['confidence'] == 'HIGH']

    print(f"\n🔥 강한 롱 신호: {len(strong_long)}개")
    for signal in strong_long[:5]:  # 상위 5개 표시
        prices = signal['entry_exit_prices']
        sizes = signal['position_sizes']
        print(f"   {signal['symbol']} {signal['timeframe']}: ${signal['current_price']:.8f}")
        print(f"      진입: ${prices['entry_price']:.8f} | 스탑: ${prices['stop_loss']:.8f} | 타겟: ${prices['take_profit']:.8f}")
        print(f"      수량: {sizes['quantity']:.4f} | 리스크: ${sizes['risk_amount']:.2f}")

    print(f"\n🔥 강한 숏 신호: {len(strong_short)}개")
    for signal in strong_short[:5]:  # 상위 5개 표시
        prices = signal['entry_exit_prices']
        sizes = signal['position_sizes']
        print(f"   {signal['symbol']} {signal['timeframe']}: ${signal['current_price']:.8f}")
        print(f"      진입: ${prices['entry_price']:.8f} | 스탑: ${prices['stop_loss']:.8f} | 타겟: ${prices['take_profit']:.8f}")
        print(f"      수량: {sizes['quantity']:.4f} | 리스크: ${sizes['risk_amount']:.2f}")

    # 심볼별 요약
    print(f"\n📈 심볼별 요약:")
    symbols = list(set([s['symbol'] for s in signals]))

    for symbol in symbols:
        symbol_signals = [s for s in signals if s['symbol'] == symbol]
        long_count = len([s for s in symbol_signals if s['action'] == 'LONG'])
        short_count = len([s for s in symbol_signals if s['action'] == 'SHORT'])

        if long_count > short_count:
            sentiment = "🟢 강세"
        elif short_count > long_count:
            sentiment = "🔴 약세"
        else:
            sentiment = "🟡 중립"

        print(f"   {symbol}: {sentiment} (롱: {long_count}, 숏: {short_count})")

def main():
    """메인 실행 함수"""
    print("🚀 암호화폐 선물 트레이딩 신호 분석기 (레버리지 20배)")
    print("=" * 80)

    # 환경변수 로드
    load_env_file()

    # 분석할 심볼과 타임프레임 (1분봉 포함)
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    timeframes = ['1m', '5m', '15m', '1h']  # 선물 거래용 타임프레임

    print(f"📊 분석 대상: {', '.join(symbols)}")
    print(f"⏱️  타임프레임: {', '.join(timeframes)}")
    print(f"⚡ 레버리지: 20배")
    print(f"💰 리스크 관리: 계좌의 2%")

    # 선물 시장 분석
    signals = analyze_futures_market(symbols, timeframes)

    # 선물 거래 리포트 생성
    generate_futures_report(signals)

    print(f"\n🎉 선물 거래 분석 완료!")
    print("⚠️  주의사항:")
    print("   - 레버리지 20배는 매우 위험합니다")
    print("   - 스탑로스를 반드시 설정하세요")
    print("   - 리스크 관리가 최우선입니다")
    print("   - 이 신호는 참고용이며, 실제 거래는 신중하게 결정하세요")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
