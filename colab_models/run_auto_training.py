#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 AI 자동 트레이딩 봇 최적화 시스템
메인 실행 스크립트

이 스크립트는 다음을 수행합니다:
1. 자동 하이퍼파라미터 튜닝
2. 모델 성능 평가
3. 트레이딩 신호 생성
4. 결과 시각화 및 저장
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from auto_hyperparameter_tuner import AutoTradingBot
from model_evaluator import ModelEvaluator
from trading_signal_generator import TradingSignalGenerator

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='AI 자동 트레이딩 봇 최적화 시스템')
    parser.add_argument('--supabase-url', required=True, help='Supabase URL')
    parser.add_argument('--supabase-key', required=True, help='Supabase API Key')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'signal', 'all'],
                       default='all', help='실행 모드')
    parser.add_argument('--symbols', nargs='+',
                       default=['ADAUSDT', 'BTCUSDT', 'ETHUSDT'],
                       help='트레이딩 심볼들')
    parser.add_argument('--timeframes', nargs='+',
                       default=['1m', '5m', '15m'],
                       help='타임프레임들')
    parser.add_argument('--model-types', nargs='+',
                       default=['lightgbm', 'xgboost'],
                       help='모델 타입들')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='하이퍼파라미터 최적화 시도 횟수')
    parser.add_argument('--output-dir', default='output',
                       help='출력 디렉토리')

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 모델 및 결과 디렉토리 생성
    models_dir = output_dir / 'models'
    results_dir = output_dir / 'results'
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print('🚀 AI 자동 트레이딩 봇 최적화 시스템 시작')
    print('=' * 60)
    print(f'📅 시작 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'🎯 심볼: {args.symbols}')
    print(f'⏰ 타임프레임: {args.timeframes}')
    print(f'🤖 모델 타입: {args.model_types}')
    print(f'🔄 최적화 시도: {args.n_trials}회')
    print(f'📁 출력 디렉토리: {output_dir}')
    print('=' * 60)

    try:
        if args.mode in ['train', 'all']:
            print('\n🔄 1단계: 모델 훈련 및 최적화')
            run_training(args, models_dir, results_dir)

        if args.mode in ['evaluate', 'all']:
            print('\n🔄 2단계: 모델 성능 평가')
            run_evaluation(args, models_dir, results_dir, output_dir)

        if args.mode in ['signal', 'all']:
            print('\n🔄 3단계: 트레이딩 신호 생성')
            run_signal_generation(args, models_dir, results_dir, output_dir)

        print('\n🎉 모든 작업 완료!')
        print(f'📁 결과 확인: {output_dir}')

    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        sys.exit(1)

def run_training(args, models_dir, results_dir):
    """모델 훈련 및 최적화 실행"""
    print('🚀 자동 하이퍼파라미터 튜닝 시작')

    # 자동 트레이딩 봇 생성
    bot = AutoTradingBot(args.supabase_url, args.supabase_key)

    # 결과 저장용 딕셔너리
    all_results = {}

    # 각 조합에 대해 최적화 실행
    for symbol in args.symbols:
        all_results[symbol] = {}

        for timeframe in args.timeframes:
            all_results[symbol][timeframe] = {}

            for model_type in args.model_types:
                print(f'\n🔄 {symbol} {timeframe} {model_type.upper()} 최적화 중...')

                model_name, metrics, best_params = bot.run_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    n_trials=args.n_trials
                )

                if model_name:
                    all_results[symbol][timeframe][model_type] = {
                        'model_name': model_name,
                        'metrics': metrics,
                        'best_params': best_params
                    }
                    print(f'✅ {model_name} 완료!')
                else:
                    print(f'❌ {symbol} {timeframe} {model_type} 실패')
                    all_results[symbol][timeframe][model_type] = None

    # 훈련 결과 저장
    training_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'symbols': args.symbols,
            'timeframes': args.timeframes,
            'model_types': args.model_types,
            'n_trials': args.n_trials
        },
        'results': all_results
    }

    summary_path = results_dir / f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)

    print(f'💾 훈련 결과 저장: {summary_path}')
    return all_results

def run_evaluation(args, models_dir, results_dir, output_dir):
    """모델 성능 평가 실행"""
    print('📊 모델 성능 평가 시작')

    # 모델 평가기 생성
    evaluator = ModelEvaluator(str(models_dir), str(results_dir))

    # 전체 평가 실행
    evaluation_results = evaluator.run_full_evaluation()

    # 평가 결과 저장
    evaluation_summary = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_results': evaluation_results
    }

    summary_path = output_dir / f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

    print(f'💾 평가 결과 저장: {summary_path}')
    return evaluation_results

def run_signal_generation(args, models_dir, results_dir, output_dir):
    """트레이딩 신호 생성 실행"""
    print('🚀 트레이딩 신호 생성 시작')

    # 트레이딩 신호 생성기 생성
    signal_generator = TradingSignalGenerator(
        args.supabase_url,
        args.supabase_key,
        str(models_dir),
        str(results_dir)
    )

    # 신호 생성 실행
    signals = signal_generator.run_signal_generation()

    if signals:
        # 신호 결과 저장
        signal_summary = {
            'timestamp': datetime.now().isoformat(),
            'signals': signals
        }

        summary_path = output_dir / f'signal_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(signal_summary, f, indent=2, ensure_ascii=False)

        print(f'💾 신호 결과 저장: {summary_path}')

        # 신호 요약 출력
        print('\n📊 신호 요약:')
        for key, signal_data in signals.items():
            ensemble = signal_data['ensemble']
            print(f"{key}: {ensemble['ensemble_signal_strength']} (매수확률: {ensemble['ensemble_buy_probability']:.1%})")

    return signals

def create_config_file():
    """설정 파일 생성"""
    config = {
        "supabase": {
            "url": "https://your-project.supabase.co",
            "key": "your-anon-key"
        },
        "training": {
            "symbols": ["ADAUSDT", "BTCUSDT", "ETHUSDT"],
            "timeframes": ["1m", "5m", "15m"],
            "model_types": ["lightgbm", "xgboost"],
            "n_trials": 100
        },
        "output": {
            "directory": "output"
        }
    }

    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'📝 설정 파일 생성: {config_path}')
    print('설정 파일을 편집한 후 다시 실행하세요.')

if __name__ == "__main__":
    # 설정 파일이 없으면 생성
    config_path = Path(__file__).parent / 'config.json'
    if not config_path.exists():
        print('📝 설정 파일이 없습니다. 생성 중...')
        create_config_file()
        sys.exit(0)

    main()

