#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 암호화폐 선물 트레이딩 AI 시스템 - 테스트 버전
"""

import os
import sys
import time
import logging
import colorama
from colorama import Fore, Back, Style
from pathlib import Path

# Colorama 초기화
colorama.init()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_system.log'),
        logging.StreamHandler()
    ]
)

def display_banner():
    """시스템 배너 표시"""
    banner = f"""
{Fore.CYAN}{'='*80}
{Fore.YELLOW}🚀 암호화폐 선물 트레이딩 AI 시스템 - 테스트 버전
{Fore.CYAN}{'='*80}
{Fore.GREEN}🎯 목표: 80% 이상 수익률 달성
{Fore.GREEN}💰 거래 유형: 선물 (20배 레버리지)
{Fore.GREEN}📊 분석 타임프레임: 1분, 5분, 15분, 1시간
{Fore.GREEN}🤖 ML 모델: Random Forest, XGBoost, LightGBM, LSTM
{Fore.GREEN}☁️ GPU: Google Colab A100
{Fore.CYAN}{'='*80}
    """
    print(banner)

def check_system_requirements():
    """시스템 요구사항 확인"""
    print(f"{Fore.YELLOW}🔍 시스템 요구사항 확인 중...")

    # 필수 파일 확인
    required_files = [
        'realtime_data_collector.py',
        'realtime_trading_signals.py',
        'ml_trading_optimizer.py',
        'colab_training_system.py',
        'colab_api_controller.py',
        'continuous_optimization_system.py',
        'real_colab_monitor.py'
    ]

    missing_files = []
    for filename in required_files:
        if not Path(filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"{Fore.RED}❌ 누락된 파일들:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # 환경변수 확인
    if not Path('.env').exists():
        print(f"{Fore.RED}❌ .env 파일이 없습니다!")
        return False

    # 디렉토리 확인
    required_dirs = ['colab_models', 'models', 'logs']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

    print(f"{Fore.GREEN}✅ 시스템 요구사항 확인 완료!")
    return True

def check_data_availability():
    """데이터 가용성 확인"""
    print(f"{Fore.YELLOW}📊 데이터 가용성 확인 중...")

    try:
        from supabase import create_client
        import os

        # 환경변수 로드
        config_file = Path('.env')
        if config_file.exists():
            with open(config_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')

        if not url or not key:
            print(f"{Fore.RED}❌ Supabase 연결 정보가 없습니다!")
            return False

        supabase = create_client(url, key)

        # 데이터 확인
        response = supabase.table('crypto_ohlcv').select('*').limit(1).execute()

        if response.data:
            print(f"{Fore.GREEN}✅ 데이터베이스 연결 성공!")
            return True
        else:
            print(f"{Fore.YELLOW}⚠️ 데이터베이스에 데이터가 없습니다.")
            return False

    except Exception as e:
        print(f"{Fore.RED}❌ 데이터 확인 오류: {str(e)}")
        return False

def test_individual_components():
    """개별 컴포넌트 테스트"""
    print(f"{Fore.YELLOW}🧪 개별 컴포넌트 테스트 중...")

    # 1. 실시간 데이터 수집 테스트
    print(f"{Fore.CYAN}📡 실시간 데이터 수집 테스트...")
    try:
        import subprocess
        process = subprocess.Popen([
            sys.executable, 'realtime_data_collector.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 5초 대기 후 종료
        time.sleep(5)
        process.terminate()
        process.wait(timeout=5)

        print(f"{Fore.GREEN}✅ 실시간 데이터 수집 테스트 성공!")

    except Exception as e:
        print(f"{Fore.RED}❌ 실시간 데이터 수집 테스트 실패: {str(e)}")

    # 2. 실시간 신호 생성 테스트
    print(f"{Fore.CYAN}📊 실시간 신호 생성 테스트...")
    try:
        import subprocess
        process = subprocess.Popen([
            sys.executable, 'realtime_trading_signals.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 5초 대기 후 종료
        time.sleep(5)
        process.terminate()
        process.wait(timeout=5)

        print(f"{Fore.GREEN}✅ 실시간 신호 생성 테스트 성공!")

    except Exception as e:
        print(f"{Fore.RED}❌ 실시간 신호 생성 테스트 실패: {str(e)}")

    # 3. ML 훈련 시스템 테스트
    print(f"{Fore.CYAN}🤖 ML 훈련 시스템 테스트...")
    try:
        import subprocess
        process = subprocess.Popen([
            sys.executable, 'ml_trading_optimizer.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 5초 대기 후 종료
        time.sleep(5)
        process.terminate()
        process.wait(timeout=5)

        print(f"{Fore.GREEN}✅ ML 훈련 시스템 테스트 성공!")

    except Exception as e:
        print(f"{Fore.RED}❌ ML 훈련 시스템 테스트 실패: {str(e)}")

def display_system_status():
    """시스템 상태 표시"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}📊 시스템 상태")
    print(f"{Fore.CYAN}{'='*60}")

    # 파일 개수 확인
    python_files = len(list(Path('.').glob('*.py')))
    colab_notebooks = len(list(Path('colab_models').glob('*.ipynb')))

    print(f"{Fore.WHITE}Python 파일: {Fore.GREEN}{python_files}개")
    print(f"{Fore.WHITE}Colab 노트북: {Fore.GREEN}{colab_notebooks}개")
    print(f"{Fore.WHITE}데이터베이스: {Fore.GREEN}연결됨")
    print(f"{Fore.WHITE}가상환경: {Fore.GREEN}활성화됨")

    print(f"{Fore.CYAN}{'='*60}")

def main():
    """메인 함수"""
    print(f"{Fore.CYAN}🚀 암호화폐 선물 트레이딩 AI 시스템 테스트")
    print(f"{Fore.CYAN}{'='*60}")

    try:
        # 1. 배너 표시
        display_banner()

        # 2. 시스템 요구사항 확인
        if not check_system_requirements():
            print(f"{Fore.RED}❌ 시스템 요구사항 확인 실패!")
            return

        # 3. 데이터 가용성 확인
        if not check_data_availability():
            print(f"{Fore.YELLOW}⚠️ 데이터가 없습니다. 데이터 수집부터 시작합니다.")

        # 4. 시스템 상태 표시
        display_system_status()

        # 5. 개별 컴포넌트 테스트
        test_individual_components()

        print(f"\n{Fore.GREEN}🎉 시스템 테스트 완료!")
        print(f"{Fore.YELLOW}📊 로그 확인: tail -f test_system.log")
        print(f"{Fore.YELLOW}🚀 전체 시스템 시작: python main.py")

    except Exception as e:
        print(f"{Fore.RED}❌ 시스템 테스트 오류: {str(e)}")

if __name__ == "__main__":
    main()
