#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 암호화폐 선물 트레이딩 AI 시스템
🎯 목표: 80% 이상 수익률 달성

기능:
1. 실시간 데이터 수집 (Binance WebSocket)
2. ML/DL 기반 신호 생성 (Random Forest, XGBoost, LightGBM, LSTM)
3. Google Colab A100 GPU 훈련
4. 선물 거래 신호 (롱/숏 포지션)
5. 자동 백테스팅 및 최적화
"""

import os
import sys
import time
import logging
import argparse
import colorama
from colorama import Fore, Back, Style
from pathlib import Path
import threading
from datetime import datetime

# Colorama 초기화
colorama.init()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_system.log'),
        logging.StreamHandler()
    ]
)

class CryptoTradingSystem:
    def __init__(self):
        self.is_running = False
        self.processes = {}

        # 시스템 구성 요소
        self.components = {
            'data_collector': 'realtime_data_collector.py',
            'trading_signals': 'realtime_trading_signals.py',
            'ml_optimizer': 'ml_trading_optimizer.py',
            'colab_training': 'colab_training_system.py',
            'colab_api': 'colab_api_controller.py',
            'continuous_opt': 'continuous_optimization_system.py',
            'progress_monitor': 'real_colab_monitor.py'
        }

        # 시스템 상태
        self.system_status = {
            'data_collection': False,
            'signal_generation': False,
            'ml_training': False,
            'colab_integration': False,
            'optimization': False
        }

    def display_banner(self):
        """시스템 배너 표시"""
        banner = f"""
{Fore.CYAN}{'='*80}
{Fore.YELLOW}🚀 암호화폐 선물 트레이딩 AI 시스템
{Fore.CYAN}{'='*80}
{Fore.GREEN}🎯 목표: 80% 이상 수익률 달성
{Fore.GREEN}💰 거래 유형: 선물 (20배 레버리지)
{Fore.GREEN}📊 분석 타임프레임: 1분, 5분, 15분, 1시간
{Fore.GREEN}🤖 ML 모델: Random Forest, XGBoost, LightGBM, LSTM
{Fore.GREEN}☁️ GPU: Google Colab A100
{Fore.CYAN}{'='*80}
        """
        print(banner)

    def check_system_requirements(self):
        """시스템 요구사항 확인"""
        print(f"{Fore.YELLOW}🔍 시스템 요구사항 확인 중...")

        # 필수 파일 확인
        missing_files = []
        for component, filename in self.components.items():
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

    def check_data_availability(self):
        """데이터 가용성 확인"""
        print(f"{Fore.YELLOW}📊 데이터 가용성 확인 중...")

        try:
            from supabase import create_client
            import os

            # 환경변수 로드
            from pathlib import Path
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

    def start_data_collection(self):
        """실시간 데이터 수집 시작"""
        print(f"{Fore.YELLOW}📡 실시간 데이터 수집 시작...")

        try:
            import subprocess
            process = subprocess.Popen([
                sys.executable, 'realtime_data_collector.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['data_collector'] = process
            self.system_status['data_collection'] = True

            print(f"{Fore.GREEN}✅ 실시간 데이터 수집 시작됨!")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 데이터 수집 시작 실패: {str(e)}")
            return False

    def start_signal_generation(self):
        """실시간 신호 생성 시작"""
        print(f"{Fore.YELLOW}📊 실시간 신호 생성 시작...")

        try:
            import subprocess
            process = subprocess.Popen([
                sys.executable, 'realtime_trading_signals.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['signal_generator'] = process
            self.system_status['signal_generation'] = True

            print(f"{Fore.GREEN}✅ 실시간 신호 생성 시작됨!")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 신호 생성 시작 실패: {str(e)}")
            return False

    def start_ml_training(self):
        """ML 모델 훈련 시작"""
        print(f"{Fore.YELLOW}🤖 ML 모델 훈련 시작...")

        # Colab API 컨트롤러 시작
        try:
            import subprocess
            process = subprocess.Popen([
                sys.executable, 'colab_api_controller.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['ml_training'] = process
            self.system_status['ml_training'] = True

            print(f"{Fore.GREEN}✅ ML 모델 훈련 시작됨!")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ ML 훈련 시작 실패: {str(e)}")
            return False

    def start_progress_monitoring(self):
        """진행 상황 모니터링 시작"""
        print(f"{Fore.YELLOW}📈 진행 상황 모니터링 시작...")

        try:
            import subprocess
            process = subprocess.Popen([
                sys.executable, 'real_colab_monitor.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['progress_monitor'] = process

            print(f"{Fore.GREEN}✅ 진행 상황 모니터링 시작됨!")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 모니터링 시작 실패: {str(e)}")
            return False

    def start_continuous_optimization(self):
        """지속적 최적화 시작"""
        print(f"{Fore.YELLOW}⚙️ 지속적 최적화 시작...")

        try:
            import subprocess
            process = subprocess.Popen([
                sys.executable, 'continuous_optimization_system.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['continuous_opt'] = process
            self.system_status['optimization'] = True

            print(f"{Fore.GREEN}✅ 지속적 최적화 시작됨!")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 최적화 시작 실패: {str(e)}")
            return False

    def display_system_status(self):
        """시스템 상태 표시"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}📊 시스템 상태")
        print(f"{Fore.CYAN}{'='*60}")

        for component, status in self.system_status.items():
            status_icon = "✅" if status else "❌"
            status_text = "실행 중" if status else "중지됨"
            print(f"{Fore.WHITE}{component.replace('_', ' ').title()}: {status_icon} {status_text}")

        print(f"{Fore.CYAN}{'='*60}")

    def start_full_system(self):
        """전체 시스템 시작"""
        print(f"{Fore.GREEN}🚀 전체 시스템 시작 중...")

        # 1. 시스템 요구사항 확인
        if not self.check_system_requirements():
            print(f"{Fore.RED}❌ 시스템 요구사항 확인 실패!")
            return False

        # 2. 데이터 가용성 확인
        if not self.check_data_availability():
            print(f"{Fore.YELLOW}⚠️ 데이터가 없습니다. 데이터 수집부터 시작합니다.")

        # 3. 실시간 데이터 수집 시작
        if not self.start_data_collection():
            print(f"{Fore.RED}❌ 데이터 수집 시작 실패!")
            return False

        # 잠시 대기
        time.sleep(3)

        # 4. 실시간 신호 생성 시작
        if not self.start_signal_generation():
            print(f"{Fore.RED}❌ 신호 생성 시작 실패!")
            return False

        # 잠시 대기
        time.sleep(3)

        # 5. ML 모델 훈련 시작
        if not self.start_ml_training():
            print(f"{Fore.RED}❌ ML 훈련 시작 실패!")
            return False

        # 6. 진행 상황 모니터링 시작
        if not self.start_progress_monitoring():
            print(f"{Fore.RED}❌ 모니터링 시작 실패!")
            return False

        # 7. 지속적 최적화 시작
        if not self.start_continuous_optimization():
            print(f"{Fore.RED}❌ 최적화 시작 실패!")
            return False

        self.is_running = True

        print(f"\n{Fore.GREEN}🎉 전체 시스템 시작 완료!")
        print(f"{Fore.YELLOW}📊 실시간 모니터링: tail -f main_system.log")
        print(f"{Fore.YELLOW}🛑 시스템 중지: Ctrl+C")

        return True

    def stop_system(self):
        """시스템 중지"""
        print(f"\n{Fore.YELLOW}⚠️ 시스템 중지 중...")

        self.is_running = False

        # 모든 프로세스 종료
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"{Fore.GREEN}✅ {name} 프로세스 종료됨")
            except:
                try:
                    process.kill()
                    print(f"{Fore.RED}❌ {name} 프로세스 강제 종료됨")
                except:
                    pass

        print(f"{Fore.GREEN}✅ 시스템 중지 완료!")

    def run_interactive_mode(self):
        """대화형 모드 실행"""
        self.display_banner()

        while True:
            print(f"\n{Fore.CYAN}메뉴 선택:")
            print(f"{Fore.WHITE}1. 전체 시스템 시작")
            print(f"{Fore.WHITE}2. 데이터 수집만 시작")
            print(f"{Fore.WHITE}3. 신호 생성만 시작")
            print(f"{Fore.WHITE}4. ML 훈련만 시작")
            print(f"{Fore.WHITE}5. 시스템 상태 확인")
            print(f"{Fore.WHITE}6. 시스템 중지")
            print(f"{Fore.WHITE}0. 종료")

            choice = input(f"\n{Fore.YELLOW}선택 (0-6): ").strip()

            if choice == '0':
                print(f"{Fore.GREEN}👋 시스템을 종료합니다.")
                break
            elif choice == '1':
                self.start_full_system()
            elif choice == '2':
                self.start_data_collection()
            elif choice == '3':
                self.start_signal_generation()
            elif choice == '4':
                self.start_ml_training()
            elif choice == '5':
                self.display_system_status()
            elif choice == '6':
                self.stop_system()
            else:
                print(f"{Fore.RED}❌ 잘못된 선택입니다.")

    def run_auto_mode(self):
        """자동 모드 실행"""
        self.display_banner()

        if self.start_full_system():
            try:
                # 메인 루프
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_system()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='암호화폐 선물 트레이딩 AI 시스템')
    parser.add_argument('--mode', choices=['interactive', 'auto'], default='interactive',
                       help='실행 모드 (interactive: 대화형, auto: 자동)')

    args = parser.parse_args()

    system = CryptoTradingSystem()

    try:
        if args.mode == 'interactive':
            system.run_interactive_mode()
        else:
            system.run_auto_mode()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}사용자에 의해 중단되었습니다.")
        system.stop_system()
    except Exception as e:
        print(f"{Fore.RED}❌ 시스템 오류: {str(e)}")
        system.stop_system()

if __name__ == "__main__":
    main()
