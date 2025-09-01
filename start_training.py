#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 암호화폐 선물 트레이딩 AI 시스템 - 학습 시작
"""

import os
import sys
import time
import logging
import colorama
from colorama import Fore, Back, Style
from pathlib import Path
import subprocess
import threading

# Colorama 초기화
colorama.init()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_start.log'),
        logging.StreamHandler()
    ]
)

class TrainingStarter:
    def __init__(self):
        self.is_running = False
        self.processes = {}

    def display_banner(self):
        """시스템 배너 표시"""
        banner = f"""
{Fore.CYAN}{'='*80}
{Fore.YELLOW}🚀 암호화폐 선물 트레이딩 AI 시스템 - 학습 시작
{Fore.CYAN}{'='*80}
{Fore.GREEN}🎯 목표: 80% 이상 수익률 달성
{Fore.GREEN}💰 거래 유형: 선물 (20배 레버리지)
{Fore.GREEN}📊 분석 타임프레임: 1분, 5분, 15분, 1시간
{Fore.GREEN}🤖 ML 모델: Random Forest, XGBoost, LightGBM, LSTM
{Fore.GREEN}☁️ GPU: Google Colab A100
{Fore.CYAN}{'='*80}
        """
        print(banner)

    def start_data_collection(self):
        """실시간 데이터 수집 시작"""
        print(f"{Fore.YELLOW}📡 실시간 데이터 수집 시작...")

        try:
            process = subprocess.Popen([
                sys.executable, 'realtime_data_collector.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['data_collector'] = process
            print(f"{Fore.GREEN}✅ 실시간 데이터 수집 시작됨! (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 데이터 수집 시작 실패: {str(e)}")
            return False

    def start_signal_generation(self):
        """실시간 신호 생성 시작"""
        print(f"{Fore.YELLOW}📊 실시간 신호 생성 시작...")

        try:
            process = subprocess.Popen([
                sys.executable, 'realtime_trading_signals.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['signal_generator'] = process
            print(f"{Fore.GREEN}✅ 실시간 신호 생성 시작됨! (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 신호 생성 시작 실패: {str(e)}")
            return False

    def start_ml_training(self):
        """ML 모델 훈련 시작"""
        print(f"{Fore.YELLOW}🤖 ML 모델 훈련 시작...")

        try:
            process = subprocess.Popen([
                sys.executable, 'ml_trading_optimizer.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['ml_training'] = process
            print(f"{Fore.GREEN}✅ ML 모델 훈련 시작됨! (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ ML 훈련 시작 실패: {str(e)}")
            return False

    def start_progress_monitoring(self):
        """진행 상황 모니터링 시작"""
        print(f"{Fore.YELLOW}📈 진행 상황 모니터링 시작...")

        try:
            process = subprocess.Popen([
                sys.executable, 'real_colab_monitor.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['progress_monitor'] = process
            print(f"{Fore.GREEN}✅ 진행 상황 모니터링 시작됨! (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 모니터링 시작 실패: {str(e)}")
            return False

    def start_continuous_optimization(self):
        """지속적 최적화 시작"""
        print(f"{Fore.YELLOW}⚙️ 지속적 최적화 시작...")

        try:
            process = subprocess.Popen([
                sys.executable, 'continuous_optimization_system.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['continuous_opt'] = process
            print(f"{Fore.GREEN}✅ 지속적 최적화 시작됨! (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"{Fore.RED}❌ 최적화 시작 실패: {str(e)}")
            return False

    def display_system_status(self):
        """시스템 상태 표시"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}📊 실행 중인 프로세스")
        print(f"{Fore.CYAN}{'='*60}")

        for name, process in self.processes.items():
            if process.poll() is None:  # 프로세스가 실행 중
                print(f"{Fore.GREEN}✅ {name}: 실행 중 (PID: {process.pid})")
            else:
                print(f"{Fore.RED}❌ {name}: 종료됨")

        print(f"{Fore.CYAN}{'='*60}")

    def monitor_processes(self):
        """프로세스 모니터링"""
        while self.is_running:
            try:
                # 프로세스 상태 확인
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:  # 프로세스가 종료됨
                        print(f"{Fore.YELLOW}⚠️ {name} 프로세스가 종료되었습니다.")
                        del self.processes[name]

                time.sleep(10)  # 10초마다 확인

            except Exception as e:
                logging.error(f"프로세스 모니터링 오류: {str(e)}")
                time.sleep(30)

    def start_full_training(self):
        """전체 훈련 시작"""
        print(f"{Fore.GREEN}🚀 전체 훈련 시스템 시작 중...")

        self.is_running = True

        # 1. 실시간 데이터 수집 시작
        if not self.start_data_collection():
            print(f"{Fore.RED}❌ 데이터 수집 시작 실패!")
            return False

        time.sleep(3)

        # 2. 실시간 신호 생성 시작
        if not self.start_signal_generation():
            print(f"{Fore.RED}❌ 신호 생성 시작 실패!")
            return False

        time.sleep(3)

        # 3. ML 모델 훈련 시작
        if not self.start_ml_training():
            print(f"{Fore.RED}❌ ML 훈련 시작 실패!")
            return False

        time.sleep(3)

        # 4. 진행 상황 모니터링 시작
        if not self.start_progress_monitoring():
            print(f"{Fore.RED}❌ 모니터링 시작 실패!")
            return False

        time.sleep(3)

        # 5. 지속적 최적화 시작
        if not self.start_continuous_optimization():
            print(f"{Fore.RED}❌ 최적화 시작 실패!")
            return False

        print(f"\n{Fore.GREEN}🎉 전체 훈련 시스템 시작 완료!")
        print(f"{Fore.YELLOW}📊 실행 중인 프로세스:")

        for name, process in self.processes.items():
            print(f"   - {name}: PID {process.pid}")

        print(f"\n{Fore.CYAN}📋 모니터링 명령어:")
        print(f"{Fore.WHITE}   실시간 로그: tail -f training_start.log")
        print(f"{Fore.WHITE}   데이터 수집: tail -f realtime_data.log")
        print(f"{Fore.WHITE}   신호 생성: tail -f realtime_signals.log")
        print(f"{Fore.WHITE}   ML 훈련: tail -f ml_trading.log")
        print(f"{Fore.WHITE}   시스템 중지: Ctrl+C")

        # 프로세스 모니터링 스레드 시작
        monitor_thread = threading.Thread(target=self.monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()

        return True

    def stop_all_processes(self):
        """모든 프로세스 중지"""
        print(f"\n{Fore.YELLOW}⚠️ 모든 프로세스 중지 중...")

        self.is_running = False

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

        print(f"{Fore.GREEN}✅ 모든 프로세스 중지 완료!")

def main():
    """메인 함수"""
    starter = TrainingStarter()

    try:
        # 배너 표시
        starter.display_banner()

        # 전체 훈련 시작
        if starter.start_full_training():
            try:
                # 메인 루프
                while starter.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}사용자에 의해 중단되었습니다.")
                starter.stop_all_processes()

    except Exception as e:
        print(f"{Fore.RED}❌ 시스템 오류: {str(e)}")
        starter.stop_all_processes()

if __name__ == "__main__":
    main()
