import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
import requests
import colorama
from colorama import Fore, Back, Style
import psutil

# Colorama 초기화
colorama.init()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('colab_monitor.log'),
        logging.StreamHandler()
    ]
)

class RealColabMonitor:
    def __init__(self):
        self.is_running = False

        # Colab 훈련 설정
        self.colab_config = {
            'symbols': ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt'],
            'timeframes': ['1m', '5m', '15m', '1h'],
            'models': ['random_forest', 'xgboost', 'lightgbm', 'lstm'],
            'total_models': 80,
            'estimated_times': {
                'random_forest': 5,    # 분
                'xgboost': 8,          # 분
                'lightgbm': 6,         # 분
                'lstm': 45             # 분
            }
        }

        # 진행 상황 추적
        self.progress = {
            'start_time': None,
            'completed_models': 0,
            'current_model': None,
            'current_step': None,
            'colab_status': {},
            'errors': [],
            'successes': [],
            'estimated_completion': None
        }

        # Colab 노트북 상태 추적
        self.notebook_status = {}

        # 모델 저장 경로
        self.model_path = Path('colab_models')
        self.models_path = Path('models')

    def calculate_total_estimated_time(self):
        """총 예상 시간 계산"""
        total_minutes = 0

        for symbol in self.colab_config['symbols']:
            for timeframe in self.colab_config['timeframes']:
                for model_type in self.colab_config['models']:
                    total_minutes += self.colab_config['estimated_times'][model_type]

        return total_minutes

    def format_time(self, minutes):
        """시간 포맷팅"""
        if minutes < 60:
            return f"{minutes:.0f}분"
        elif minutes < 1440:  # 24시간
            hours = minutes / 60
            return f"{hours:.1f}시간"
        else:
            days = minutes / 1440
            return f"{days:.1f}일"

    def check_colab_notebooks(self):
        """Colab 노트북 파일 확인"""
        try:
            notebooks = list(self.model_path.glob('*.ipynb'))
            return len(notebooks)
        except Exception as e:
            logging.error(f"노트북 확인 오류: {str(e)}")
            return 0

    def check_trained_models(self):
        """훈련된 모델 파일 확인"""
        try:
            model_files = list(self.models_path.glob('*_model.*'))
            return len(model_files)
        except Exception as e:
            logging.error(f"모델 파일 확인 오류: {str(e)}")
            return 0

    def get_colab_status(self):
        """Colab 상태 확인 (시뮬레이션)"""
        try:
            # 실제로는 Colab API를 통해 상태 확인
            # 여기서는 파일 시스템 기반으로 시뮬레이션

            total_notebooks = self.check_colab_notebooks()
            total_models = self.check_trained_models()

            # 진행률 계산
            if total_notebooks > 0:
                progress_percent = (total_models / self.colab_config['total_models']) * 100
            else:
                progress_percent = 0

            return {
                'total_notebooks': total_notebooks,
                'total_models': total_models,
                'progress_percent': progress_percent,
                'status': 'running' if total_models < self.colab_config['total_models'] else 'completed'
            }

        except Exception as e:
            logging.error(f"Colab 상태 확인 오류: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def display_progress_header(self):
        """진행 상황 헤더 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}🤖 Google Colab ML 모델 훈련 실시간 모니터링")
        print(f"{Fore.CYAN}{'='*80}")

        total_time = self.calculate_total_estimated_time()
        print(f"{Fore.GREEN}📊 총 훈련 모델: {self.colab_config['total_models']}개")
        print(f"{Fore.GREEN}⏰ 예상 총 소요 시간: {self.format_time(total_time)}")
        print(f"{Fore.GREEN}📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.CYAN}{'='*80}\n")

    def display_current_status(self):
        """현재 상태 표시"""
        # Colab 상태 확인
        colab_status = self.get_colab_status()

        # 화면 클리어
        os.system('clear' if os.name == 'posix' else 'cls')

        # 헤더 표시
        self.display_progress_header()

        # Colab 상태 표시
        print(f"{Fore.YELLOW}🔄 Colab 훈련 상태:")
        print(f"   📝 생성된 노트북: {Fore.WHITE}{colab_status['total_notebooks']}개")
        print(f"   ✅ 완료된 모델: {Fore.WHITE}{colab_status['total_models']}개")
        print(f"   📈 진행률: {Fore.WHITE}{colab_status['progress_percent']:.1f}%")
        print(f"   🎯 상태: {Fore.WHITE}{colab_status['status'].upper()}")

        # 예상 완료 시간 계산
        if colab_status['total_models'] > 0 and self.progress['start_time']:
            current_time = datetime.now()
            elapsed_time = current_time - self.progress['start_time']

            avg_time_per_model = elapsed_time.total_seconds() / colab_status['total_models'] / 60  # 분
            remaining_models = self.colab_config['total_models'] - colab_status['total_models']
            remaining_minutes = remaining_models * avg_time_per_model

            if remaining_minutes > 0:
                estimated_completion = current_time + timedelta(minutes=remaining_minutes)
                print(f"   ⏰ 예상 완료: {Fore.WHITE}{estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   ⏱️ 남은 시간: {Fore.WHITE}{self.format_time(remaining_minutes)}")

        # 진행률 바
        print(f"\n{Fore.GREEN}📈 전체 진행률:")
        bar_length = 50
        filled_length = int(bar_length * colab_status['progress_percent'] / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"   {bar} {colab_status['progress_percent']:.1f}%")
        print(f"   ✅ 완료: {colab_status['total_models']}/{self.colab_config['total_models']} 모델")

        # 시스템 리소스 표시
        print(f"\n{Fore.BLUE}💻 시스템 리소스:")
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"   🖥️ CPU: {Fore.WHITE}{cpu_usage:.1f}%")
        print(f"   💾 메모리: {Fore.WHITE}{memory.percent:.1f}%")

        # 다음 단계 안내
        print(f"\n{Fore.MAGENTA}🚀 다음 단계:")
        if colab_status['total_notebooks'] == 0:
            print(f"   1. {Fore.WHITE}Colab 노트북 생성: python colab_training_system.py")
        elif colab_status['total_models'] == 0:
            print(f"   1. {Fore.WHITE}Google Colab에 노트북 업로드")
            print(f"   2. {Fore.WHITE}Supabase 연결 정보 입력")
            print(f"   3. {Fore.WHITE}모든 셀 실행")
        elif colab_status['total_models'] < self.colab_config['total_models']:
            print(f"   1. {Fore.WHITE}Colab에서 훈련 진행 중...")
            print(f"   2. {Fore.WHITE}완료된 모델 파일 다운로드")
        else:
            print(f"   1. {Fore.WHITE}🎉 모든 모델 훈련 완료!")
            print(f"   2. {Fore.WHITE}실시간 신호 생성 시작: python realtime_trading_signals.py")

        # 최근 활동 표시
        print(f"\n{Fore.CYAN}📋 최근 활동:")
        if self.progress['successes']:
            for success in self.progress['successes'][-3:]:
                print(f"   ✅ {success}")

        if self.progress['errors']:
            for error in self.progress['errors'][-3:]:
                print(f"   ❌ {error}")

        print(f"\n{Fore.CYAN}{'='*80}")

    def monitor_colab_progress(self):
        """Colab 진행 상황 모니터링"""
        while self.is_running:
            try:
                # 이전 상태 저장
                prev_status = self.progress['colab_status'].copy() if self.progress['colab_status'] else {}

                # 현재 상태 확인
                current_status = self.get_colab_status()
                self.progress['colab_status'] = current_status

                # 변화 감지
                if prev_status:
                    if current_status['total_models'] > prev_status.get('total_models', 0):
                        new_models = current_status['total_models'] - prev_status.get('total_models', 0)
                        success_msg = f"{new_models}개 모델 훈련 완료!"
                        self.progress['successes'].append(success_msg)
                        logging.info(f"✅ {success_msg}")

                    if current_status['status'] == 'completed' and prev_status.get('status') != 'completed':
                        completion_msg = "🎉 모든 모델 훈련 완료!"
                        self.progress['successes'].append(completion_msg)
                        logging.info(f"🎉 {completion_msg}")

                # 시작 시간 설정
                if not self.progress['start_time'] and current_status['total_models'] > 0:
                    self.progress['start_time'] = datetime.now()

                time.sleep(10)  # 10초마다 확인

            except Exception as e:
                logging.error(f"Colab 모니터링 오류: {str(e)}")
                time.sleep(30)

    def display_progress_loop(self):
        """진행 상황 표시 루프"""
        while self.is_running:
            try:
                self.display_current_status()
                time.sleep(2)  # 2초마다 업데이트
            except Exception as e:
                logging.error(f"진행 상황 표시 오류: {str(e)}")

    def start_monitoring(self):
        """모니터링 시작"""
        self.is_running = True

        # Colab 진행 상황 모니터링 스레드
        colab_thread = threading.Thread(target=self.monitor_colab_progress)
        colab_thread.daemon = True
        colab_thread.start()

        # 진행 상황 표시 스레드
        display_thread = threading.Thread(target=self.display_progress_loop)
        display_thread.daemon = True
        display_thread.start()

        # 메인 루프
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_running = False
        logging.info("모니터링 중지됨")

    def generate_progress_report(self):
        """진행 상황 리포트 생성"""
        try:
            colab_status = self.get_colab_status()

            report = {
                'timestamp': datetime.now().isoformat(),
                'total_models': self.colab_config['total_models'],
                'completed_models': colab_status['total_models'],
                'progress_percent': colab_status['progress_percent'],
                'status': colab_status['status'],
                'start_time': self.progress['start_time'].isoformat() if self.progress['start_time'] else None,
                'end_time': datetime.now().isoformat(),
                'successes': self.progress['successes'],
                'errors': self.progress['errors']
            }

            # 리포트 저장
            report_file = Path('colab_progress_report.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logging.info(f"📊 진행 상황 리포트 생성 완료: {report_file}")

            return report

        except Exception as e:
            logging.error(f"진행 상황 리포트 생성 오류: {str(e)}")
            return None

def main():
    """메인 실행 함수"""
    print(f"{Fore.CYAN}🚀 Google Colab ML 모델 훈련 실시간 모니터링")
    print(f"{Fore.CYAN}{'='*60}")

    try:
        monitor = RealColabMonitor()

        # Ctrl+C 핸들러
        def signal_handler(signum, frame):
            print(f"\n{Fore.YELLOW}⚠️ 모니터링 중지 중...")
            monitor.stop_monitoring()
            report = monitor.generate_progress_report()
            if report:
                print(f"{Fore.GREEN}📊 진행 상황 리포트 생성 완료!")
            exit(0)

        import signal
        signal.signal(signal.SIGINT, signal_handler)

        # 모니터링 시작
        monitor.start_monitoring()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"{Fore.RED}❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
