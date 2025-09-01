import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from supabase import create_client
import psutil
import requests
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Colorama 초기화
colorama.init()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_progress.log'),
        logging.StreamHandler()
    ]
)

class TrainingProgressMonitor:
    def __init__(self):
        self.load_env_file()
        self.supabase = self.get_supabase_client()
        self.is_running = False

        # 훈련 설정
        self.training_config = {
            'symbols': ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt'],
            'timeframes': ['1m', '5m', '15m', '1h'],
            'models': ['random_forest', 'xgboost', 'lightgbm', 'lstm'],
            'total_models': 80,  # 5 * 4 * 4
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
            'errors': [],
            'successes': [],
            'estimated_completion': None
        }

        # 성능 모니터링
        self.system_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'gpu_usage': 0,
            'network_usage': 0
        }

        # 진행률 바
        self.progress_bar = None

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

    def calculate_total_estimated_time(self):
        """총 예상 시간 계산"""
        total_minutes = 0

        for symbol in self.training_config['symbols']:
            for timeframe in self.training_config['timeframes']:
                for model_type in self.training_config['models']:
                    total_minutes += self.training_config['estimated_times'][model_type]

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

    def update_system_stats(self):
        """시스템 통계 업데이트"""
        try:
            # CPU 사용률
            self.system_stats['cpu_usage'] = psutil.cpu_percent(interval=1)

            # 메모리 사용률
            memory = psutil.virtual_memory()
            self.system_stats['memory_usage'] = memory.percent

            # 네트워크 사용률 (간단한 추정)
            network = psutil.net_io_counters()
            self.system_stats['network_usage'] = (network.bytes_sent + network.bytes_recv) / 1024 / 1024  # MB

        except Exception as e:
            logging.error(f"시스템 통계 업데이트 오류: {str(e)}")

    def display_progress_header(self):
        """진행 상황 헤더 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}🤖 ML 모델 훈련 진행 상황 모니터링")
        print(f"{Fore.CYAN}{'='*80}")

        total_time = self.calculate_total_estimated_time()
        print(f"{Fore.GREEN}📊 총 훈련 모델: {self.training_config['total_models']}개")
        print(f"{Fore.GREEN}⏰ 예상 총 소요 시간: {self.format_time(total_time)}")
        print(f"{Fore.GREEN}📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.CYAN}{'='*80}\n")

    def display_current_status(self):
        """현재 상태 표시"""
        if not self.progress['current_model']:
            return

        current_time = datetime.now()
        elapsed_time = current_time - self.progress['start_time']

        # 진행률 계산
        progress_percent = (self.progress['completed_models'] / self.training_config['total_models']) * 100

        # 예상 완료 시간 계산
        if self.progress['completed_models'] > 0:
            avg_time_per_model = elapsed_time.total_seconds() / self.progress['completed_models'] / 60  # 분
            remaining_models = self.training_config['total_models'] - self.progress['completed_models']
            remaining_minutes = remaining_models * avg_time_per_model
            estimated_completion = current_time + timedelta(minutes=remaining_minutes)
        else:
            estimated_completion = None

        # 화면 클리어 (터미널에서)
        os.system('clear' if os.name == 'posix' else 'cls')

        # 헤더 재표시
        self.display_progress_header()

        # 현재 상태 표시
        print(f"{Fore.YELLOW}🔄 현재 진행 중:")
        print(f"   📝 모델: {Fore.WHITE}{self.progress['current_model']}")
        print(f"   ⚙️ 단계: {Fore.WHITE}{self.progress['current_step']}")
        print(f"   ⏱️ 경과 시간: {Fore.WHITE}{str(elapsed_time).split('.')[0]}")

        if estimated_completion:
            print(f"   🎯 예상 완료: {Fore.WHITE}{estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n{Fore.GREEN}📈 전체 진행률:")

        # 진행률 바
        bar_length = 50
        filled_length = int(bar_length * progress_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"   {bar} {progress_percent:.1f}%")
        print(f"   ✅ 완료: {self.progress['completed_models']}/{self.training_config['total_models']} 모델")

        # 시스템 리소스 표시
        print(f"\n{Fore.BLUE}💻 시스템 리소스:")
        print(f"   🖥️ CPU: {Fore.WHITE}{self.system_stats['cpu_usage']:.1f}%")
        print(f"   💾 메모리: {Fore.WHITE}{self.system_stats['memory_usage']:.1f}%")
        print(f"   🌐 네트워크: {Fore.WHITE}{self.system_stats['network_usage']:.1f} MB")

        # 최근 성공/실패 표시
        if self.progress['successes']:
            print(f"\n{Fore.GREEN}✅ 최근 성공:")
            for success in self.progress['successes'][-3:]:
                print(f"   {success}")

        if self.progress['errors']:
            print(f"\n{Fore.RED}❌ 최근 오류:")
            for error in self.progress['errors'][-3:]:
                print(f"   {error}")

        print(f"\n{Fore.CYAN}{'='*80}")

    def start_model_training(self, symbol, timeframe, model_type):
        """모델 훈련 시작"""
        try:
            self.progress['current_model'] = f"{symbol.upper()} {timeframe} {model_type.upper()}"
            self.progress['current_step'] = "데이터 수집 중..."

            # 예상 시간 계산
            estimated_minutes = self.training_config['estimated_times'][model_type]

            logging.info(f"🚀 {self.progress['current_model']} 훈련 시작 (예상: {estimated_minutes}분)")

            # 단계별 진행 상황 업데이트
            steps = [
                "데이터 수집 중...",
                "특성 엔지니어링 중...",
                "모델 훈련 중...",
                "하이퍼파라미터 튜닝 중...",
                "성능 평가 중...",
                "모델 저장 중..."
            ]

            for i, step in enumerate(steps):
                self.progress['current_step'] = step
                time.sleep(estimated_minutes * 60 / len(steps))  # 시뮬레이션

            # 훈련 완료
            self.progress['completed_models'] += 1
            success_msg = f"{self.progress['current_model']} 훈련 완료!"
            self.progress['successes'].append(success_msg)

            logging.info(f"✅ {success_msg}")

            return True

        except Exception as e:
            error_msg = f"{self.progress['current_model']} 훈련 실패: {str(e)}"
            self.progress['errors'].append(error_msg)
            logging.error(f"❌ {error_msg}")
            return False

    def simulate_training_process(self):
        """훈련 과정 시뮬레이션"""
        try:
            self.progress['start_time'] = datetime.now()

            for symbol in self.training_config['symbols']:
                for timeframe in self.training_config['timeframes']:
                    for model_type in self.training_config['models']:
                        if not self.is_running:
                            break

                        # 모델 훈련 시작
                        success = self.start_model_training(symbol, timeframe, model_type)

                        if not success:
                            continue

                        # 잠시 대기 (실제로는 훈련 시간)
                        time.sleep(2)

            # 모든 훈련 완료
            total_time = datetime.now() - self.progress['start_time']
            logging.info(f"🎉 모든 모델 훈련 완료! 총 소요 시간: {str(total_time).split('.')[0]}")

        except Exception as e:
            logging.error(f"❌ 훈련 과정 시뮬레이션 오류: {str(e)}")

    def monitor_system_resources(self):
        """시스템 리소스 모니터링"""
        while self.is_running:
            try:
                self.update_system_stats()
                time.sleep(5)  # 5초마다 업데이트
            except Exception as e:
                logging.error(f"❌ 시스템 리소스 모니터링 오류: {str(e)}")

    def display_progress_loop(self):
        """진행 상황 표시 루프"""
        while self.is_running:
            try:
                self.display_current_status()
                time.sleep(1)  # 1초마다 업데이트
            except Exception as e:
                logging.error(f"❌ 진행 상황 표시 오류: {str(e)}")

    def start_monitoring(self):
        """모니터링 시작"""
        self.is_running = True

        # 시스템 리소스 모니터링 스레드
        resource_thread = threading.Thread(target=self.monitor_system_resources)
        resource_thread.daemon = True
        resource_thread.start()

        # 진행 상황 표시 스레드
        progress_thread = threading.Thread(target=self.display_progress_loop)
        progress_thread.daemon = True
        progress_thread.start()

        # 훈련 과정 시뮬레이션
        self.simulate_training_process()

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_running = False

    def generate_training_report(self):
        """훈련 리포트 생성"""
        try:
            if not self.progress['start_time']:
                return None

            total_time = datetime.now() - self.progress['start_time']

            report = {
                'timestamp': datetime.now().isoformat(),
                'total_models': self.training_config['total_models'],
                'completed_models': self.progress['completed_models'],
                'success_rate': (self.progress['completed_models'] / self.training_config['total_models']) * 100,
                'total_time': str(total_time),
                'start_time': self.progress['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'errors': self.progress['errors'],
                'successes': self.progress['successes'],
                'system_stats': self.system_stats
            }

            # 리포트 저장
            report_file = Path('training_report.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logging.info(f"📊 훈련 리포트 생성 완료: {report_file}")

            return report

        except Exception as e:
            logging.error(f"❌ 훈련 리포트 생성 오류: {str(e)}")
            return None

def main():
    """메인 실행 함수"""
    print(f"{Fore.CYAN}🚀 ML 모델 훈련 진행 상황 모니터링 시스템")
    print(f"{Fore.CYAN}{'='*60}")

    try:
        monitor = TrainingProgressMonitor()

        # Ctrl+C 핸들러
        def signal_handler(signum, frame):
            print(f"\n{Fore.YELLOW}⚠️ 모니터링 중지 중...")
            monitor.stop_monitoring()
            report = monitor.generate_training_report()
            if report:
                print(f"{Fore.GREEN}📊 훈련 리포트 생성 완료!")
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
