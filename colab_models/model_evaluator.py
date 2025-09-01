# 📊 모델 성능 평가 도구
# 훈련된 AI 트레이딩 모델들의 성능 분석 및 비교

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, models_dir='models', results_dir='results'):
        """모델 평가기 초기화"""
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.results = {}
        self.evaluation_summary = {}

    def load_models(self):
        """저장된 모델들 로드"""
        print('🔄 모델 로드 중...')

        if not os.path.exists(self.models_dir):
            print(f'❌ 모델 디렉토리가 없습니다: {self.models_dir}')
            return

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]

        for model_file in model_files:
            try:
                model_path = os.path.join(self.models_dir, model_file)
                model = joblib.load(model_path)

                # 모델 이름에서 정보 추출
                model_name = model_file.replace('.pkl', '')
                parts = model_name.split('_')

                if len(parts) >= 4:
                    symbol = parts[0]
                    timeframe = parts[1]
                    model_type = parts[2]
                    timestamp = '_'.join(parts[3:])

                    key = f"{symbol}_{timeframe}_{model_type}"
                    self.models[key] = {
                        'model': model,
                        'model_name': model_name,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'model_type': model_type,
                        'timestamp': timestamp,
                        'file_path': model_path
                    }

                    print(f'✅ {model_name} 로드 완료')

            except Exception as e:
                print(f'❌ {model_file} 로드 실패: {e}')

        print(f'📊 총 {len(self.models)}개 모델 로드 완료')

    def load_results(self):
        """결과 파일들 로드"""
        print('🔄 결과 파일 로드 중...')

        if not os.path.exists(self.results_dir):
            print(f'❌ 결과 디렉토리가 없습니다: {self.results_dir}')
            return

        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('_results.json')]

        for result_file in result_files:
            try:
                result_path = os.path.join(self.results_dir, result_file)

                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                model_name = result_data.get('model_name', result_file.replace('_results.json', ''))
                self.results[model_name] = result_data

                print(f'✅ {result_file} 로드 완료')

            except Exception as e:
                print(f'❌ {result_file} 로드 실패: {e}')

        print(f'📊 총 {len(self.results)}개 결과 파일 로드 완료')

    def create_performance_summary(self):
        """성능 요약 테이블 생성"""
        print('📊 성능 요약 생성 중...')

        summary_data = []

        for model_name, result in self.results.items():
            metrics = result.get('metrics', {})
            summary_data.append({
                'Model_Name': model_name,
                'Symbol': result.get('symbol', 'Unknown'),
                'Timeframe': result.get('timeframe', 'Unknown'),
                'Model_Type': result.get('model_type', 'Unknown'),
                'Timestamp': result.get('timestamp', 'Unknown'),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1_Score': metrics.get('f1', 0),
                'ROC_AUC': metrics.get('roc_auc', 0)
            })

        self.summary_df = pd.DataFrame(summary_data)

        if len(self.summary_df) > 0:
            print('✅ 성능 요약 생성 완료')
            print(f'📊 총 {len(self.summary_df)}개 모델 평가')

            # 최고 성능 모델들
            print('\n🏆 최고 성능 모델들:')
            print('F1 Score 기준:')
            print(self.summary_df.nlargest(5, 'F1_Score')[['Model_Name', 'Symbol', 'Timeframe', 'Model_Type', 'F1_Score', 'Accuracy']])

            print('\nAccuracy 기준:')
            print(self.summary_df.nlargest(5, 'Accuracy')[['Model_Name', 'Symbol', 'Timeframe', 'Model_Type', 'Accuracy', 'F1_Score']])

        return self.summary_df

    def plot_performance_comparison(self):
        """성능 비교 시각화"""
        if self.summary_df is None or len(self.summary_df) == 0:
            print('❌ 성능 요약 데이터가 없습니다')
            return

        print('📈 성능 비교 시각화 생성 중...')

        # 1. 전체 성능 산점도
        fig1 = px.scatter(
            self.summary_df,
            x='Accuracy',
            y='F1_Score',
            color='Model_Type',
            size='ROC_AUC',
            hover_data=['Symbol', 'Timeframe', 'Model_Name'],
            title='모델 성능 비교 (Accuracy vs F1 Score)',
            labels={'Accuracy': '정확도', 'F1_Score': 'F1 점수', 'ROC_AUC': 'ROC AUC'}
        )
        fig1.show()

        # 2. 심볼별 성능 비교
        fig2 = px.box(
            self.summary_df,
            x='Symbol',
            y='F1_Score',
            color='Model_Type',
            title='심볼별 F1 Score 분포'
        )
        fig2.show()

        # 3. 타임프레임별 성능 비교
        fig3 = px.box(
            self.summary_df,
            x='Timeframe',
            y='F1_Score',
            color='Model_Type',
            title='타임프레임별 F1 Score 분포'
        )
        fig3.show()

        # 4. 모델 타입별 성능 비교
        fig4 = px.box(
            self.summary_df,
            x='Model_Type',
            y=['Accuracy', 'F1_Score', 'ROC_AUC'],
            title='모델 타입별 성능 비교'
        )
        fig4.show()

        # 5. 히트맵
        pivot_df = self.summary_df.pivot_table(
            values='F1_Score',
            index='Symbol',
            columns='Timeframe',
            aggfunc='mean'
        )

        fig5 = px.imshow(
            pivot_df,
            title='심볼-타임프레임별 평균 F1 Score 히트맵',
            labels=dict(x='타임프레임', y='심볼', color='F1 Score')
        )
        fig5.show()

        print('✅ 성능 비교 시각화 완료')

    def analyze_model_stability(self, test_data=None):
        """모델 안정성 분석"""
        print('🔍 모델 안정성 분석 중...')

        stability_results = {}

        for key, model_info in self.models.items():
            model = model_info['model']
            model_name = model_info['model_name']

            try:
                # 교차 검증으로 안정성 측정
                if hasattr(model, 'feature_importances_'):
                    # 특성 중요도 분석
                    feature_importance = pd.DataFrame({
                        'feature': model_info.get('feature_columns', []),
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    stability_results[model_name] = {
                        'feature_importance': feature_importance,
                        'top_features': feature_importance.head(10).to_dict('records'),
                        'importance_std': feature_importance['importance'].std(),
                        'importance_mean': feature_importance['importance'].mean()
                    }

                    print(f'✅ {model_name} 안정성 분석 완료')

            except Exception as e:
                print(f'❌ {model_name} 안정성 분석 실패: {e}')

        self.stability_results = stability_results
        return stability_results

    def plot_feature_importance(self, top_n=10):
        """특성 중요도 시각화"""
        if not hasattr(self, 'stability_results'):
            print('❌ 안정성 분석 결과가 없습니다. 먼저 analyze_model_stability()를 실행하세요.')
            return

        print('📊 특성 중요도 시각화 생성 중...')

        for model_name, stability_data in self.stability_results.items():
            feature_importance = stability_data['feature_importance']

            # 상위 N개 특성만 선택
            top_features = feature_importance.head(top_n)

            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f'{model_name} - 상위 {top_n}개 중요 특성',
                labels={'importance': '중요도', 'feature': '특성'}
            )
            fig.show()

        print('✅ 특성 중요도 시각화 완료')

    def compare_model_parameters(self):
        """모델 파라미터 비교"""
        print('🔧 모델 파라미터 비교 중...')

        param_comparison = []

        for model_name, result in self.results.items():
            best_params = result.get('best_params', {})

            param_data = {
                'Model_Name': model_name,
                'Symbol': result.get('symbol', 'Unknown'),
                'Timeframe': result.get('timeframe', 'Unknown'),
                'Model_Type': result.get('model_type', 'Unknown')
            }

            # 파라미터 추가
            for param, value in best_params.items():
                param_data[param] = value

            param_comparison.append(param_data)

        self.param_df = pd.DataFrame(param_comparison)

        if len(self.param_df) > 0:
            print('✅ 파라미터 비교 완료')

            # 모델 타입별 파라미터 분포
            numeric_cols = self.param_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['Model_Name']]

            for col in numeric_cols:
                fig = px.box(
                    self.param_df,
                    x='Model_Type',
                    y=col,
                    title=f'{col} 파라미터 분포 (모델 타입별)'
                )
                fig.show()

        return self.param_df

    def generate_evaluation_report(self, output_path='evaluation_report.html'):
        """종합 평가 리포트 생성"""
        print('📋 종합 평가 리포트 생성 중...')

        # HTML 리포트 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI 트레이딩 모델 평가 리포트</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 AI 트레이딩 모델 평가 리포트</h1>
                <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        # 요약 통계
        if hasattr(self, 'summary_df') and len(self.summary_df) > 0:
            html_content += f"""
            <div class="section">
                <h2>📊 전체 요약</h2>
                <div class="metric">
                    <strong>총 모델 수:</strong> {len(self.summary_df)}
                </div>
                <div class="metric">
                    <strong>평균 F1 Score:</strong> {self.summary_df['F1_Score'].mean():.4f}
                </div>
                <div class="metric">
                    <strong>평균 Accuracy:</strong> {self.summary_df['Accuracy'].mean():.4f}
                </div>
                <div class="metric">
                    <strong>최고 F1 Score:</strong> {self.summary_df['F1_Score'].max():.4f}
                </div>
            </div>

            # 최고 성능 모델들
            top_models = self.summary_df.nlargest(5, 'F1_Score')
            html_content += """
            <div class="section">
                <h2>🏆 최고 성능 모델들 (F1 Score 기준)</h2>
                <table>
                    <tr>
                        <th>모델명</th>
                        <th>심볼</th>
                        <th>타임프레임</th>
                        <th>모델 타입</th>
                        <th>F1 Score</th>
                        <th>Accuracy</th>
                    </tr>
            """

            for _, row in top_models.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['Model_Name']}</td>
                        <td>{row['Symbol']}</td>
                        <td>{row['Timeframe']}</td>
                        <td>{row['Model_Type']}</td>
                        <td class="highlight">{row['F1_Score']:.4f}</td>
                        <td>{row['Accuracy']:.4f}</td>
                    </tr>
                """

            html_content += """
                </table>
            </div>
            """

        # 모델 타입별 성능
        if hasattr(self, 'summary_df') and len(self.summary_df) > 0:
            model_type_stats = self.summary_df.groupby('Model_Type').agg({
                'F1_Score': ['mean', 'std', 'count'],
                'Accuracy': ['mean', 'std']
            }).round(4)

            html_content += """
            <div class="section">
                <h2>📈 모델 타입별 성능 통계</h2>
                <table>
                    <tr>
                        <th>모델 타입</th>
                        <th>모델 수</th>
                        <th>평균 F1 Score</th>
                        <th>F1 Score 표준편차</th>
                        <th>평균 Accuracy</th>
                        <th>Accuracy 표준편차</th>
                    </tr>
            """

            for model_type in model_type_stats.index:
                stats = model_type_stats.loc[model_type]
                html_content += f"""
                    <tr>
                        <td>{model_type}</td>
                        <td>{stats[('F1_Score', 'count')]}</td>
                        <td>{stats[('F1_Score', 'mean')]:.4f}</td>
                        <td>{stats[('F1_Score', 'std')]:.4f}</td>
                        <td>{stats[('Accuracy', 'mean')]:.4f}</td>
                        <td>{stats[('Accuracy', 'std')]:.4f}</td>
                    </tr>
                """

            html_content += """
                </table>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        # HTML 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f'✅ 평가 리포트 생성 완료: {output_path}')
        return output_path

    def run_full_evaluation(self):
        """전체 평가 프로세스 실행"""
        print('🚀 전체 모델 평가 시작')
        print('=' * 60)

        # 1. 모델 및 결과 로드
        self.load_models()
        self.load_results()

        # 2. 성능 요약 생성
        self.create_performance_summary()

        # 3. 성능 비교 시각화
        self.plot_performance_comparison()

        # 4. 모델 안정성 분석
        self.analyze_model_stability()

        # 5. 특성 중요도 시각화
        self.plot_feature_importance()

        # 6. 파라미터 비교
        self.compare_model_parameters()

        # 7. 평가 리포트 생성
        self.generate_evaluation_report()

        print('=' * 60)
        print('✅ 전체 모델 평가 완료!')

        return {
            'summary_df': self.summary_df,
            'stability_results': self.stability_results,
            'param_df': self.param_df
        }

# 사용 예시
if __name__ == "__main__":
    # 모델 평가기 생성
    evaluator = ModelEvaluator()

    # 전체 평가 실행
    results = evaluator.run_full_evaluation()

    # 결과 확인
    print('\n📊 평가 결과 요약:')
    if results['summary_df'] is not None:
        print(f"총 모델 수: {len(results['summary_df'])}")
        print(f"평균 F1 Score: {results['summary_df']['F1_Score'].mean():.4f}")
        print(f"최고 F1 Score: {results['summary_df']['F1_Score'].max():.4f}")

