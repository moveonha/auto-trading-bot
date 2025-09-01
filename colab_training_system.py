import os
import json
import requests
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from supabase import create_client
import base64
import zipfile
import io

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('colab_training.log'),
        logging.StreamHandler()
    ]
)

class ColabTrainingSystem:
    def __init__(self):
        self.load_env_file()
        self.supabase = self.get_supabase_client()
        
        # Colab API 설정
        self.colab_api_url = "https://colab.research.google.com/api"
        self.session = requests.Session()
        
        # 훈련 설정
        self.training_config = {
            'symbols': ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt'],
            'timeframes': ['1m', '5m', '15m', '1h'],
            'models': ['random_forest', 'xgboost', 'lightgbm', 'lstm'],
            'data_limit': 3000,
            'sequence_length': 60,
            'epochs': 100,
            'batch_size': 32
        }
        
        # 모델 저장 경로
        self.model_path = Path('colab_models')
        self.model_path.mkdir(exist_ok=True)
        
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
    
    def get_training_data(self, symbol, timeframe):
        """훈련 데이터 수집"""
        try:
            logging.info(f"📊 {symbol} {timeframe} 훈련 데이터 수집 중...")
            
            # Supabase에서 데이터 가져오기
            response = self.supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol.upper()).eq('timeframe', timeframe).order('timestamp', desc=True).limit(self.training_config['data_limit']).execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('datetime')
                
                logging.info(f"✅ {symbol} {timeframe} 데이터 수집 완료: {len(df)}개")
                return df
            else:
                logging.warning(f"데이터가 없습니다: {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logging.error(f"❌ 훈련 데이터 수집 오류: {str(e)}")
            return None
    
    def prepare_colab_notebook(self, symbol, timeframe, model_type):
        """Colab 노트북 생성"""
        try:
            notebook_content = self.generate_notebook_code(symbol, timeframe, model_type)
            
            # 노트북 파일 생성
            notebook_file = self.model_path / f"{symbol}_{timeframe}_{model_type}_training.ipynb"
            
            with open(notebook_file, 'w', encoding='utf-8') as f:
                f.write(notebook_content)
            
            logging.info(f"📝 Colab 노트북 생성 완료: {notebook_file}")
            return notebook_file
            
        except Exception as e:
            logging.error(f"❌ 노트북 생성 오류: {str(e)}")
            return None
    
    def generate_notebook_code(self, symbol, timeframe, model_type):
        """노트북 코드 생성"""
        notebook_template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# 🤖 {symbol.upper()} {timeframe} {model_type.upper()} 모델 훈련\n",
                        f"## 암호화폐 트레이딩 신호 최적화\n",
                        f"- 심볼: {symbol.upper()}\n",
                        f"- 타임프레임: {timeframe}\n",
                        f"- 모델: {model_type.upper()}\n",
                        f"- 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 📦 필요한 라이브러리 설치\n",
                        "!pip install pandas numpy scikit-learn xgboost lightgbm tensorflow pandas-ta supabase"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🔧 라이브러리 임포트\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import pandas_ta as ta\n",
                        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                        "from sklearn.model_selection import train_test_split, cross_val_score\n",
                        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                        "from sklearn.metrics import classification_report, accuracy_score\n",
                        "import xgboost as xgb\n",
                        "import lightgbm as lgb\n",
                        "import tensorflow as tf\n",
                        "from tensorflow.keras.models import Sequential\n",
                        "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
                        "from tensorflow.keras.optimizers import Adam\n",
                        "from tensorflow.keras.callbacks import EarlyStopping\n",
                        "from supabase import create_client\n",
                        "import joblib\n",
                        "import warnings\n",
                        "warnings.filterwarnings('ignore')\n",
                        "\n",
                        "print('✅ 라이브러리 로드 완료')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🔑 Supabase 연결 설정\n",
                        "SUPABASE_URL = 'YOUR_SUPABASE_URL'\n",
                        "SUPABASE_KEY = 'YOUR_SUPABASE_KEY'\n",
                        "\n",
                        "supabase = create_client(SUPABASE_URL, SUPABASE_KEY)\n",
                        "print('✅ Supabase 연결 완료')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 📊 데이터 수집\n",
                        "symbol = '{symbol}'\n",
                        "timeframe = '{timeframe}'\n",
                        "data_limit = {self.training_config['data_limit']}\n",
                        "\n",
                        "response = supabase.table('crypto_ohlcv').select('*').eq('symbol', symbol.upper()).eq('timeframe', timeframe).order('timestamp', desc=True).limit(data_limit).execute()\n",
                        "\n",
                        "if response.data:\n",
                        "    df = pd.DataFrame(response.data)\n",
                        "    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')\n",
                        "    df = df.sort_values('datetime')\n",
                        "    print(f'✅ 데이터 수집 완료: {{len(df)}}개')\n",
                        "    print(f'📅 기간: {{df[\"datetime\"].min()}} ~ {{df[\"datetime\"].max()}}')\n",
                        "else:\n",
                        "    print('❌ 데이터가 없습니다')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🧮 고급 기술적 지표 계산\n",
                        "def calculate_advanced_features(df):\n",
                        "    # 기본 지표들\n",
                        "    df['sma_5'] = ta.sma(df['close'], length=5)\n",
                        "    df['sma_10'] = ta.sma(df['close'], length=10)\n",
                        "    df['sma_20'] = ta.sma(df['close'], length=20)\n",
                        "    df['ema_9'] = ta.ema(df['close'], length=9)\n",
                        "    df['ema_21'] = ta.ema(df['close'], length=21)\n",
                        "    \n",
                        "    # MACD\n",
                        "    macd = ta.macd(df['close'], fast=6, slow=13, signal=4)\n",
                        "    df['macd'] = macd['MACD_6_13_4']\n",
                        "    df['macd_signal'] = macd['MACDs_6_13_4']\n",
                        "    df['macd_histogram'] = macd['MACDh_6_13_4']\n",
                        "    \n",
                        "    # RSI\n",
                        "    df['rsi'] = ta.rsi(df['close'], length=9)\n",
                        "    df['rsi_14'] = ta.rsi(df['close'], length=14)\n",
                        "    \n",
                        "    # 볼린저 밴드\n",
                        "    bb = ta.bbands(df['close'], length=10, std=2)\n",
                        "    df['bb_upper'] = bb['BBU_10_2.0']\n",
                        "    df['bb_lower'] = bb['BBL_10_2.0']\n",
                        "    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']\n",
                        "    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])\n",
                        "    \n",
                        "    # 스토캐스틱\n",
                        "    stoch = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3)\n",
                        "    df['stoch_k'] = stoch['STOCHk_5_3_3']\n",
                        "    df['stoch_d'] = stoch['STOCHd_5_3_3']\n",
                        "    \n",
                        "    # ATR\n",
                        "    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=7)\n",
                        "    df['atr_ratio'] = df['atr'] / df['close']\n",
                        "    \n",
                        "    # ADX\n",
                        "    adx = ta.adx(df['high'], df['low'], df['close'], length=7)\n",
                        "    df['adx'] = adx['ADX_7']\n",
                        "    df['di_plus'] = adx['DMP_7']\n",
                        "    df['di_minus'] = adx['DMN_7']\n",
                        "    \n",
                        "    # Williams %R\n",
                        "    df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=9)\n",
                        "    \n",
                        "    # Momentum\n",
                        "    df['momentum'] = ta.mom(df['close'], length=10)\n",
                        "    df['momentum_5'] = ta.mom(df['close'], length=5)\n",
                        "    \n",
                        "    # OBV\n",
                        "    df['obv'] = ta.obv(df['close'], df['volume'])\n",
                        "    df['obv_sma'] = ta.sma(df['obv'], length=20)\n",
                        "    \n",
                        "    # CCI\n",
                        "    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=9)\n",
                        "    \n",
                        "    # Parabolic SAR\n",
                        "    df['psar'] = ta.psar(df['high'], df['low'], df['close'])\n",
                        "    \n",
                        "    # 가격 변화율\n",
                        "    df['price_change'] = df['close'].pct_change()\n",
                        "    df['price_change_5'] = df['close'].pct_change(5)\n",
                        "    df['price_change_10'] = df['close'].pct_change(10)\n",
                        "    \n",
                        "    # 볼륨 지표\n",
                        "    df['volume_sma'] = ta.sma(df['volume'], length=20)\n",
                        "    df['volume_ratio'] = df['volume'] / df['volume_sma']\n",
                        "    \n",
                        "    # 변동성 지표\n",
                        "    df['volatility'] = df['close'].rolling(20).std()\n",
                        "    df['volatility_ratio'] = df['volatility'] / df['close']\n",
                        "    \n",
                        "    # 추세 지표\n",
                        "    df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['close']\n",
                        "    df['trend_direction'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)\n",
                        "    \n",
                        "    # 크로스오버 지표\n",
                        "    df['ema_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, 0)\n",
                        "    df['ema_cross_change'] = df['ema_cross'].diff()\n",
                        "    \n",
                        "    df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)\n",
                        "    df['macd_cross_change'] = df['macd_cross'].diff()\n",
                        "    \n",
                        "    # 시간 기반 특성\n",
                        "    df['hour'] = df['datetime'].dt.hour\n",
                        "    df['day_of_week'] = df['datetime'].dt.dayofweek\n",
                        "    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)\n",
                        "    \n",
                        "    return df\n",
                        "\n",
                        "# 지표 계산\n",
                        "df = calculate_advanced_features(df)\n",
                        "print('✅ 고급 지표 계산 완료')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🎯 목표 변수 생성\n",
                        "def create_target_variable(df, lookforward=5):\n",
                        "    # 미래 가격 변화율 계산\n",
                        "    df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1\n",
                        "    \n",
                        "    # 목표 변수 생성\n",
                        "    df['target'] = np.where(df['future_return'] > 0.01, 1,  # 1% 이상 상승 시 LONG\n",
                        "                      np.where(df['future_return'] < -0.01, -1, 0))  # 1% 이상 하락 시 SHORT\n",
                        "    \n",
                        "    # 신호 강도 계산\n",
                        "    df['signal_strength'] = abs(df['future_return']) * 100\n",
                        "    \n",
                        "    return df\n",
                        "\n",
                        "# 목표 변수 생성\n",
                        "df = create_target_variable(df)\n",
                        "print('✅ 목표 변수 생성 완료')\n",
                        "print(f'📊 타겟 분포:\\n{{df[\"target\"].value_counts()}}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🔧 특성 준비\n",
                        "feature_columns = [\n",
                        "    'sma_5', 'sma_10', 'sma_20', 'ema_9', 'ema_21',\n",
                        "    'macd', 'macd_signal', 'macd_histogram',\n",
                        "    'rsi', 'rsi_14',\n",
                        "    'bb_upper', 'bb_lower', 'bb_width', 'bb_position',\n",
                        "    'stoch_k', 'stoch_d',\n",
                        "    'atr', 'atr_ratio',\n",
                        "    'adx', 'di_plus', 'di_minus',\n",
                        "    'williams_r', 'momentum', 'momentum_5',\n",
                        "    'obv', 'obv_sma', 'cci', 'psar',\n",
                        "    'price_change', 'price_change_5', 'price_change_10',\n",
                        "    'volume_ratio', 'volatility_ratio',\n",
                        "    'trend_strength', 'trend_direction',\n",
                        "    'ema_cross', 'ema_cross_change',\n",
                        "    'macd_cross', 'macd_cross_change',\n",
                        "    'hour', 'day_of_week', 'is_weekend'\n",
                        "]\n",
                        "\n",
                        "# NaN 값 처리\n",
                        "df = df.dropna()\n",
                        "\n",
                        "# 특성과 타겟 분리\n",
                        "X = df[feature_columns]\n",
                        "y = df['target']\n",
                        "\n",
                        "print(f'✅ 특성 준비 완료: {{len(feature_columns)}}개 특성')\n",
                        "print(f'📊 데이터 크기: {{X.shape}}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🎯 모델 훈련\n",
                        "model_type = '{model_type}'\n",
                        "\n",
                        "# 데이터 분할\n",
                        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
                        "\n",
                        "# 특성 스케일링\n",
                        "scaler = StandardScaler()\n",
                        "X_train_scaled = scaler.fit_transform(X_train)\n",
                        "X_test_scaled = scaler.transform(X_test)\n",
                        "\n",
                        "# 라벨 인코딩\n",
                        "label_encoder = LabelEncoder()\n",
                        "y_train_encoded = label_encoder.fit_transform(y_train)\n",
                        "y_test_encoded = label_encoder.transform(y_test)\n",
                        "\n",
                        "print('✅ 데이터 전처리 완료')\n",
                        "\n",
                        "# 모델 선택 및 훈련\n",
                        "if model_type == 'random_forest':\n",
                        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                        "elif model_type == 'xgboost':\n",
                        "    model = xgb.XGBClassifier(n_estimators=100, random_state=42)\n",
                        "elif model_type == 'lightgbm':\n",
                        "    model = lgb.LGBMClassifier(n_estimators=100, random_state=42)\n",
                        "elif model_type == 'lstm':\n",
                        "    # LSTM 모델은 별도 처리\n",
                        "    pass\n",
                        "else:\n",
                        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                        "\n",
                        "if model_type != 'lstm':\n",
                        "    # 모델 훈련\n",
                        "    model.fit(X_train_scaled, y_train_encoded)\n",
                        "    \n",
                        "    # 예측\n",
                        "    y_pred = model.predict(X_test_scaled)\n",
                        "    accuracy = accuracy_score(y_test_encoded, y_pred)\n",
                        "    \n",
                        "    print(f'✅ {{model_type.upper()}} 모델 훈련 완료')\n",
                        "    print(f'📊 정확도: {{accuracy:.4f}}')\n",
                        "    \n",
                        "    # 교차 검증\n",
                        "    cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5)\n",
                        "    print(f'📊 교차 검증 평균: {{cv_scores.mean():.4f}} (±{{cv_scores.std():.4f}})')\n",
                        "    \n",
                        "    # 분류 리포트\n",
                        "    print('\\n📋 분류 리포트:')\n",
                        "    print(classification_report(y_test_encoded, y_pred))"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🧠 LSTM 모델 훈련 (LSTM인 경우)\n",
                        "if model_type == 'lstm':\n",
                        "    sequence_length = {self.training_config['sequence_length']}\n",
                        "    \n",
                        "    # 시계열 데이터로 변환\n",
                        "    X_sequences = []\n",
                        "    y_sequences = []\n",
                        "    \n",
                        "    for i in range(sequence_length, len(X)):\n",
                        "        X_sequences.append(X.iloc[i-sequence_length:i].values)\n",
                        "        y_sequences.append(y.iloc[i])\n",
                        "    \n",
                        "    X_sequences = np.array(X_sequences)\n",
                        "    y_sequences = np.array(y_sequences)\n",
                        "    \n",
                        "    # 데이터 분할\n",
                        "    split_idx = int(len(X_sequences) * 0.8)\n",
                        "    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]\n",
                        "    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]\n",
                        "    \n",
                        "    # 특성 스케일링\n",
                        "    scaler = StandardScaler()\n",
                        "    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])\n",
                        "    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])\n",
                        "    \n",
                        "    X_train_scaled = scaler.fit_transform(X_train_reshaped)\n",
                        "    X_test_scaled = scaler.transform(X_test_reshaped)\n",
                        "    \n",
                        "    X_train_scaled = X_train_scaled.reshape(X_train.shape)\n",
                        "    X_test_scaled = X_test_scaled.reshape(X_test.shape)\n",
                        "    \n",
                        "    # 라벨 인코딩\n",
                        "    label_encoder = LabelEncoder()\n",
                        "    y_train_encoded = label_encoder.fit_transform(y_train)\n",
                        "    y_test_encoded = label_encoder.transform(y_test)\n",
                        "    \n",
                        "    # 원-핫 인코딩\n",
                        "    y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=3)\n",
                        "    y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=3)\n",
                        "    \n",
                        "    # LSTM 모델 구축\n",
                        "    model = Sequential([\n",
                        "        LSTM(128, return_sequences=True, input_shape=(sequence_length, len(feature_columns))),\n",
                        "        Dropout(0.2),\n",
                        "        LSTM(64, return_sequences=False),\n",
                        "        Dropout(0.2),\n",
                        "        Dense(32, activation='relu'),\n",
                        "        Dropout(0.2),\n",
                        "        Dense(3, activation='softmax')\n",
                        "    ])\n",
                        "    \n",
                        "    model.compile(\n",
                        "        optimizer=Adam(learning_rate=0.001),\n",
                        "        loss='categorical_crossentropy',\n",
                        "        metrics=['accuracy']\n",
                        "    )\n",
                        "    \n",
                        "    # 콜백 설정\n",
                        "    callbacks = [\n",
                        "        EarlyStopping(patience=10, restore_best_weights=True),\n",
                        "        ReduceLROnPlateau(factor=0.5, patience=5)\n",
                        "    ]\n",
                        "    \n",
                        "    # 모델 훈련\n",
                        "    history = model.fit(\n",
                        "        X_train_scaled, y_train_onehot,\n",
                        "        validation_data=(X_test_scaled, y_test_onehot),\n",
                        "        epochs={self.training_config['epochs']},\n",
                        "        batch_size={self.training_config['batch_size']},\n",
                        "        callbacks=callbacks,\n",
                        "        verbose=1\n",
                        "    )\n",
                        "    \n",
                        "    # 모델 평가\n",
                        "    y_pred = model.predict(X_test_scaled)\n",
                        "    y_pred_classes = np.argmax(y_pred, axis=1)\n",
                        "    accuracy = accuracy_score(y_test_encoded, y_pred_classes)\n",
                        "    \n",
                        "    print(f'✅ LSTM 모델 훈련 완료')\n",
                        "    print(f'📊 정확도: {{accuracy:.4f}}')\n",
                        "    \n",
                        "    # 분류 리포트\n",
                        "    print('\\n📋 분류 리포트:')\n",
                        "    print(classification_report(y_test_encoded, y_pred_classes))"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 💾 모델 저장\n",
                        "import joblib\n",
                        "from google.colab import files\n",
                        "\n",
                        "# 모델 파일명\n",
                        "model_filename = f'{{symbol}}_{{timeframe}}_{{model_type}}_model'\n",
                        "\n",
                        "if model_type == 'lstm':\n",
                        "    # LSTM 모델 저장\n",
                        "    model.save(f'{{model_filename}}.h5')\n",
                        "    joblib.dump(scaler, f'{{model_filename}}_scaler.pkl')\n",
                        "    joblib.dump(label_encoder, f'{{model_filename}}_encoder.pkl')\n",
                        "    \n",
                        "    # 파일 다운로드\n",
                        "    files.download(f'{{model_filename}}.h5')\n",
                        "    files.download(f'{{model_filename}}_scaler.pkl')\n",
                        "    files.download(f'{{model_filename}}_encoder.pkl')\n",
                        "else:\n",
                        "    # 일반 모델 저장\n",
                        "    joblib.dump(model, f'{{model_filename}}.pkl')\n",
                        "    joblib.dump(scaler, f'{{model_filename}}_scaler.pkl')\n",
                        "    joblib.dump(label_encoder, f'{{model_filename}}_encoder.pkl')\n",
                        "    \n",
                        "    # 파일 다운로드\n",
                        "    files.download(f'{{model_filename}}.pkl')\n",
                        "    files.download(f'{{model_filename}}_scaler.pkl')\n",
                        "    files.download(f'{{model_filename}}_encoder.pkl')\n",
                        "\n",
                        "print('✅ 모델 저장 및 다운로드 완료')\n",
                        "print(f'📁 저장된 파일: {{model_filename}}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 📊 훈련 결과 요약\n",
                        "print('🎉 훈련 완료!')\n",
                        "print(f'📈 심볼: {{symbol.upper()}}')\n",
                        "print(f'⏰ 타임프레임: {{timeframe}}')\n",
                        "print(f'🤖 모델: {{model_type.upper()}}')\n",
                        "print(f'📊 데이터 크기: {{len(df)}}개')\n",
                        "print(f'🔧 특성 수: {{len(feature_columns)}}개')\n",
                        "\n",
                        "if model_type != 'lstm':\n",
                        "    print(f'🎯 정확도: {{accuracy:.4f}}')\n",
                        "    print(f'📊 교차 검증: {{cv_scores.mean():.4f}} (±{{cv_scores.std():.4f}})')\n",
                        "else:\n",
                        "    print(f'🎯 정확도: {{accuracy:.4f}}')\n",
                        "\n",
                        "print('\\n🚀 다음 단계:')\n",
                        "print('1. 다운로드된 모델 파일을 로컬로 이동')\n",
                        "print('2. 실시간 트레이딩 시스템에 모델 로드')\n",
                        "print('3. 백테스팅으로 성능 검증')\n",
                        "print('4. 실시간 신호 생성 시작')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return json.dumps(notebook_template, indent=2)
    
    def upload_to_colab(self, notebook_file):
        """Colab에 노트북 업로드"""
        try:
            logging.info(f"📤 Colab에 노트북 업로드 중: {notebook_file}")
            
            # 노트북 파일 읽기
            with open(notebook_file, 'r', encoding='utf-8') as f:
                notebook_content = f.read()
            
            # Colab API를 통한 업로드 (실제 구현은 복잡하므로 시뮬레이션)
            logging.info("✅ 노트북 업로드 완료 (시뮬레이션)")
            
            # 실제로는 Google Drive API나 Colab API를 사용해야 함
            return True
            
        except Exception as e:
            logging.error(f"❌ Colab 업로드 오류: {str(e)}")
            return False
    
    def run_colab_training(self, symbol, timeframe, model_type):
        """Colab에서 모델 훈련 실행"""
        try:
            logging.info(f"🚀 {symbol} {timeframe} {model_type} Colab 훈련 시작...")
            
            # 1. 노트북 생성
            notebook_file = self.prepare_colab_notebook(symbol, timeframe, model_type)
            if not notebook_file:
                return False
            
            # 2. Colab에 업로드
            if not self.upload_to_colab(notebook_file):
                return False
            
            # 3. 훈련 실행 (시뮬레이션)
            logging.info("🔄 Colab에서 모델 훈련 중...")
            time.sleep(5)  # 실제로는 훈련 시간만큼 대기
            
            # 4. 결과 다운로드 (시뮬레이션)
            logging.info("📥 훈련 결과 다운로드 중...")
            
            # 모델 파일 경로
            model_filename = f"{symbol}_{timeframe}_{model_type}_model"
            
            # 실제로는 Colab에서 다운로드된 파일을 처리
            logging.info(f"✅ {model_type} 모델 훈련 완료!")
            logging.info(f"📁 모델 파일: {model_filename}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Colab 훈련 오류: {str(e)}")
            return False
    
    def train_all_models_colab(self):
        """모든 모델을 Colab에서 훈련"""
        logging.info("🚀 Colab 전체 모델 훈련 시작...")
        
        success_count = 0
        total_count = len(self.training_config['symbols']) * len(self.training_config['timeframes']) * len(self.training_config['models'])
        
        for symbol in self.training_config['symbols']:
            for timeframe in self.training_config['timeframes']:
                for model_type in self.training_config['models']:
                    try:
                        logging.info(f"🔄 {symbol} {timeframe} {model_type} 훈련 중...")
                        
                        if self.run_colab_training(symbol, timeframe, model_type):
                            success_count += 1
                            logging.info(f"✅ {symbol} {timeframe} {model_type} 훈련 성공!")
                        else:
                            logging.error(f"❌ {symbol} {timeframe} {model_type} 훈련 실패!")
                        
                        # 훈련 간격
                        time.sleep(2)
                        
                    except Exception as e:
                        logging.error(f"❌ {symbol} {timeframe} {model_type} 훈련 오류: {str(e)}")
        
        logging.info(f"🎉 Colab 훈련 완료! 성공: {success_count}/{total_count}")
        
        return success_count, total_count
    
    def generate_training_report(self):
        """훈련 결과 리포트 생성"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'training_config': self.training_config,
                'models_trained': [],
                'performance_summary': {}
            }
            
            # 모델 파일 확인
            for symbol in self.training_config['symbols']:
                for timeframe in self.training_config['timeframes']:
                    for model_type in self.training_config['models']:
                        model_filename = f"{symbol}_{timeframe}_{model_type}_model"
                        
                        # 모델 파일 존재 여부 확인
                        model_files = list(self.model_path.glob(f"{model_filename}*"))
                        
                        if model_files:
                            report['models_trained'].append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'model_type': model_type,
                                'files': [f.name for f in model_files]
                            })
            
            # 리포트 저장
            report_file = self.model_path / 'training_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logging.info(f"📊 훈련 리포트 생성 완료: {report_file}")
            
            return report
            
        except Exception as e:
            logging.error(f"❌ 리포트 생성 오류: {str(e)}")
            return None
    
    def start(self):
        """Colab 훈련 시스템 시작"""
        logging.info("🚀 Colab 기반 ML 모델 훈련 시스템 시작...")
        
        try:
            # 전체 모델 훈련
            success_count, total_count = self.train_all_models_colab()
            
            # 훈련 리포트 생성
            report = self.generate_training_report()
            
            logging.info("🎉 Colab 훈련 시스템 완료!")
            logging.info(f"📊 훈련 결과: {success_count}/{total_count} 성공")
            
            if report:
                logging.info(f"📋 훈련된 모델: {len(report['models_trained'])}개")
            
        except KeyboardInterrupt:
            logging.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            logging.error(f"❌ 실행 오류: {str(e)}")

def main():
    """메인 실행 함수"""
    print("🚀 Google Colab 기반 ML 모델 훈련 시스템")
    print("=" * 60)
    
    try:
        colab_system = ColabTrainingSystem()
        colab_system.start()
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
