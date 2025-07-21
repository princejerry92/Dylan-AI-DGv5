import pandas as pd
import numpy as np
import xgboost as xgb
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import time
import joblib
import warnings
import gc

# Constants
SYMBOL = "XAUUSD"
LOOKBACK_BARS = 200
PIP_VALUE = 5.00  # 50 pips TP
SL_PIPS = 3.50    # 35 pips SL
CHECK_INTERVAL = 900  # 15 minutes in seconds

# Feature columns for base models (from training)
base_feature_cols_15m = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'atr', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'hour', 'session',
    'candle_body', 'upper_shadow', 'lower_shadow', 'intraday_momentum',
    'volume_momentum', 'lagged_return_1', 'lagged_return_2', 'lagged_return_3'
]

base_feature_cols_1h = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'atr', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'hour', 'session',
    'candle_body', 'upper_shadow', 'lower_shadow', 'intraday_momentum'
]

base_feature_cols_4h = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'atr', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'hour', 'session',
    'candle_body', 'upper_shadow', 'lower_shadow', 'intraday_momentum'
]

# Feature columns for meta-learner (aligned with training)
# Removed volatility_regime_15m, volatility_regime_1h, volatility_regime_4h
meta_feature_cols = [
    'hl2_15m', 'hlc3_15m', 'ohlc4_15m', 'true_range_15m', 'atr_14_15m', 'atr_21_15m', 'atr_50_15m',
    'rsi_14_15m', 'rsi_21_15m', 'macd_15m', 'macd_signal_15m', 'macd_histogram_15m', 'bb_upper_15m',
    'bb_lower_15m', 'bb_position_15m', 'sma_10_15m', 'sma_10_slope_15m', 'sma_20_15m', 'sma_20_slope_15m',
    'sma_50_15m', 'sma_50_slope_15m', 'candle_body_15m', 'upper_shadow_15m', 'lower_shadow_15m',
    'body_to_range_15m', 'volume_sma_15m', 'volume_ratio_15m', 'price_volume_15m', 'momentum_5_15m',
    'high_momentum_5_15m', 'low_momentum_5_15m', 'momentum_10_15m', 'high_momentum_10_15m',
    'low_momentum_10_15m', 'momentum_20_15m', 'high_momentum_20_15m', 'low_momentum_20_15m',
    'pred_15m', 'pred_direction_15m', 'volatility_regime', 'vol_regime_low', 'vol_regime_medium',
    'vol_regime_high', 'hour', 'day_of_week', 'asian_session', 'london_session', 'ny_session',
    'rsi_divergence', 'rsi_overbought', 'rsi_oversold', 'recent_high', 'recent_low', 'dist_to_high',
    'dist_to_low', 'is_doji', 'is_hammer', 'is_shooting_star',
    'hl2_1h', 'hlc3_1h', 'ohlc4_1h', 'true_range_1h', 'atr_14_1h', 'atr_21_1h', 'atr_50_1h',
    'rsi_14_1h', 'rsi_21_1h', 'macd_1h', 'macd_signal_1h', 'macd_histogram_1h', 'bb_upper_1h',
    'bb_lower_1h', 'bb_position_1h', 'sma_10_1h', 'sma_10_slope_1h', 'sma_20_1h', 'sma_20_slope_1h',
    'sma_50_1h', 'sma_50_slope_1h', 'candle_body_1h', 'upper_shadow_1h', 'lower_shadow_1h',
    'body_to_range_1h', 'volume_sma_1h', 'volume_ratio_1h', 'price_volume_1h', 'momentum_5_1h',
    'high_momentum_5_1h', 'low_momentum_5_1h', 'momentum_10_1h', 'high_momentum_10_1h',
    'low_momentum_10_1h', 'momentum_20_1h', 'high_momentum_20_1h', 'low_momentum_20_1h',
    'pred_1h', 'pred_direction_1h', 'volatility_regime_1h', 'vol_regime_low_1h', 'vol_regime_medium_1h',
    'vol_regime_high_1h', 'hour_1h', 'day_of_week_1h', 'asian_session_1h', 'london_session_1h',
    'ny_session_1h', 'rsi_divergence_1h', 'rsi_overbought_1h', 'rsi_oversold_1h', 'recent_high_1h',
    'recent_low_1h', 'dist_to_high_1h', 'dist_to_low_1h', 'is_doji_1h', 'is_hammer_1h',
    'is_shooting_star_1h',
    'hl2_4h', 'hlc3_4h', 'ohlc4_4h', 'true_range_4h', 'atr_14_4h', 'atr_21_4h', 'atr_50_4h',
    'rsi_14_4h', 'rsi_21_4h', 'macd_4h', 'macd_signal_4h', 'macd_histogram_4h', 'bb_upper_4h',
    'bb_lower_4h', 'bb_position_4h', 'sma_10_4h', 'sma_10_slope_4h', 'sma_20_4h', 'sma_20_slope_4h',
    'sma_50_4h', 'sma_50_slope_4h', 'candle_body_4h', 'upper_shadow_4h', 'lower_shadow_4h',
    'body_to_range_4h', 'volume_sma_4h', 'volume_ratio_4h', 'price_volume_4h', 'momentum_5_4h',
    'high_momentum_5_4h', 'low_momentum_5_4h', 'momentum_10_4h', 'high_momentum_10_4h',
    'low_momentum_10_4h', 'momentum_20_4h', 'high_momentum_20_4h', 'low_momentum_20_4h',
    'pred_4h', 'pred_direction_4h', 'volatility_regime_4h', 'vol_regime_low_4h', 'vol_regime_medium_4h',
    'vol_regime_high_4h', 'hour_4h', 'day_of_week_4h', 'asian_session_4h', 'london_session_4h',
    'ny_session_4h', 'rsi_divergence_4h', 'rsi_overbought_4h', 'rsi_oversold_4h', 'recent_high_4h',
    'recent_low_4h', 'dist_to_high_4h', 'dist_to_low_4h', 'is_doji_4h', 'is_hammer_4h',
    'is_shooting_star_4h', 'price_accel_15m', 'vol_adj_mom_1', 'vol_adj_mom_4', 'vol_adj_mom_16',
    'pred_consensus'
]

def create_technical_indicators(df, timeframe_suffix=''):
    """Create comprehensive technical indicators with type safety"""
    df = df.copy()
    
    suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""
    
    numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
    
    df[f'hl2{suffix}'] = (df['high'] + df['low']) / 2
    df[f'hlc3{suffix}'] = (df['high'] + df['low'] + df['close']) / 3
    df[f'ohlc4{suffix}'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    df[f'true_range{suffix}'] = np.maximum(
        np.maximum(df['high'] - df['low'], 
                  np.abs(df['high'] - df['close'].shift(1))),
        np.abs(df['low'] - df['close'].shift(1))
    )
    
    for period in [14, 21, 50]:
        df[f'atr_{period}{suffix}'] = df[f'true_range{suffix}'].rolling(period, min_periods=1).mean()
    
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8) # Added 1e-8 to prevent division by zero
        df[f'rsi_{period}{suffix}'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0)))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df[f'macd{suffix}'] = exp1 - exp2
    df[f'macd_signal{suffix}'] = df[f'macd{suffix}'].ewm(span=9, adjust=False).mean()
    df[f'macd_histogram{suffix}'] = df[f'macd{suffix}'] - df[f'macd_signal{suffix}']
    
    sma_20 = df['close'].rolling(20, min_periods=1).mean()
    std_20 = df['close'].rolling(20, min_periods=1).std()
    df[f'bb_upper{suffix}'] = sma_20 + (std_20 * 2)
    df[f'bb_lower{suffix}'] = sma_20 - (std_20 * 2)
    # Fixed FutureWarning by replacing fillna(method='ffill') with ffill()
    df[f'bb_position{suffix}'] = (df['close'] - df[f'bb_lower{suffix}']) / ((df[f'bb_upper{suffix}'] - df[f'bb_lower{suffix}']).replace(0, np.nan).ffill() + 1e-8)

    for period in [10, 20, 50]:
        df[f'sma_{period}{suffix}'] = df['close'].rolling(period, min_periods=1).mean()
        df[f'sma_{period}_slope{suffix}'] = df[f'sma_{period}{suffix}'].diff(5) # Consider if 5 is appropriate for all timeframes
    
    df[f'candle_body{suffix}'] = np.abs(df['close'] - df['open'])
    df[f'upper_shadow{suffix}'] = df['high'] - np.maximum(df['open'], df['close'])
    df[f'lower_shadow{suffix}'] = np.minimum(df['open'], df['close']) - df['low']
    df[f'body_to_range{suffix}'] = df[f'candle_body{suffix}'] / ((df['high'] - df['low']) + 1e-8)
    
    df[f'volume_sma{suffix}'] = df['tick_volume'].rolling(20, min_periods=1).mean()
    df[f'volume_ratio{suffix}'] = df['tick_volume'] / (df[f'volume_sma{suffix}'] + 1e-8)
    df[f'price_volume{suffix}'] = df['close'].pct_change() * df['tick_volume']
    
    for period in [5, 10, 20]:
        df[f'momentum_{period}{suffix}'] = df['close'].pct_change(period)
        df[f'high_momentum_{period}{suffix}'] = df['high'].pct_change(period)
        df[f'low_momentum_{period}{suffix}'] = df['low'].pct_change(period)
    
    print(f"Columns after technical indicators for {timeframe_suffix}: {df.columns.tolist()}")
    return df

def create_directional_features(df, pred_df, tf):
    """Create directional features with robust time column handling and aligned feature names"""
    df = df.copy()
    
    # Ensure 'time' is a column and reset index if needed
    if 'time' in df.columns and df.index.name == 'time':
        df = df.reset_index(drop=True)
    elif df.index.name == 'time':
        df = df.reset_index()
    elif 'time' not in df.columns:
        raise ValueError("DataFrame must have a 'time' column or DatetimeIndex for time-based features.")

    # Ensure pred_df has 'time' as a column
    if 'time' not in pred_df.columns:
        raise ValueError("Prediction DataFrame missing 'time' column")
    
    # Sort both DataFrames by time to ensure alignment
    df = df.sort_values('time').reset_index(drop=True)
    pred_df = pred_df.sort_values('time').reset_index(drop=True)

    # Debug prints to check alignment
    print(f"df shape: {df.shape}, unique times: {df['time'].nunique()}")
    print(f"pred_df shape: {pred_df.shape}, unique times: {pred_df['time'].nunique()}")

    try:
        df = df.merge(
            pred_df[['time', 'prediction']], 
            on='time', 
            how='left'  # Removed validate='one_to_one' for now
        )
    except KeyError as e:
        raise ValueError(f"Merge failed. Available columns in df: {df.columns.tolist()}. Pred_df columns: {pred_df.columns.tolist()}") from e
    
    df.rename(columns={'prediction': f'pred_{tf}'}, inplace=True)
    if df[f'pred_{tf}'].isna().any():
        print(f"Filling missing predictions for {tf} timeframe with 0")
        df[f'pred_{tf}'] = df[f'pred_{tf}'].fillna(0)
    
    df[f'pred_direction_{tf}'] = np.sign(df[f'pred_{tf}'])
    
    # Volatility regime
    atr_col_to_use = f'atr_14_{tf}'
    if atr_col_to_use not in df.columns:
        raise KeyError(f"Column {atr_col_to_use} not found in DataFrame for volatility regime in create_directional_features. Available: {df.columns.tolist()}")
    
    df[atr_col_to_use] = df[atr_col_to_use].fillna(df[atr_col_to_use].mean())
    
    vol_regime_name = 'volatility_regime' if tf == '15m' else f'volatility_regime_{tf}'
    df[vol_regime_name] = pd.cut(df[atr_col_to_use], bins=3, labels=['low', 'medium', 'high'], include_lowest=True)
    if 'medium' not in df[vol_regime_name].cat.categories:
        df[vol_regime_name] = df[vol_regime_name].cat.add_categories('medium')
    df[vol_regime_name] = df[vol_regime_name].fillna('medium')

    vol_low_name = 'vol_regime_low' if tf == '15m' else f'vol_regime_low_{tf}'
    vol_med_name = 'vol_regime_medium' if tf == '15m' else f'vol_regime_medium_{tf}'
    vol_high_name = 'vol_regime_high' if tf == '15m' else f'vol_regime_high_{tf}'
    df[vol_low_name] = (df[vol_regime_name] == 'low').astype(int)
    df[vol_med_name] = (df[vol_regime_name] == 'medium').astype(int)
    df[vol_high_name] = (df[vol_regime_name] == 'high').astype(int)
    
    # Time-based features
    current_hour_col = f'_internal_hour_{tf}'
    if 'time' in df.columns:
        dt_series = pd.to_datetime(df['time'])
        hour_name = 'hour' if tf == '15m' else f'hour_{tf}'
        dow_name = 'day_of_week' if tf == '15m' else f'day_of_week_{tf}'
        df[hour_name] = dt_series.dt.hour
        df[dow_name] = dt_series.dt.dayofweek
        df[current_hour_col] = df[hour_name]
    else:
        raise ValueError("DataFrame must have a 'time' column or DatetimeIndex for time-based features.")

    # Market session features
    asian_name = 'asian_session' if tf == '15m' else f'asian_session_{tf}'
    london_name = 'london_session' if tf == '15m' else f'london_session_{tf}'
    ny_name = 'ny_session' if tf == '15m' else f'ny_session_{tf}'
    df[asian_name] = ((df[current_hour_col] >= 0) & (df[current_hour_col] < 8)).astype(int)
    df[london_name] = ((df[current_hour_col] >= 8) & (df[current_hour_col] < 16)).astype(int)
    df[ny_name] = ((df[current_hour_col] >= 16) & (df[current_hour_col] < 24)).astype(int)
    df.drop(columns=[current_hour_col], inplace=True, errors='ignore')

    # Compute RSI divergence, recent high/low, etc.
    rsi_col_for_divergence = f'rsi_14_{tf}'
    body_to_range_col = f'body_to_range_{tf}'
    lower_shadow_col = f'lower_shadow_{tf}'
    upper_shadow_col = f'upper_shadow_{tf}'
    candle_body_col = f'candle_body_{tf}'

    required_cols_for_extra_features = [
        rsi_col_for_divergence, 'high', 'low', 'close', body_to_range_col,
        lower_shadow_col, upper_shadow_col, candle_body_col
    ]
    for req_col in required_cols_for_extra_features:
        if req_col not in df.columns:
            raise KeyError(f"Required column '{req_col}' for extra feature calculation is missing in DataFrame for timeframe {tf}. Available: {df.columns.tolist()}")

    if tf == '15m':
        df['rsi_divergence'] = np.abs(df[rsi_col_for_divergence] - 50) / (50 + 1e-8)
        df['rsi_overbought'] = (df[rsi_col_for_divergence] > 70).astype(int)
        df['rsi_oversold'] = (df[rsi_col_for_divergence] < 30).astype(int)
        
        df['recent_high'] = df['high'].rolling(50, min_periods=1).max()
        df['recent_low'] = df['low'].rolling(50, min_periods=1).min()
        df['dist_to_high'] = (df['recent_high'] - df['close']) / (df['close'] + 1e-8)
        df['dist_to_low'] = (df['close'] - df['recent_low']) / (df['close'] + 1e-8)
        
        df['is_doji'] = (df[body_to_range_col] < 0.1).astype(int)
        df['is_hammer'] = ((df[lower_shadow_col] > 2 * df[candle_body_col]) & 
                          (df[upper_shadow_col] < df[candle_body_col])).astype(int)
        df['is_shooting_star'] = ((df[upper_shadow_col] > 2 * df[candle_body_col]) & 
                                 (df[lower_shadow_col] < df[candle_body_col])).astype(int)
    else:
        df[f'rsi_divergence_{tf}'] = np.abs(df[rsi_col_for_divergence] - 50) / (50 + 1e-8)
        df[f'rsi_overbought_{tf}'] = (df[rsi_col_for_divergence] > 70).astype(int)
        df[f'rsi_oversold_{tf}'] = (df[rsi_col_for_divergence] < 30).astype(int)
        
        df[f'recent_high_{tf}'] = df['high'].rolling(50, min_periods=1).max()
        df[f'recent_low_{tf}'] = df['low'].rolling(50, min_periods=1).min()
        df[f'dist_to_high_{tf}'] = (df[f'recent_high_{tf}'] - df['close']) / (df['close'] + 1e-8)
        df[f'dist_to_low_{tf}'] = (df['close'] - df[f'recent_low_{tf}']) / (df['close'] + 1e-8)
        
        df[f'is_doji_{tf}'] = (df[body_to_range_col] < 0.1).astype(int)
        df[f'is_hammer_{tf}'] = ((df[lower_shadow_col] > 2 * df[candle_body_col]) & 
                                (df[upper_shadow_col] < df[candle_body_col])).astype(int)
        df[f'is_shooting_star_{tf}'] = ((df[upper_shadow_col] > 2 * df[candle_body_col]) & 
                                       (df[lower_shadow_col] < df[candle_body_col])).astype(int)
    
    return df

class EnhancedDataValidator:
    """Validates MT5 data quality with fixed datetime handling"""
    @staticmethod
    def validate_data(df, timeframe, expected_bars):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        if len(df) < expected_bars * 0.9: # Allow for slight discrepancies, e.g. start of market
            raise ValueError(f"Insufficient data: {len(df)}/{expected_bars} bars for {timeframe}")
        
        max_allowed_gap = {
            '15m': timedelta(minutes=60), # Max 4 bars missing
            '1h': timedelta(hours=4),     # Max 4 bars missing
            '4h': timedelta(hours=16)     # Max 4 bars missing
        }[timeframe]
        
        if 'time' not in df.columns:
            raise ValueError("DataFrame missing 'time' column for validation.")
            
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
             raise TypeError(f"'time' column is not a datetime type. Is {df['time'].dtype}")

        last_bar_time = df['time'].iloc[-1]
        current_time = datetime.now(pytz.utc) # Ensure current time is UTC
        
        # Ensure last_bar_time is timezone-aware (it should be if from pd.to_datetime(..., utc=True))
        if last_bar_time.tzinfo is None or last_bar_time.tzinfo.utcoffset(last_bar_time) is None:
            print(f"Warning: last_bar_time for {timeframe} is naive. Localizing to UTC. This should not happen if data fetching is correct.")
            last_bar_time = pytz.utc.localize(last_bar_time)
        else: # If already aware, ensure it's UTC
            last_bar_time = last_bar_time.astimezone(pytz.utc)

        if current_time - last_bar_time > max_allowed_gap:
            raise ValueError(f"Stale data detected for {timeframe}. Last bar: {last_bar_time}, Current time: {current_time}, Gap: {current_time - last_bar_time}")
        
        if 'tick_volume' not in df.columns:
            raise ValueError(f"DataFrame missing 'tick_volume' column for {timeframe}.")
        if df['tick_volume'].iloc[-1] == 0 and timeframe == '15m': # More critical for smaller TFs
            # Could be market close, or weekend. Allow if not too old.
            if current_time - last_bar_time < timedelta(minutes=20): # If very recent bar has 0 volume
                 print(f"Warning: Zero volume in latest bar for {timeframe} at {last_bar_time}. Possible inactive market.")
        
        if 'close' not in df.columns:
            raise ValueError(f"DataFrame missing 'close' column for {timeframe}.")
        price_change = df['close'].pct_change().abs()
        if (price_change > 0.05).any(): # Check entire series for spikes
            print(f"Warning: Extreme price movement (>5%) detected in {timeframe} data.")
        
        return True

def fetch_mt5_data(symbol, timeframe, num_bars=200):
    """Fetches data from MT5, validates it, and returns a DataFrame with 'time' as a column."""
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
    
    timeframe_map = {
        '15m': mt5.TIMEFRAME_M15,
        '1h': mt5.TIMEFRAME_H1,
        '4h': mt5.TIMEFRAME_H4
    }
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, num_bars)

    if rates is None or len(rates) == 0:
        mt5.shutdown() # Ensure shutdown on error
        raise ValueError(f"No data received for {symbol} {timeframe}. MT5 error: {mt5.last_error()}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) # 'time' is a column, UTC-aware
    
    # Ensure standard column names expected by subsequent functions
    df.rename(columns={'tick_volume': 'tick_volume', 'real_volume': 'real_volume'}, inplace=True)
    if 'real_volume' not in df.columns: # MT5 might not provide real_volume for all symbols/brokers
        df['real_volume'] = 0 
    if 'spread' not in df.columns: # spread is per tick, not per bar from copy_rates
        df['spread'] = 0 # Or calculate from tick data if available, or use typical spread

    try:
        # Validate data while 'time' is a column
        EnhancedDataValidator.validate_data(df, timeframe, num_bars)
    except Exception as e:
        mt5.shutdown() # Ensure shutdown on error
        print(f"Data validation failed for {timeframe}: {e}")
        raise
    
    return df # DataFrame returned with 'time' as a column

def load_model_safely(model_path, model_type='xgb'):
    """Load model with multiple fallback strategies"""
    try:
        if model_type == 'xgb':
            model = xgb.XGBClassifier() if 'classifier' in str(model_path).lower() else xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        elif model_type == 'joblib':
            return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path} with primary method ({model_type}): {e}")
        try:
            if model_type == 'xgb': # Try joblib as fallback for xgb
                print(f"Attempting fallback joblib load for {model_path}")
                return joblib.load(model_path)
            elif model_type == 'joblib': # Try xgb as fallback for joblib
                print(f"Attempting fallback xgb load for {model_path}")
                model = xgb.XGBClassifier() if 'classifier' in str(model_path).lower() else xgb.XGBRegressor()
                model.load_model(model_path)
                return model
        except Exception as e2:
            print(f"Error loading model {model_path} with fallback method: {e2}")
            raise ValueError(f"Failed to load model from {model_path} using both methods.") from e2

from pathlib import Path

base_path = Path(__file__).parent

# Load models with enhanced error handling
try:
    meta_learner = load_model_safely(str(base_path / 'meta_learner_optv3.pkl'), 'joblib')
    base_model_15m = load_model_safely(str(base_path / 'xgboost_15m.json'), 'xgb')
    base_model_1h = load_model_safely(str(base_path / 'xgboost_1h.json'), 'xgb')
    base_model_4h = load_model_safely(str(base_path / 'xgboost_4h.json'), 'xgb')
    print("All models loaded successfully")
except Exception as e:
    print(f"Critical error loading models: {e}")
    raise SystemExit(f"Model loading failed: {e}") # Exit if models can't load

def prepare_base_features(df, timeframe):
    df_feat = df.copy()
    
    high_low = df_feat['high'] - df_feat['low']
    high_close_prev = np.abs(df_feat['high'] - df_feat['close'].shift(1))
    low_close_prev = np.abs(df_feat['low'] - df_feat['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    df_feat['atr'] = tr.rolling(window=14, min_periods=1).mean()
    
    delta = df_feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8) 
    df_feat['rsi_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df_feat['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_feat['close'].ewm(span=26, adjust=False).mean()
    df_feat['macd'] = exp1 - exp2
    df_feat['macd_signal'] = df_feat['macd'].ewm(span=9, adjust=False).mean()
    df_feat['macd_hist'] = df_feat['macd'] - df_feat['macd_signal']
    
    if 'time' in df_feat.columns:
        dt_series = pd.to_datetime(df_feat['time'])
        df_feat['hour'] = dt_series.dt.hour
    else: 
        df_feat['hour'] = 0 
        print(f"Warning: No time information for 'hour' in prepare_base_features for {timeframe}. Defaulting to 0.")

    df_feat['session'] = pd.cut(df_feat['hour'],
                          bins=[-1, 7, 15, 23], 
                          labels=[0, 1, 2], 
                          ordered=False) 
    df_feat['session'] = df_feat['session'].fillna(0).astype(int)

    df_feat['candle_body'] = df_feat['close'] - df_feat['open'] 
    df_feat['upper_shadow'] = df_feat['high'] - np.maximum(df_feat['open'], df_feat['close'])
    df_feat['lower_shadow'] = np.minimum(df_feat['open'], df_feat['close']) - df_feat['low']
    
    if timeframe == '15m':
        df_feat['intraday_momentum'] = df_feat['close'].pct_change(4) 
        df_feat['volume_momentum'] = df_feat['tick_volume'].pct_change(4)
        df_feat['lagged_return_1'] = df_feat['close'].pct_change().shift(1)
        df_feat['lagged_return_2'] = df_feat['close'].pct_change().shift(2)
        df_feat['lagged_return_3'] = df_feat['close'].pct_change().shift(3)
        feature_cols = base_feature_cols_15m
    elif timeframe == '1h':
        df_feat['intraday_momentum'] = df_feat['close'].pct_change(8) 
        feature_cols = base_feature_cols_1h
    elif timeframe == '4h':
        df_feat['intraday_momentum'] = df_feat['close'].pct_change(6) 
        feature_cols = base_feature_cols_4h
    else:
        raise ValueError(f"Unknown timeframe for base features: {timeframe}")

    for col in feature_cols:
        if col not in df_feat.columns:
            print(f"Warning: Base feature '{col}' for {timeframe} was missing. Adding it as 0.")
            df_feat[col] = 0
    
    df_feat = df_feat.fillna(0) 
    
    cols_to_return = feature_cols[:] 
    if 'time' in df_feat.columns: 
        if 'time' not in cols_to_return:
             cols_to_return.append('time')
    
    missing_in_df_feat = [col for col in cols_to_return if col not in df_feat.columns]
    if missing_in_df_feat:
        raise KeyError(f"Columns {missing_in_df_feat} are expected but not found in df_feat after processing for {timeframe}. Available: {df_feat.columns.tolist()}")

    return df_feat[cols_to_return]


def prepare_meta_features(df_15m_orig, df_1h_orig, df_4h_orig):
    df_15m = df_15m_orig.copy()
    df_1h = df_1h_orig.copy()
    df_4h = df_4h_orig.copy()

    for df_item in [df_15m, df_1h, df_4h]:
        if df_item.index.name == 'time':
            df_item.reset_index(inplace=True)
        elif 'time' not in df_item.columns:
            raise ValueError("DataFrame missing 'time' column at start of prepare_meta_features")

    df_15m_base_feats = prepare_base_features(df_15m, '15m')
    df_1h_base_feats = prepare_base_features(df_1h, '1h')
    df_4h_base_feats = prepare_base_features(df_4h, '4h')

    # Ensure base feature columns for prediction exist
    pred_15m_input = df_15m_base_feats[base_feature_cols_15m]
    pred_1h_input = df_1h_base_feats[base_feature_cols_1h]
    pred_4h_input = df_4h_base_feats[base_feature_cols_4h]
    
    # Convert session to int if it's categorical for base model prediction
    if 'session' in pred_15m_input.columns and isinstance(pred_15m_input['session'].dtype, pd.CategoricalDtype): # MODIFIED
        pred_15m_input['session'] = pred_15m_input['session'].astype(int)
    if 'session' in pred_1h_input.columns and isinstance(pred_1h_input['session'].dtype, pd.CategoricalDtype): # MODIFIED
        pred_1h_input['session'] = pred_1h_input['session'].astype(int)
    if 'session' in pred_4h_input.columns and isinstance(pred_4h_input['session'].dtype, pd.CategoricalDtype): # MODIFIED
        pred_4h_input['session'] = pred_4h_input['session'].astype(int)

    # Get base model predictions
    pred_15m_arr = base_model_15m.predict(pred_15m_input) # Renamed to avoid clash later
    pred_1h_arr = base_model_1h.predict(pred_1h_input)
    pred_4h_arr = base_model_4h.predict(pred_4h_input)

    # Store predictions in DataFrames with 'prediction' column name for create_directional_features
    pred_df_15m = pd.DataFrame({'time': df_15m_base_feats['time'], 'prediction': pred_15m_arr}) # MODIFIED
    pred_df_1h = pd.DataFrame({'time': df_1h_base_feats['time'], 'prediction': pred_1h_arr})   # MODIFIED
    pred_df_4h = pd.DataFrame({'time': df_4h_base_feats['time'], 'prediction': pred_4h_arr})   # MODIFIED

    df_15m_tech = create_technical_indicators(df_15m, '15m')
    df_1h_tech = create_technical_indicators(df_1h, '1h')
    df_4h_tech = create_technical_indicators(df_4h, '4h')

    df_15m_final = create_directional_features(df_15m_tech, pred_df_15m, '15m')
    df_1h_final = create_directional_features(df_1h_tech, pred_df_1h, '1h')
    df_4h_final = create_directional_features(df_4h_tech, pred_df_4h, '4h')

    merged = df_15m_final.merge(
        df_1h_final, on='time', how='left', suffixes=('', '_DROP1H')
    ).merge(
        df_4h_final, on='time', how='left', suffixes=('', '_DROP4H')
    )
    
    cols_to_drop = [col for col in merged.columns if '_DROP1H' in col or '_DROP4H' in col]
    merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    merged['price_accel_15m'] = merged['close'].pct_change().diff()
    if 'atr_14_15m' not in merged.columns: 
        raise KeyError("'atr_14_15m' is missing from merged DataFrame before vol_adj_mom calculation.")
    for period in [1, 4, 16]:
        merged[f'vol_adj_mom_{period}'] = (
            merged['close'].pct_change(period) / 
            (merged['atr_14_15m'].rolling(period, min_periods=1).mean() + 1e-8)
        )

    pred_cols_for_consensus = ['pred_15m', 'pred_1h', 'pred_4h'] # These are from create_directional_features
    if all(col in merged.columns for col in pred_cols_for_consensus):
        merged['pred_consensus'] = merged[pred_cols_for_consensus].mean(axis=1)
    elif 'pred_15m' in merged.columns: 
        print("Warning: Not all prediction columns available for consensus, using 'pred_15m'.")
        merged['pred_consensus'] = merged['pred_15m']
    else: 
        print("Warning: 'pred_15m' also missing for consensus. Defaulting 'pred_consensus' to 0.")
        merged['pred_consensus'] = 0

    categorical_meta_cols_to_check = ['volatility_regime', 'volatility_regime_1h', 'volatility_regime_4h']
    
    for col in meta_feature_cols:
        if col not in merged.columns:
            print(f"Warning: Meta feature '{col}' was missing. Adding it.")
            if col in categorical_meta_cols_to_check:
                merged[col] = pd.Series(index=merged.index, dtype='category')
                default_categories = ['low', 'medium', 'high']
                # This logic for adding categories might be slightly off if series is already category
                # but if col not in merged.columns, it's newly created as category.
                current_categories = merged[col].cat.categories.tolist()
                new_categories = [cat for cat in default_categories if cat not in current_categories]
                if new_categories:
                     merged[col] = merged[col].cat.add_categories(new_categories)
                merged[col] = merged[col].fillna('medium') # Fill all with 'medium' as it was just created
            else:
                merged[col] = 0
    
    for col in merged.columns:
        if col in categorical_meta_cols_to_check and isinstance(merged[col].dtype, pd.CategoricalDtype): # MODIFIED
            if merged[col].isnull().any():
                print(f"Filling NaNs in categorical column '{col}' with 'medium'.")
                if 'medium' not in merged[col].cat.categories:
                    merged[col] = merged[col].cat.add_categories(['medium'])
                merged[col] = merged[col].fillna('medium')
        elif pd.api.types.is_numeric_dtype(merged[col]): # Can leave as is or use isinstance with NumericDtype
            if merged[col].isnull().any():
                merged[col] = merged[col].fillna(0)

    try:
        final_features = merged[meta_feature_cols].copy() # MODIFIED: added .copy()
    except KeyError as e:
        missing_cols = [col for col in meta_feature_cols if col not in merged.columns]
        print(f"Error: Columns required by meta_feature_cols are missing from the 'merged' DataFrame: {missing_cols}")
        print(f"Available columns in 'merged': {merged.columns.tolist()}")
        raise e

    allowed_dtypes = ['int64', 'float64', 'int32', 'float32', 'bool', 'category']
    # For pandas >= 1.0, pd.CategoricalDtype should be used for isinstance checks
    # For older pandas, final_features.select_dtypes(include=['category']) works.
    # The allowed_dtypes list with 'category' string should work with select_dtypes.
    
    non_allowed_cols = final_features.select_dtypes(exclude=allowed_dtypes).columns
    
    if len(non_allowed_cols) > 0:
        print(f"Error: Non-allowed dtypes found in final features: {non_allowed_cols.tolist()}")
        print(final_features[non_allowed_cols].dtypes)
        for col_name in non_allowed_cols:
            # Check if it's one of the specific categorical columns that might be object type
            if col_name in ['volatility_regime', 'volatility_regime_1h', 'volatility_regime_4h'] and final_features[col_name].dtype == 'object':
                print(f"Attempting to convert problematic column '{col_name}' to category.")
                # Ensure categories are consistent if converting here
                expected_cats = ['low', 'medium', 'high']
                final_features[col_name] = pd.Categorical(final_features[col_name], categories=expected_cats, ordered=False)
                # If there were NaNs before, fill them with 'medium' after conversion
                if 'medium' not in final_features[col_name].cat.categories: # Should not happen if cats are predefined
                    final_features[col_name] = final_features[col_name].cat.add_categories(['medium'])
                final_features[col_name] = final_features[col_name].fillna('medium')
            else:
                raise ValueError(f"Column '{col_name}' has non-allowed dtype {final_features[col_name].dtype}. All features must be numeric or categorical for XGBoost prediction.")
        
        non_allowed_cols_after_fix = final_features.select_dtypes(exclude=allowed_dtypes).columns
        if len(non_allowed_cols_after_fix) > 0:
            raise ValueError(f"Still have non-allowed dtypes after attempted fix: {non_allowed_cols_after_fix.tolist()}. Dtypes: {final_features[non_allowed_cols_after_fix].dtypes}")
            
    # Add individual base model predictions (original numpy arrays) as additional columns
    # These are for inspection or use in main loop, not for meta_learner input here (as they are not in meta_feature_cols)
    final_features['base_pred_15m'] = pred_15m_arr
    final_features['base_pred_1h'] = pred_1h_arr
    final_features['base_pred_4h'] = pred_4h_arr

    return final_features



def prepare_meta_features(df_15m_orig, df_1h_orig, df_4h_orig):
    df_15m = df_15m_orig.copy()
    df_1h = df_1h_orig.copy()
    df_4h = df_4h_orig.copy()

    for df_item in [df_15m, df_1h, df_4h]:
        if df_item.index.name == 'time':
            df_item.reset_index(inplace=True)
        elif 'time' not in df_item.columns:
            raise ValueError("DataFrame missing 'time' column at start of prepare_meta_features")

    df_15m_base_feats = prepare_base_features(df_15m, '15m')
    df_1h_base_feats = prepare_base_features(df_1h, '1h')
    df_4h_base_feats = prepare_base_features(df_4h, '4h')

    pred_15m_input = df_15m_base_feats[base_feature_cols_15m]
    pred_1h_input = df_1h_base_feats[base_feature_cols_1h]
    pred_4h_input = df_4h_base_feats[base_feature_cols_4h]
    
    if 'session' in pred_15m_input.columns and isinstance(pred_15m_input['session'].dtype, pd.CategoricalDtype):
        pred_15m_input['session'] = pred_15m_input['session'].astype(int)
    if 'session' in pred_1h_input.columns and isinstance(pred_1h_input['session'].dtype, pd.CategoricalDtype):
        pred_1h_input['session'] = pred_1h_input['session'].astype(int)
    if 'session' in pred_4h_input.columns and isinstance(pred_4h_input['session'].dtype, pd.CategoricalDtype):
        pred_4h_input['session'] = pred_4h_input['session'].astype(int)

    pred_15m_arr = base_model_15m.predict(pred_15m_input)
    pred_1h_arr = base_model_1h.predict(pred_1h_input)
    pred_4h_arr = base_model_4h.predict(pred_4h_input)

    pred_df_15m = pd.DataFrame({'time': df_15m_base_feats['time'], 'prediction': pred_15m_arr})
    pred_df_1h = pd.DataFrame({'time': df_1h_base_feats['time'], 'prediction': pred_1h_arr})
    pred_df_4h = pd.DataFrame({'time': df_4h_base_feats['time'], 'prediction': pred_4h_arr})

    df_15m_tech = create_technical_indicators(df_15m, '15m')
    df_1h_tech = create_technical_indicators(df_1h, '1h')
    df_4h_tech = create_technical_indicators(df_4h, '4h')

    df_15m_final = create_directional_features(df_15m_tech, pred_df_15m, '15m')
    df_1h_final = create_directional_features(df_1h_tech, pred_df_1h, '1h')
    df_4h_final = create_directional_features(df_4h_tech, pred_df_4h, '4h')

    merged = df_15m_final.merge(
        df_1h_final, on='time', how='left', suffixes=('', '_DROP1H')
    ).merge(
        df_4h_final, on='time', how='left', suffixes=('', '_DROP4H')
    )
    
    cols_to_drop = [col for col in merged.columns if '_DROP1H' in col or '_DROP4H' in col]
    merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    merged['price_accel_15m'] = merged['close'].pct_change().diff()
    if 'atr_14_15m' not in merged.columns: 
        raise KeyError("'atr_14_15m' is missing from merged DataFrame before vol_adj_mom calculation.")
    for period in [1, 4, 16]:
        merged[f'vol_adj_mom_{period}'] = (
            merged['close'].pct_change(period) / 
            (merged['atr_14_15m'].rolling(period, min_periods=1).mean() + 1e-8)
        )

    pred_cols_for_consensus = ['pred_15m', 'pred_1h', 'pred_4h']
    if all(col in merged.columns for col in pred_cols_for_consensus):
        merged['pred_consensus'] = merged[pred_cols_for_consensus].mean(axis=1)
    elif 'pred_15m' in merged.columns: 
        merged['pred_consensus'] = merged['pred_15m']
    else: 
        merged['pred_consensus'] = 0

    categorical_meta_cols_to_check = ['volatility_regime', 'volatility_regime_1h', 'volatility_regime_4h']
    
    for col in meta_feature_cols:
        if col not in merged.columns:
            if col in categorical_meta_cols_to_check:
                merged[col] = pd.Series(index=merged.index, dtype='category')
                merged[col] = merged[col].cat.add_categories(['low', 'medium', 'high'])
            else:
                merged[col] = 0

    # ========================================================================
    # START: FINAL CLEANUP SECTION (THE MAIN FIX)
    # This ensures the returned DataFrame is completely free of NaNs.
    # It intelligently fills based on data type.
    # ========================================================================
    merged = merged.ffill() # Forward fill to handle gaps from merging

    for col in merged.columns:
        # Fill categorical columns with 'medium'
        if pd.api.types.is_categorical_dtype(merged[col]):
            if merged[col].isnull().any():
                if 'medium' in merged[col].cat.categories:
                    merged[col] = merged[col].fillna('medium')
                else: # Fallback if 'medium' somehow isn't a category
                    merged[col] = merged[col].fillna(merged[col].cat.categories[0])
        
        # Fill numeric columns with 0
        elif pd.api.types.is_numeric_dtype(merged[col]):
            if merged[col].isnull().any():
                merged[col] = merged[col].fillna(0)
    # ========================================================================
    # END: FINAL CLEANUP SECTION
    # ========================================================================

    try:
        final_features = merged[meta_feature_cols].copy()
    except KeyError as e:
        missing_cols = [col for col in meta_feature_cols if col not in merged.columns]
        raise KeyError(f"Columns required by meta_feature_cols are missing: {missing_cols}") from e
            
    final_features['base_pred_15m'] = pred_15m_arr
    final_features['base_pred_1h'] = pred_1h_arr
    final_features['base_pred_4h'] = pred_4h_arr

    return final_features

# In predict.py

def get_latest_prediction():
    """Fetch data, prepare features, and return latest prediction results and historical data as a dict."""
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        fetch_count = LOOKBACK_BARS + 60
        df_15m = fetch_mt5_data(SYMBOL, '15m', fetch_count)
        df_1h = fetch_mt5_data(SYMBOL, '1h', fetch_count)
        df_4h = fetch_mt5_data(SYMBOL, '4h', fetch_count)

        features_df = prepare_meta_features(df_15m, df_1h, df_4h)
        if features_df.empty:
            return {"error": "Feature preparation resulted in empty DataFrame"}

        X_latest_features = features_df.tail(1)

        X_for_prediction = X_latest_features[meta_feature_cols]
        raw_predictions = meta_learner.predict(X_for_prediction)

        if isinstance(raw_predictions, np.ndarray) and raw_predictions.ndim > 0:
            y_pred_class = raw_predictions.item(0)
        else:
            y_pred_class = raw_predictions

        if hasattr(meta_learner, 'predict_proba'):
            pred_probs = meta_learner.predict_proba(X_for_prediction)[0]
            meta_pred_direction = int(np.argmax(pred_probs))
        else:
            meta_pred_direction = int(y_pred_class)

        base_pred_15m = X_latest_features['base_pred_15m'].values[0]
        base_pred_1h = X_latest_features['base_pred_1h'].values[0]
        base_pred_4h = X_latest_features['base_pred_4h'].values[0]

        def interpret_direction(pred):
            return 1 if pred > 0.48 else -1

        base_15m_direction = interpret_direction(base_pred_15m)
        base_1h_direction = interpret_direction(base_pred_1h)
        base_4h_direction = interpret_direction(base_pred_4h)

        trade_signal = 0
        if meta_pred_direction == 1:
            trade_signal = 1
        elif meta_pred_direction == 2:
            trade_signal = -1

        current_price = df_15m['close'].iloc[-1] if not df_15m.empty else None
        direction, tp_price, sl_price, meta_pred_price = 'HOLD', None, None, current_price

        if current_price is not None:
            if trade_signal == 1:
                direction = 'BUY'
                tp_price = current_price + PIP_VALUE
                sl_price = current_price - SL_PIPS
                meta_pred_price = current_price + (PIP_VALUE / 2)
            elif trade_signal == -1:
                direction = 'SELL'
                tp_price = current_price - PIP_VALUE
                sl_price = current_price + SL_PIPS
                meta_pred_price = current_price - (PIP_VALUE / 2)

            base_15m_pred_price = current_price + (base_15m_direction * PIP_VALUE)
            base_1h_pred_price = current_price + (base_1h_direction * PIP_VALUE)
            base_4h_pred_price = current_price + (base_4h_direction * PIP_VALUE)
        else:
            base_15m_pred_price, base_1h_pred_price, base_4h_pred_price = None, None, None

        # --- NEW: Prepare historical and metrics data for frontend ---
        historical_data = []
        if not df_15m.empty:
            # Get last 20 candles for the chart
            chart_data = df_15m.tail(20)
            historical_data = chart_data[['time', 'close']].to_dict('records')
            # Convert datetime objects to ISO 8601 strings for JSON compatibility
            for record in historical_data:
                record['time'] = record['time'].isoformat()
        
        # Get latest metrics from the feature dataframe
        latest_metrics = {
            "rsi": X_latest_features['rsi_14_15m'].iloc[0] if 'rsi_14_15m' in X_latest_features else 50,
            "macd_hist": X_latest_features['macd_histogram_15m'].iloc[0] if 'macd_histogram_15m' in X_latest_features else 0,
            "bb_pos": X_latest_features['bb_position_15m'].iloc[0] if 'bb_position_15m' in X_latest_features else 0.5,
            "atr": X_latest_features['atr_14_15m'].iloc[0] if 'atr_14_15m' in X_latest_features else 0,
        }
        # --- END NEW SECTION ---

        result = {
            # Core prediction
            "meta_pred_price": meta_pred_price,
            "direction": direction,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "current_price": current_price,
            "trade_signal": trade_signal,
            # Base model data
            "base_15m_pred_price": base_15m_pred_price,
            "base_1h_pred_price": base_1h_pred_price,
            "base_4h_pred_price": base_4h_pred_price,
            "base_15m_direction": base_15m_direction,
            "base_1h_direction": base_1h_direction,
            "base_4h_direction": base_4h_direction,
            # NEW: Data for frontend visualizations
            "historical_data": historical_data,
            "latest_metrics": latest_metrics,
        }
        return result

    except Exception as e:
        import traceback
        print(f"An error occurred in get_latest_prediction: {e}")
        print(traceback.format_exc())
        return {"error": f"Backend Error: {e}"}
    finally:
        gc.collect()
        if mt5.terminal_info() is not None:
            mt5.shutdown()
