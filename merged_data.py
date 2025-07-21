import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_technical_indicators(df, timeframe_suffix=''):
    """Create comprehensive technical indicators (copied from meta_features.py)"""
    df = df.copy()
    
    # Add underscore to suffix if not empty
    suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""
    
    # Basic price features
    df[f'hl2{suffix}'] = (df['high'] + df['low']) / 2
    df[f'hlc3{suffix}'] = (df['high'] + df['low'] + df['close']) / 3
    df[f'ohlc4{suffix}'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Volatility indicators
    df[f'true_range{suffix}'] = np.maximum(
        np.maximum(df['high'] - df['low'], 
                  np.abs(df['high'] - df['close'].shift(1))),
        np.abs(df['low'] - df['close'].shift(1))
    )
    d
    # ATR with multiple periods
    for period in [14, 21, 50]:
        df[f'atr_{period}{suffix}'] = df[f'true_range{suffix}'].rolling(period, min_periods=1).mean()
    
    # RSI with multiple periods
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        df[f'rsi_{period}{suffix}'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0)))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df[f'macd{suffix}'] = exp1 - exp2
    df[f'macd_signal{suffix}'] = df[f'macd{suffix}'].ewm(span=9).mean()
    df[f'macd_histogram{suffix}'] = df[f'macd{suffix}'] - df[f'macd_signal{suffix}']
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(20, min_periods=1).mean()
    std_20 = df['close'].rolling(20, min_periods=1).std()
    df[f'bb_upper{suffix}'] = sma_20 + (std_20 * 2)
    df[f'bb_lower{suffix}'] = sma_20 - (std_20 * 2)
    df[f'bb_position{suffix}'] = (df['close'] - df[f'bb_lower{suffix}']) / (df[f'bb_upper{suffix}'] - df[f'bb_lower{suffix}']).replace(0, np.nan).fillna(0)
    
    # Moving averages and slopes
    for period in [10, 20, 50]:
        df[f'sma_{period}{suffix}'] = df['close'].rolling(period, min_periods=1).mean()
        df[f'sma_{period}_slope{suffix}'] = df[f'sma_{period}{suffix}'].diff(5)
    
    # Price action features
    df[f'candle_body{suffix}'] = np.abs(df['close'] - df['open'])
    df[f'upper_shadow{suffix}'] = df['high'] - np.maximum(df['open'], df['close'])
    df[f'lower_shadow{suffix}'] = np.minimum(df['open'], df['close']) - df['low']
    df[f'body_to_range{suffix}'] = df[f'candle_body{suffix}'] / (df['high'] - df['low'] + 1e-8)
    
    # Volume features
    df[f'volume_sma{suffix}'] = df['tick_volume'].rolling(20, min_periods=1).mean()
    df[f'volume_ratio{suffix}'] = df['tick_volume'] / (df[f'volume_sma{suffix}'] + 1e-8)
    df[f'price_volume{suffix}'] = df['close'].pct_change() * df['tick_volume']
    
    # Momentum features
    for period in [5, 10, 20]:
        df[f'momentum_{period}{suffix}'] = df['close'].pct_change(period)
        df[f'high_momentum_{period}{suffix}'] = df['high'].pct_change(period)
        df[f'low_momentum_{period}{suffix}'] = df['low'].pct_change(period)
    
    print(f"Columns after technical indicators for {timeframe_suffix}: {df.columns.tolist()}")
    return df

def create_directional_features(df, pred_df, tf):
    """Create directional features using the prediction DataFrame for the same timeframe"""
    df = df.copy()
    
    # Log time ranges for debugging
    print(f"df time range: {df['time'].min()} to {df['time'].max()}")
    print(f"pred_df time range: {pred_df['time'].min()} to {pred_df['time'].max()}")
    
    # Merge predictions
    df = df.merge(pred_df[['time', 'prediction']], on='time', how='left')
    df.rename(columns={'prediction': f'pred_{tf}'}, inplace=True)
    
    # Check for NaN values in prediction column
    pred_col = f'pred_{tf}'
    nan_ratio = df[pred_col].isna().mean()
    print(f"NaN ratio for {pred_col}: {nan_ratio:.4f}")
    if nan_ratio == 1.0:
        print(f"Warning: {pred_col} is entirely NaN after merge. Setting to 0 as a fallback.")
        df[pred_col] = 0
    else:
        # Fill missing predictions with forward fill then backward fill
        df[pred_col] = df[pred_col].fillna(method='ffill').fillna(method='bfill')
    
    # Additional check: if still NaN, set to 0
    if df[pred_col].isna().any():
        print(f"Warning: Some NaN values remain in {pred_col} after fill. Setting remaining NaNs to 0.")
        df[pred_col] = df[pred_col].fillna(0)
    
    # Prediction-based features
    df[f'pred_direction_{tf}'] = np.sign(df[pred_col])
    
    # Validate presence of technical indicators
    required_cols = [f'atr_14_{tf}', f'rsi_14_{tf}', f'body_to_range_{tf}', f'lower_shadow_{tf}', f'upper_shadow_{tf}', f'candle_body_{tf}']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df: {missing_cols}. Ensure technical indicators are generated correctly.")
    
    # Volatility regime classification
    df['volatility_regime'] = pd.cut(df[f'atr_14_{tf}'], bins=3, labels=['low', 'medium', 'high'])
    df['vol_regime_low'] = (df['volatility_regime'] == 'low').astype(int)
    df['vol_regime_medium'] = (df['volatility_regime'] == 'medium').astype(int)
    df['vol_regime_high'] = (df['volatility_regime'] == 'high').astype(int)
    
    # Time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Market session features
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    
    # RSI divergence features
    df['rsi_divergence'] = np.abs(df[f'rsi_14_{tf}'] - 50) / 50  # Distance from neutral
    df['rsi_overbought'] = (df[f'rsi_14_{tf}'] > 70).astype(int)
    df['rsi_oversold'] = (df[f'rsi_14_{tf}'] < 30).astype(int)
    
    # Support/Resistance proximity (using recent highs/lows)
    df['recent_high'] = df['high'].rolling(50).max()
    df['recent_low'] = df['low'].rolling(50).min()
    df['dist_to_high'] = (df['recent_high'] - df['close']) / df['close']
    df['dist_to_low'] = (df['close'] - df['recent_low']) / df['close']
    
    # Price pattern features
    df['is_doji'] = (df[f'body_to_range_{tf}'] < 0.1).astype(int)
    df['is_hammer'] = ((df[f'lower_shadow_{tf}'] > 2 * df[f'candle_body_{tf}']) & 
                      (df[f'upper_shadow_{tf}'] < df[f'candle_body_{tf}'])).astype(int)
    df['is_shooting_star'] = ((df[f'upper_shadow_{tf}'] > 2 * df[f'candle_body_{tf}']) & 
                             (df[f'lower_shadow_{tf}'] < df[f'candle_body_{tf}'])).astype(int)
    
    return df

def generate_merged_data(data_15m_path, data_1h_path, data_4h_path,
                        pred_15m_path, pred_1h_path, pred_4h_path):
    """
    Generate merged data with existing predictions and features for each timeframe.
    
    Parameters:
    - data_*_path: Paths to xauusd data files (15m, 1h, 4h)
    - pred_*_path: Paths to prediction files (15m, 1h, 4h)
    """
    print("Starting merged data generation...")
    
    # Load data
    data_15m = pd.read_csv(data_15m_path)
    data_1h = pd.read_csv(data_1h_path)
    data_4h = pd.read_csv(data_4h_path)
    
    # Load predictions
    pred_15m = pd.read_csv(pred_15m_path)
    pred_1h = pd.read_csv(pred_1h_path)
    pred_4h = pd.read_csv(pred_4h_path)
    
    # Validate data
    required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    for name, df in [('15m', data_15m), ('1h', data_1h), ('4h', data_4h)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} data missing columns: {missing_cols}")
    
    # Validate predictions
    required_pred_cols = ['time', 'predicted']
    for name, df in [('pred_15m', pred_15m), ('pred_1h', pred_1h), ('pred_4h', pred_4h)]:
        missing_cols = [col for col in required_pred_cols if col not in df.columns]
        if missing_cols:
            print(f"Columns in {name}: {df.columns.tolist()}")
            raise ValueError(f"{name} data missing columns: {missing_cols}")
        # Rename 'predicted' to 'prediction' for consistency
        if 'predicted' in df.columns:
            df.rename(columns={'predicted': 'prediction'}, inplace=True)
    
    # Convert time columns
    for df in [data_15m, data_1h, data_4h, pred_15m, pred_1h, pred_4h]:
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S%z')
    
    # Create merged data for each timeframe
    print("Creating merged data for 15m...")
    data_15m_with_indicators = create_technical_indicators(data_15m, '15m')
    merged_15m = create_directional_features(data_15m_with_indicators, pred_15m, '15m')
    
    print("Creating merged data for 1h...")
    data_1h_with_indicators = create_technical_indicators(data_1h, '1h')
    merged_1h = create_directional_features(data_1h_with_indicators, pred_1h, '1h')
    
    print("Creating merged data for 4h...")
    data_4h_with_indicators = create_technical_indicators(data_4h, '4h')
    merged_4h = create_directional_features(data_4h_with_indicators, pred_4h, '4h')
    
    # Save the merged data
    merged_15m.to_csv('merged_data_15m.csv', index=False)
    merged_1h.to_csv('merged_data_1h.csv', index=False)
    merged_4h.to_csv('merged_data_4h.csv', index=False)
    
    print("Merged data files saved: merged_data_15m.csv, merged_data_1h.csv, merged_data_4h.csv")
    
    return merged_15m, merged_1h, merged_4h

if __name__ == "__main__":
    try:
        merged_15m, merged_1h, merged_4h = generate_merged_data(
            data_15m_path='xauusd_15m.csv',
            data_1h_path='xauusd_1h.csv',
            data_4h_path='xauusd_4h.csv',
            pred_15m_path='predictions_15m.csv',
            pred_1h_path='predictions_1h.csv',
            pred_4h_path='predictions_4h.csv'
        )
        
        print("\nMerged Data Generation Complete!")
        print(f"15m data shape: {merged_15m.shape}")
        print(f"1h data shape: {merged_1h.shape}")
        print(f"4h data shape: {merged_4h.shape}")
        
    except Exception as e:
        print(f"Error generating merged data: {e}")
        print("Please ensure all data files are in the correct format and location.")
