import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

# Constants
PIP_VALUE = 0.40  # 40 pips for XAUUSD
TIME_WINDOWS = {'15m': 1, '1h': 4, '4h': 16}  # 15m candles per timeframe

def load_and_prepare_data():
    """Load and merge data from all timeframes"""
    print("Loading data...")
    data_15m = pd.read_csv('merged_data_15m.csv')
    data_1h = pd.read_csv('merged_data_1h.csv')
    data_4h = pd.read_csv('merged_data_4h.csv')

    # Convert time columns
    for df in [data_15m, data_1h, data_4h]:
        df['time'] = pd.to_datetime(df['time'], utc=True)

    # Filter to overlapping time range
    start_time = pd.to_datetime('2025-01-08 06:45:00+00:00')
    end_time = pd.to_datetime('2025-04-15 08:00:00+00:00')
    
    data_15m = data_15m[(data_15m['time'] >= start_time) & (data_15m['time'] <= end_time)].copy()
    data_1h = data_1h[(data_1h['time'] >= start_time) & (data_1h['time'] <= end_time)].copy()
    data_4h = data_4h[(data_4h['time'] >= start_time) & (data_4h['time'] <= end_time)].copy()

    # Merge data
    print("Merging data...")
    merged_data = data_15m.copy()
    merged_data = merged_data.merge(
        data_1h.drop(columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']),
        on='time', how='left', suffixes=('', '_1h'))
    merged_data = merged_data.merge(
        data_4h.drop(columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']),
        on='time', how='left', suffixes=('', '_4h'))

    # Forward fill and clean
    merged_data = merged_data.ffill().dropna()
    
    # Encode volatility regimes
    vol_mapping = {'low': 0, 'medium': 1, 'high': 2}
    for col in ['volatility_regime', 'volatility_regime_1h', 'volatility_regime_4h']:
        if col in merged_data.columns:
            merged_data[col] = merged_data[col].map(vol_mapping).fillna(-1)
    
    return merged_data.reset_index(drop=True)

def create_enhanced_features(df):
    """Add advanced features to improve model performance"""
    print("Creating enhanced features...")
    # Price acceleration and volatility-adjusted features
    df['price_accel_15m'] = df['close'].pct_change().diff()
    
    for period in [1, 4, 16]:  # 15m, 1h, 4h windows
        df[f'vol_adj_mom_{period}'] = (
            df['close'].pct_change(period) / 
            (df['atr_14_15m'].rolling(period).mean().add(1e-8)))
    
    # Prediction consensus
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    if pred_cols:
        df['pred_consensus'] = df[pred_cols].mean(axis=1)
    
    
    
    return df

def create_targets(df):
    """Create timeframe-specific targets with proper lookahead"""
    print("Creating targets...")
    # 15m target
    df['target_15m'] = np.sign(df['close'].shift(-1) - df['close'])
    
    # 1h target (using resampled data)
    df_1h = df.set_index('time').resample('1h').last().ffill()
    df_1h['target_1h'] = np.sign(df_1h['close'].shift(-1) - df_1h['close'])
    df = df.merge(df_1h[['target_1h']], left_on='time', right_index=True, how='left')
    
    # 4h target
    df_4h = df.set_index('time').resample('4h').last().ffill()
    df_4h['target_4h'] = np.sign(df_4h['close'].shift(-1) - df_4h['close'])
    df = df.merge(df_4h[['target_4h']], left_on='time', right_index=True, how='left')
    
    return df.dropna().reset_index(drop=True)

def evaluate_trades(df, predictions, timeframes=['15m', '1h', '4h']):
    """Comprehensive trade evaluation with timeframe-aware windows"""
    results = []
    df = df.copy()
    
    for timeframe in timeframes:
        window = TIME_WINDOWS[timeframe]
        pred_col = f'pred_{timeframe}'
        
        if pred_col not in predictions.columns:
            continue
            
        for i in tqdm(range(len(df) - window), desc=f"Evaluating {timeframe}"):
            current_close = df['close'].iloc[i]
            future_data = df.iloc[i+1:i+1+window]
            
            pred_direction = predictions[pred_col].iloc[i]
            if pred_direction == 0:
                continue
                
            # Calculate price extremes in the lookahead window
            max_high = future_data['high'].max()
            min_low = future_data['low'].min()
            
            # Trade metrics
            hit_tp = False
            hit_sl = False
            
            if pred_direction > 0:  # Buy
                hit_tp = max_high >= current_close + PIP_VALUE
                hit_sl = min_low <= current_close - (PIP_VALUE/2)  # 20 pip SL
            else:  # Sell
                hit_tp = min_low <= current_close - PIP_VALUE
                hit_sl = max_high >= current_close + (PIP_VALUE/2)
            
            results.append({
                'time': df['time'].iloc[i],
                'timeframe': timeframe,
                'direction': pred_direction,
                'hit_tp': int(hit_tp),
                'hit_sl': int(hit_sl),
                'mfe': (max_high - current_close) if pred_direction > 0 else (current_close - min_low),
                'mae': (current_close - min_low) if pred_direction > 0 else (max_high - current_close)
            })
    
    return pd.DataFrame(results)

def main():
    # Load and prepare data
    merged_data = load_and_prepare_data()
    merged_data = create_enhanced_features(merged_data)
    merged_data = create_targets(merged_data)
    
    # Define features and targets
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                   'spread', 'real_volume', 'target_15m', 'target_1h', 'target_4h']
    feature_cols = [col for col in merged_data.columns if col not in exclude_cols]
    
    X = merged_data[feature_cols]
    y = pd.DataFrame({
        '15m': (merged_data['target_15m'] > 0).astype(int),
        '1h': (merged_data['target_1h'] > 0).astype(int),
        '4h': (merged_data['target_4h'] > 0).astype(int)
    })
    
    # Time-based train-test split (safer than random split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    
    # Align test data with original DataFrame
    test_data = merged_data.iloc[X_test.index].copy()
    
    print(f"\nTraining set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining model...")
    model = XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating performance...")
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=['15m', '1h', '4h'], index=X_test.index)
    
    # Add predictions to test data
    test_data[['pred_15m', 'pred_1h', 'pred_4h']] = y_pred_df * 2 - 1  # Convert to -1/1
    
    # Directional accuracy
    for timeframe in ['15m', '1h', '4h']:
        acc = accuracy_score(y_test[timeframe], y_pred_df[timeframe])
        print(f"\n{timeframe} Directional Accuracy: {acc:.4f}")
        print(classification_report(y_test[timeframe], y_pred_df[timeframe]))
    
    # Trade evaluation
    print("\nEvaluating trades...")
    trade_results = evaluate_trades(test_data, test_data[['pred_15m', 'pred_1h', 'pred_4h']])
    
    print("\nTrade Performance Summary:")
    for timeframe in ['15m', '1h', '4h']:
        tf_results = trade_results[trade_results['timeframe'] == timeframe]
        if len(tf_results) == 0:
            print(f"\nNo trades for {timeframe}")
            continue
            
        tp_rate = tf_results['hit_tp'].mean()
        sl_rate = tf_results['hit_sl'].mean()
        rr_ratio = tf_results['mfe'].mean() / tf_results['mae'].mean()
        
        print(f"\n{timeframe} Performance:")
        print(f"TP Hit Rate: {tp_rate:.2%}")
        print(f"SL Hit Rate: {sl_rate:.2%}")
        print(f"Avg Risk-Reward: {rr_ratio:.2f}")
        print(f"Win Rate: {(tp_rate / (tp_rate + sl_rate)):.2%}" if (tp_rate + sl_rate) > 0 else "N/A")
    
    # Save model
    joblib.dump(model, 'meta_learner_optimized.pkl')
    print("\nOptimized model saved to 'meta_learner_optimized.pkl'")

if __name__ == "__main__":
    main()
