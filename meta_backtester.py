import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# Constants
PIP_VALUE = 0.50  # 50 pips TP
SL_PIPS = 0.35    # 35 pips SL
TIME_WINDOW = 4   # 4 candles (1 hour) for 15m data

# Load data
data_15m = pd.read_csv('merged_data_15m.csv')
data_1h = pd.read_csv('merged_data_1h.csv')
data_4h = pd.read_csv('merged_data_4h.csv')

# Convert time columns
for df in [data_15m, data_1h, data_4h]:
    df['time'] = pd.to_datetime(df['time'], utc=True)

# Filter to backtest range (April 15, 2025, to May 29, 2025)
start_time = pd.to_datetime('2025-04-15 00:00:00+00:00')
end_time = pd.to_datetime('2025-05-29 23:45:00+00:00')
data_15m = data_15m[(data_15m['time'] >= start_time) & (data_15m['time'] <= end_time)].copy()
data_1h = data_1h[(data_1h['time'] >= start_time) & (data_1h['time'] <= end_time)].copy()
data_4h = data_4h[(data_4h['time'] >= start_time) & (data_4h['time'] <= end_time)].copy()

# Merge 1h and 4h features into 15m data
merged_data = data_15m.copy()
merged_data = merged_data.merge(
    data_1h.drop(columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']),
    on='time', how='left', suffixes=('', '_1h')
)
merged_data = merged_data.merge(
    data_4h.drop(columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']),
    on='time', how='left', suffixes=('', '_4h')
)

# Forward fill 1h and 4h features
merged_data = merged_data.ffill()

# Encode volatility_regime columns to numeric
vol_mapping = {'low': 0, 'medium': 1, 'high': 2}
for col in ['volatility_regime', 'volatility_regime_1h', 'volatility_regime_4h']:
    if col in merged_data.columns:
        merged_data[col] = merged_data[col].map(vol_mapping)

# Add enhanced features
merged_data['price_accel_15m'] = merged_data['close'].pct_change().diff()
for period in [1, 4, 16]:
    merged_data[f'vol_adj_mom_{period}'] = (
        merged_data['close'].pct_change(period) /
        (merged_data['atr_14_15m'].rolling(period).mean() + 1e-8)
    )
pred_cols = [c for c in merged_data.columns if c.startswith('pred_')]
if pred_cols:
    merged_data['pred_consensus'] = merged_data[pred_cols].mean(axis=1)

# Drop any rows with NaN
merged_data = merged_data.dropna().reset_index(drop=True)

# Define feature columns (same as in training)
feature_cols = [col for col in merged_data.columns if col not in [
    'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'target_15m', 'target_1h', 'target_4h'
]]

# Load the meta-learner model
model = joblib.load('meta_learner_optv3.pkl')
print("Meta-learner model loaded from 'meta_learner_optv3.pkl'")

# Generate predictions using the meta-learner
X = merged_data[feature_cols]
y_pred = model.predict(X)
y_pred_df = pd.DataFrame(y_pred, columns=['15m', '1h', '4h'], index=X.index)

# Add predictions to the dataset (convert to -1/1 format)
merged_data[['pred_15m', 'pred_1h', 'pred_4h']] = (y_pred_df * 2 - 1)

# Function to evaluate a single trade
def evaluate_trade(entry_price, direction, future_data):
    max_high = future_data['high'].max()
    min_low = future_data['low'].min()
    
    if direction > 0:  # Buy
        if max_high >= entry_price + PIP_VALUE:
            return PIP_VALUE, True  # TP hit
        elif min_low <= entry_price - SL_PIPS:
            return -SL_PIPS, False  # SL hit
        else:
            exit_price = future_data['close'].iloc[-1]
            return (exit_price - entry_price), (exit_price > entry_price)
    else:  # Sell
        if min_low <= entry_price - PIP_VALUE:
            return PIP_VALUE, True  # TP hit
        elif max_high >= entry_price + SL_PIPS:
            return -SL_PIPS, False  # SL hit
        else:
            exit_price = future_data['close'].iloc[-1]
            return (entry_price - exit_price), (exit_price < entry_price)

# Backtesting loop
results = []
in_trade = False
entry_time = None
entry_price = None
direction = None

for i in tqdm(range(len(merged_data) - TIME_WINDOW), desc="Backtesting"):
    current_time = merged_data['time'].iloc[i]
    current_close = merged_data['close'].iloc[i]
    pred_1h = merged_data['pred_1h'].iloc[i]
    
    if not in_trade and pred_1h != 0:  # Enter trade if not in trade and signal exists
        in_trade = True
        entry_time = current_time
        entry_price = current_close
        direction = 1 if pred_1h > 0 else -1
    elif in_trade:
        # Evaluate trade over the next TIME_WINDOW candles
        future_data = merged_data.iloc[i+1:i+1+TIME_WINDOW]
        profit, is_win = evaluate_trade(entry_price, direction, future_data)
        
        if (profit in [PIP_VALUE, -SL_PIPS] or i + TIME_WINDOW >= len(merged_data) - 1):
            results.append({
                'entry_time': entry_time,
                'exit_time': future_data['time'].iloc[-1],
                'direction': 'Buy' if direction > 0 else 'Sell',
                'entry_price': entry_price,
                'exit_price': future_data['close'].iloc[-1],
                'profit': profit,
                'is_win': is_win,
                'tp_hit': profit == PIP_VALUE,
                'sl_hit': profit == -SL_PIPS
            })
            in_trade = False
            entry_time = None
            entry_price = None
            direction = None

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate performance metrics
total_trades = len(results_df)
total_profit = results_df['profit'].sum()
win_trades = results_df[results_df['is_win']].shape[0]
win_rate = win_trades / total_trades if total_trades > 0 else 0
equity = results_df['profit'].cumsum()
max_drawdown = (equity - equity.cummax()).min() if total_trades > 0 else 0

# Print results
print(f"\nBacktest Results (April 15, 2025 - May 29, 2025):")
print(f"Total Trades: {total_trades}")
print(f"Total Profit: {total_profit:.2f}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Max Drawdown: {max_drawdown:.2f}")

# Detailed trade log
#print("\nTrade Log:")
#for _, trade in results_df.iterrows():
    #print(f"{trade['entry_time']} - {trade['direction']} @ {trade['entry_price']:.4f} -> "
          #f"{trade['exit_time']} @ {trade['exit_price']:.4f} = {trade['profit']:.2f} "
          #f"(Win: {trade['is_win']}, TP: {trade['tp_hit']}, SL: {trade['sl_hit']})")

# Save results
results_df.to_csv('backtest_results.csv', index=False)
print("Backtest results saved to 'backtest_results.csv'")
