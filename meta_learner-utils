import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
from scipy.stats import pearsonr
from ta.momentum import RSIIndicator

class EnhancedDataValidator:
    """Validates MT5 data quality"""
    @staticmethod
    def validate_data(df, timeframe, expected_bars):
        if len(df) < expected_bars * 0.9:
            raise ValueError(f"Insufficient data: {len(df)}/{expected_bars} bars")
        
        max_allowed_gap = {
            '15m': timedelta(weeks=2),
            '1h': timedelta(weeks=2),
            '4h': timedelta(weeks=2)
        }[timeframe]
        
        if datetime.now(pytz.utc) - df.index[-1] > max_allowed_gap:
            raise ValueError("Stale data detected")
        
        if df['tick_volume'].iloc[-1] == 0:
            raise ValueError("Zero volume in latest bar")
        
        price_change = df['close'].pct_change().abs()
        if price_change.max() > 0.05:
            print("Warning: Extreme price movement detected")
        
        return True

class CorrectFeatureEngineer:
    """Enhanced feature engineering pipeline"""
    @staticmethod
    def calculate_rsi(series, period=14):
        return RSIIndicator(series, window=period).rsi()

    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def add_engineered_features(df, timeframe):
        df['atr'] = (df['high'].rolling(14).max() - df['low'].rolling(14).min()) / 14
        df['volatility_regime'] = pd.cut(df['atr'], 
                                       bins=[0, df['atr'].quantile(0.3), 
                                             df['atr'].quantile(0.7), np.inf],
                                       labels=[0, 1, 2])
        
        df['rsi_14'] = CorrectFeatureEngineer.calculate_rsi(df['close'], 14)
        macd, signal, hist = CorrectFeatureEngineer.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        df['hour'] = df.index.hour
        df['session'] = pd.cut(df['hour'],
                             bins=[0, 8, 16, 24],
                             labels=['asian', 'european', 'us'],
                             include_lowest=True).astype('category')
        
        df['candle_body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        if timeframe == '15m':
            df['intraday_momentum'] = df['close'].pct_change(4)
            df['volume_momentum'] = df['tick_volume'].pct_change(4)
            df['lagged_return_1'] = df['close'].pct_change().shift(1)
            df['lagged_return_2'] = df['close'].pct_change().shift(2)
            df['lagged_return_3'] = df['close'].pct_change().shift(3)
        elif timeframe == '1h':
            df['intraday_momentum'] = df['close'].pct_change(8)
        elif timeframe == '4h':
            df['intraday_momentum'] = df['close'].pct_change(6)
            
        return df

def compute_rsi_divergence(df, window=4, rsi_period=14):
    """Compute RSI divergence signal"""
    df['rsi_14'] = CorrectFeatureEngineer.calculate_rsi(df['close'], rsi_period)
    signals = pd.Series(0, index=df.index)
    
    for i in range(window, len(df)):
        price_slice = df['close'].iloc[i-window:i]
        rsi_slice = df['rsi_14'].iloc[i-window:i]
        
        price_high_idx = price_slice.idxmax()
        price_low_idx = price_slice.idxmin()
        rsi_high_idx = rsi_slice.idxmax()
        rsi_low_idx = rsi_slice.idxmin()
        
        if price_high_idx == df.index[i-1] and rsi_high_idx != df.index[i-1]:
            if df['rsi_14'].iloc[i-1] > 70:
                signals.iloc[i] = -1  # Bearish divergence
        elif price_low_idx == df.index[i-1] and rsi_low_idx != df.index[i-1]:
            if df['rsi_14'].iloc[i-1] < 30:
                signals.iloc[i] = 1   # Bullish divergence
    
    return signals

class AdvancedTradeLogic:
    """Enhanced trade decision making"""
    @staticmethod
    def determine_trade_direction(predictions, dataframes, meta_tp):
        current_15m = dataframes['15m'].iloc[-1]
        current_1h = dataframes['1h'].iloc[-1]
        current_4h = dataframes['4h'].iloc[-1]
        
        regime = current_15m['volatility_regime']
        weights = {
            '15m': 0.3 if regime == 0 else 0.4 if regime == 1 else 0.5,
            '1h': 0.4 if regime == 0 else 0.3 if regime == 1 else 0.2,
            '4h': 0.3 if regime == 0 else 0.3 if regime == 1 else 0.3
        }
        
        current_price = current_15m['close']
        direction = 'BUY' if meta_tp > current_price else 'SELL' if meta_tp < current_price else 'HOLD'
        
        rsi_div_signal = current_15m['rsi_divergence_signal']
        rsi_ok = (direction == 'BUY' and rsi_div_signal >= 0) or \
                 (direction == 'SELL' and rsi_div_signal <= 0)
        
        trend_ok = (direction == 'BUY' and predictions['4h'] > current_price) or \
                   (direction == 'SELL' and predictions['4h'] < current_price)
        
        atr = current_15m['atr']
        position_size = 0.01 * (100 / atr) if atr > 0 else 0.01
        
        min_move = current_price * 0.001  # 0.1%
        if abs(meta_tp - current_price) < min_move:
            direction = 'HOLD'
        
        return {
            'direction': direction if all([rsi_ok, trend_ok]) else 'HOLD',
            'confidence': min(0.99, abs(meta_tp - current_price) / atr) if atr > 0 else 0.5,
            'position_size': position_size,
            'stop_loss': current_price - (2 * atr if direction == 'BUY' else -2 * atr),
            'take_profit': meta_tp,
            'current_price': current_price,
            'timeframe_weights': weights
        }

def fetch_mt5_data(symbol, timeframe, num_bars=200):
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
    
    timeframe_map = {
        '15m': mt5.TIMEFRAME_M15,
        '1h': mt5.TIMEFRAME_H1,
        '4h': mt5.TIMEFRAME_H4
    }
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, num_bars)
    mt5.shutdown()
    
    if rates is None:
        raise ValueError(f"No data received for {symbol} {timeframe}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    EnhancedDataValidator.validate_data(df, timeframe, num_bars)
    
    return df
