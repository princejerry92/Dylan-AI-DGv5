import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from meta_utils import CorrectFeatureEngineer, EnhancedDataValidator
import logging
import os

# Setup logging
logging.basicConfig(filename='xgboost_training_15m.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def load_data(symbol='XAUUSD', timeframe='4h'):
    """Load and validate historical data"""
    csv_path = f'../DGv4/xauusd_{timeframe}.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    EnhancedDataValidator.validate_data(df, timeframe, 10000)
    return df

def prepare_data(df, timeframe='4h'):# change according to the time frame needed to train
    """Prepare features and target"""
    df = CorrectFeatureEngineer.add_engineered_features(df, timeframe)
    df['next_close'] = df['close'].shift(-1)
    df['return'] = (df['next_close'] - df['close']) / df['close']
    df['session'] = df['session'].cat.codes
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if col not in ['next_close', 'return', 'volatility_regime']]
    return df, feature_cols

def train_model(df, feature_cols, train_size=800, test_size=200):
    """Train XGBoost with walk-forward validation and save all predictions"""
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300]
    }
    
    best_model, best_score = None, float('inf')
    all_predictions = []
    
    for start in range(0, len(df) - train_size - test_size + 1, test_size):
        train_end = start + train_size
        test_end = train_end + test_size
        train_data = df.iloc[start:train_end]
        test_data = df.iloc[train_end:test_end]
        
        X_train = train_data[feature_cols]
        y_train = train_data['return']
        X_test = test_data[feature_cols]
        y_test = test_data['return']
        
        for params in ParameterGrid(param_grid):
            model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)
            model.fit(X_train, y_train)
            
            pred = model.predict(X_test)
            score = mean_squared_error(y_test, pred)
            if score < best_score:
                best_score = score
                best_model = model
        
        test_pred = best_model.predict(X_test)
        pred_df = pd.DataFrame({
            'time': test_data.index,
            'actual': test_data['next_close'],
            'predicted': test_data['close'] * (1 + test_pred)
        })
        all_predictions.append(pred_df)
        
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        mae = mean_absolute_error(y_test, test_pred)
        dir_acc = np.mean(np.sign(test_pred) == np.sign(y_test))
        
        logging.info(f'Step {start//test_size + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, DirAcc={dir_acc:.4f}')
        print(f'Step {start//test_size + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, DirAcc={dir_acc:.4f}')

    
    all_predictions = pd.concat(all_predictions)
    return best_model, all_predictions

def main():
    symbol = 'XAUUSD'
    timeframe = '4h'
    
    df = load_data(symbol, timeframe)
    df, feature_cols = prepare_data(df, timeframe)
    
    model, predictions = train_model(df, feature_cols)
    
    model.save_model(f'xgboost_{timeframe}.json')
    predictions.to_csv(f'predictions_{timeframe}.csv', index=False)
    
    last_100 = predictions.tail(100)
    last_100.to_csv(f'last_100_predictions_{timeframe}.csv', index=False)
    
    logging.info(f"Training complete. Predictions saved to predictions_{timeframe}.csv")
    print(f"Training complete. Predictions saved to predictions_{timeframe}.csv")

if __name__ == '__main__':
    main()
