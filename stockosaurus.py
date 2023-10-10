import pandas as pd
import numpy as np

# Load Stock Data from CSV
# Ensure your CSV file has a column 'DATES' and other columns as stock series
file_path = 'stock_data.csv'
stock_data = pd.read_csv(file_path, parse_dates=['DATES'])
stock_data.set_index('DATES', inplace=True)

# Trading Signals Function
def generate_trading_signals_with_labels_and_stock_name_v3(data, stock_name,
                                                           sma_short_window=40, sma_long_window=100,
                                                           macd_short_window=12, macd_long_window=26, macd_signal_window=9,
                                                           bollinger_window=20, bollinger_num_std_dev=2):
    signals = pd.DataFrame(index=data.index)
    
    # Add stock name or code
    signals['stock_name'] = stock_name
    
    # SMA Signals
    signals['sma_short_mavg'] = data['Close'].rolling(window=sma_short_window, min_periods=1, center=False).mean()
    signals['sma_long_mavg'] = data['Close'].rolling(window=sma_long_window, min_periods=1, center=False).mean()
    signals['sma_signal'] = np.where(signals['sma_short_mavg'] > signals['sma_long_mavg'], 1.0, 0.0)
    signals['sma_positions'] = signals['sma_signal'].diff()
    signals['sma_action'] = signals['sma_positions'].apply(lambda x: 'Buy' if x == 1 else ('Sell' if x == -1 else 'Hold'))
    
    # MACD Signals
    exp1 = data['Close'].ewm(span=macd_short_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=macd_long_window, adjust=False).mean()
    signals['macd'] = exp1 - exp2
    signals['macd_signal_line'] = signals['macd'].ewm(span=macd_signal_window, adjust=False).mean()
    signals['macd_signal'] = np.where(signals['macd'] > signals['macd_signal_line'], 1.0, 0.0)
    signals['macd_positions'] = signals['macd_signal'].diff()
    signals['macd_action'] = signals['macd_positions'].apply(lambda x: 'Buy' if x == 1 else ('Sell' if x == -1 else 'Hold'))
    
    # Bollinger Bands Signals
    signals['bollinger_mavg'] = data['Close'].rolling(window=bollinger_window).mean()
    signals['bollinger_std'] = data['Close'].rolling(window=bollinger_window).std()
    signals['bollinger_upper_band'] = signals['bollinger_mavg'] + (signals['bollinger_std'] * bollinger_num_std_dev)
    signals['bollinger_lower_band'] = signals['bollinger_mavg'] - (signals['bollinger_std'] * bollinger_num_std_dev)
    signals['bollinger_signal'] = np.where(data['Close'] > signals['bollinger_upper_band'], -1.0, 0.0)
    signals['bollinger_signal'] = np.where(data['Close'] < signals['bollinger_lower_band'], 1.0, signals['bollinger_signal']) 
    signals['bollinger_positions'] = signals['bollinger_signal'].diff()
    signals['bollinger_action'] = signals['bollinger_positions'].apply(lambda x: 'Buy' if x == 1 else ('Sell' if x == -1 else 'Hold'))

    # Add another signal column here
    # signals['signal'] = ...
    
    return signals

# Initialize a DataFrame to store all signals
all_signals = pd.DataFrame()

# Generate Trading Signals for each stock series in the DataFrame
for stock_name in stock_data.columns:
    stock_series = pd.DataFrame(stock_data[stock_name].rename('Close'))
    signals = generate_trading_signals_with_labels_and_stock_name_v3(stock_series, stock_name)
    all_signals = pd.concat([all_signals, signals], axis=0)

# Save to CSV
csv_file_path = 'result/trading_signals.csv'
all_signals.to_csv(csv_file_path)

