import argparse
import yaml
from twelvedata import TDClient
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from hurst import compute_Hc
import sys
import os
import hashlib
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Module 1: Stationarity Analysis (ADF Test) ---
def run_adf_test(series):
    """Performs the Augmented Dickey-Fuller test on a series."""
    # Test on the original series
    result_orig = adfuller(series)
    p_value_orig = result_orig[1]

    # Test on the first difference
    result_diff = adfuller(series.diff().dropna())
    p_value_diff = result_diff[1]

    conclusion = "I(0)" if p_value_orig < 0.05 else "I(1)" if p_value_diff < 0.05 else "Non-Stationary"
    return {
        "p_value_orig": p_value_orig,
        "p_value_diff": p_value_diff,
        "conclusion": conclusion
    }

# --- Module 2: Cointegration Analysis ---
def run_engle_granger_test(series_a, series_b):
    """Performs the Engle-Granger cointegration test."""
    # 1. Perform OLS regression to find the hedge ratio
    series_b_with_const = sm.add_constant(series_b)
    model = sm.OLS(series_a, series_b_with_const).fit()
    # Access hedge ratio by name for robustness, not by position
    hedge_ratio = model.params[series_b.name]

    # 2. Test the residuals (the spread) for stationarity
    spread = series_a - hedge_ratio * series_b
    coint_result = adfuller(spread)
    p_value = coint_result[1]
    
    conclusion = "The pair IS cointegrated" if p_value < 0.05 else "The pair is NOT cointegrated"
    return {
        "p_value": p_value,
        "hedge_ratio": hedge_ratio,
        "conclusion": conclusion
    }

def run_johansen_test(data_df):
    """Performs the Johansen cointegration test."""
    result = coint_johansen(data_df, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1
    max_eig_stat = result.lr2
    # Check if trace statistic for r=0 exceeds the 95% critical value
    is_cointegrated = trace_stat[0] > result.cvt[0, 1]
    rank = 1 if is_cointegrated else 0
    conclusion = f"Cointegration detected (rank = {rank})" if is_cointegrated else "No cointegration detected"
    return {
        "trace_statistic": trace_stat[0],
        "critical_value_95": result.cvt[0, 1],
        "conclusion": conclusion,
        "rank": rank
    }

# --- Module 3: Mean-Reversion Analysis (Hurst Exponent) ---
def calculate_hurst_exponent(series):
    """Calculates the Hurst Exponent for a series."""
    # Use kind='random_walk' for residual series, as they are not price series
    H, _, _ = compute_Hc(series, kind='random_walk', simplified=True)
    if H < 0.5:
        interpretation = "MEAN-REVERTING"
    elif H > 0.5:
        interpretation = "TRENDING"
    else:
        interpretation = "RANDOM WALK"
    return {"H": H, "interpretation": interpretation}

# --- Data Handling & Configuration ---
# --- Module 4: Plotting --- 
def plot_coint_analysis(raw_df, log_df, spread, start_period, end_period):
    """Generates and saves a 5-panel Plotly chart of the cointegration analysis."""
    symbol1, symbol2 = raw_df.columns
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                        subplot_titles=(f'Raw Price: {symbol1}', 
                                        f'Raw Price: {symbol2}', 
                                        f'Log Price: {symbol1}', 
                                        f'Log Price: {symbol2}', 
                                        'Cointegrated Spread'))

    # Add traces with connectgaps=False to prevent drawing lines over non-trading periods
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df[symbol1], name=symbol1, line=dict(color='blue'), connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df[symbol2], name=symbol2, line=dict(color='black'), connectgaps=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=log_df.index, y=log_df[symbol1], name=f'{symbol1} (Log)', line=dict(color='red'), connectgaps=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=log_df.index, y=log_df[symbol2], name=f'{symbol2} (Log)', line=dict(color='green'), connectgaps=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread', line=dict(color='gray'), connectgaps=False), row=5, col=1)
    
    fig.add_hline(y=spread.mean(), line_dash="dash", line_color="red", annotation_text="Mean", row=5, col=1)

    fig.update_layout(
        title_text=f'Cointegration Analysis: {symbol1} & {symbol2} ({start_period} to {end_period})',
        height=1200, # Increased height for 5 panels
        showlegend=False
    )
    
    # Save plot to HTML file
    filename = f"{symbol1.replace('/', '_')}-{symbol2.replace('/', '_')}_analysis.html"
    fig.write_html(filename)
    print(f"Plot saved to {filename}")

# --- Data Handling & Configuration ---
def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

def fetch_data(symbols, start_date, end_date, interval, api_key, alignment='asof'):
    """Fetches and aligns historical price data using a robust 'asof' merge strategy with batching."""
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # v3 includes batched fetching logic
    request_string = f"{''.join(symbols)}-{start_date}-{end_date}-{interval}-{alignment}-v3-batched"
    filename_hash = hashlib.md5(request_string.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{filename_hash}.csv")

    if os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if all(s in df.columns for s in symbols):
            return df[symbols]
        else:
            print("Cache is stale or invalid. Refetching...")

    print("Fetching data from Twelve Data API with batching...")
    td = TDClient(apikey=api_key)
    data_frames = {}
    for symbol in symbols:
        try:
            print(f"Fetching all historical data for {symbol}...")
            all_data = []
            current_end_date = end_date
            
            while True:
                ts = td.time_series(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=current_end_date,
                    timezone='America/New_York',
                    outputsize=5000
                ).as_pandas()

                if ts.empty:
                    break  # No more data

                all_data.append(ts)
                
                if len(ts) < 5000:
                    break  # Fetched all available data

                # Set the end_date for the next batch to the earliest timestamp of the current batch
                new_end_date = ts.index.min() - pd.Timedelta(seconds=1)
                if pd.to_datetime(new_end_date) < pd.to_datetime(start_date):
                    break
                current_end_date = new_end_date.strftime('%Y-%m-%d %H:%M:%S')

            if not all_data:
                print(f"Warning: No data returned for {symbol}.")
                return None

            # Combine all chunks, sort, and remove duplicates
            full_ts = pd.concat(all_data)
            full_ts.sort_index(inplace=True)
            full_ts = full_ts[~full_ts.index.duplicated(keep='first')]
            # Ensure index is a DatetimeIndex
            full_ts.index = pd.to_datetime(full_ts.index)

            data_frames[symbol] = full_ts[['close']].rename(columns={'close': symbol})
            print(f"Successfully fetched {len(full_ts)} data points for {symbol}.")

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    if len(data_frames) < len(symbols):
        print("Error: Could not fetch data for all requested symbols.")
        return None

    # Identify which asset is likely the stock (less data) vs crypto (more data)
    # Asof merge requires a 'left' and 'right' DataFrame.
    sorted_symbols = sorted(data_frames.keys(), key=lambda s: len(data_frames[s]))
    left_df = data_frames[sorted_symbols[0]]
    right_df = data_frames[sorted_symbols[1]]

    # Use merge_asof to align timestamps, finding the nearest crypto price for each stock price
    df = pd.merge_asof(
        left=left_df,
        right=right_df,
        left_index=True,
        right_index=True,
        direction='nearest', # Find the closest timestamp
        tolerance=pd.Timedelta('1 minute') # Within a 1-minute window
    )
    df.dropna(inplace=True) # Drop rows where no match was found within the tolerance

    # Resample the DataFrame to the specified interval to introduce NaNs in non-trading periods.
    # This allows `connectgaps=False` in the plotting function to work correctly.
    if not df.empty:
        # Resample using the interval directly (e.g., '30min'). Pandas handles the frequency string.
        df = df.resample(interval).asfreq()

    if not df.empty:
        df.to_csv(cache_file)
        print(f"Data saved to cache: {cache_file}")

    return df

# --- Main Application Logic ---
def main():
    parser = argparse.ArgumentParser(description="Crypto Cointegration Analyzer")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (e.g., config.yaml)')
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.get('settings', {})
    pairs_to_analyze = config.get('pairs_to_analyze', [])

    if not pairs_to_analyze:
        print("No pairs specified in the configuration file.")
        return

    print(f"Processing {len(pairs_to_analyze)} pairs from {args.config}...")

    timeframe = settings['timeframe']
    start_period = settings['start_date']
    api_key = settings['twelvedata_api_key']

    if 'YOUR_API_KEY' in api_key:
        print("Error: Please replace 'YOUR_API_KEY' in config.yaml with your Twelve Data API key.")
        return

    for pair in pairs_to_analyze:
        symbol1, symbol2 = pair[0], pair[1]
        print(f"\n=====================================================")
        print(f"Analysis Report for: {symbol1} & {symbol2}")
        print(f"=====================================================")

        # Fetch data for both symbols at once to ensure alignment
        df = fetch_data([symbol1, symbol2], start_period, datetime.now().strftime('%Y-%m-%d'), timeframe, api_key)
        if df is None or df.empty:
            print("Could not fetch data for one or both symbols. Skipping pair.")
            continue
        
        if df.empty:
            print("No overlapping data available for the pair after cleaning. Skipping.")
            continue

        # Create a cleaned version of the data for statistical analysis
        analysis_df = df.dropna()
        log_analysis_df = np.log(analysis_df)
        end_period = df.index.max().strftime('%Y-%m-%d')
        print(f"Period: {start_period} to {end_period}")

        # --- Run Analyses on Cleaned Data ---
        # 1. Stationarity
        adf1 = run_adf_test(log_analysis_df[symbol1])
        adf2 = run_adf_test(log_analysis_df[symbol2])
        print("\n--- Stationarity Test (ADF) ---")
        print(f"{symbol1}: {adf1['conclusion']} (p-value on diff: {adf1['p_value_diff']:.2f})")
        print(f"{symbol2}: {adf2['conclusion']} (p-value on diff: {adf2['p_value_diff']:.2f})")

        # Proceed only if both are I(1)
        if adf1['conclusion'] != 'I(1)' or adf2['conclusion'] != 'I(1)':
            final_verdict = "Not a candidate. Both series must be I(1)."
        else:
            # 2. Cointegration Tests
            eg_result = run_engle_granger_test(log_analysis_df[symbol1], log_analysis_df[symbol2])
            is_eg_coint = eg_result['p_value'] < 0.05
            print("\n--- Cointegration Test (Engle-Granger) ---")
            print(f"Conclusion: {eg_result['conclusion']} (p-value: {eg_result['p_value']:.2f}) {'✅' if is_eg_coint else '❌'}")
            if is_eg_coint:
                print(f"Hedge Ratio: {eg_result['hedge_ratio']:.2f}")

            jo_result = run_johansen_test(log_analysis_df[[symbol1, symbol2]])
            is_jo_coint = jo_result['rank'] > 0
            print("\n--- Cointegration Test (Johansen) ---")
            print(f"Conclusion: {jo_result['conclusion']} {'✅' if is_jo_coint else '❌'}")
            
            # 3. Mean-Reversion and Plotting (only if cointegrated)
            if is_eg_coint:
                # Calculate spread on the cleaned data for Hurst exponent
                spread_analysis = log_analysis_df[symbol1] - eg_result['hedge_ratio'] * log_analysis_df[symbol2]
                hurst_result = calculate_hurst_exponent(spread_analysis)
                is_mean_reverting = hurst_result['H'] < 0.5
                print("\n--- Mean-Reversion Test (Hurst Exponent on Spread) ---")
                print(f"Hurst Exponent: {hurst_result['H']:.2f}")
                print(f"Conclusion: The spread is {hurst_result['interpretation']} {'🥳' if is_mean_reverting else ''}")
                
                # Determine final verdict
                if is_jo_coint and is_mean_reverting:
                    final_verdict = "Strong candidate for pairs trading."
                else:
                    final_verdict = "Potential candidate, but spread is not strongly mean-reverting or Johansen test failed."
                
                # Generate and save plot using the original resampled data (with NaNs)
                log_df = np.log(df)
                spread_plot = log_df[symbol1] - eg_result['hedge_ratio'] * log_df[symbol2]
                plot_coint_analysis(df[[symbol1, symbol2]], log_df[[symbol1, symbol2]], spread_plot, start_period, end_period)
            else:
                final_verdict = "Not a candidate for pairs trading."

        print("\n-----------------------------------------------------")
        print(f"Final Verdict: {final_verdict}")
        print("-----------------------------------------------------")

if __name__ == "__main__":
    main()
