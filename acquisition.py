import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get Optional parameter
from typing import Optional
from stock_indicators import Quote, indicators
from IPython.display import display
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import time
import random

class StockLoader:
    """
    A class designed to fetch and process stock data for a list of tickers over a specified date range or period.

    Attributes
    ----------
    tickers : list[str]
        List of stock ticker symbols to fetch data for.
    ticker_list : list[str]
        A copy of the tickers list, used for plotting purposes.
    start_date : str, optional
        The starting date from which to fetch the stock data. Format: YYYY-MM-DD.
    end_date : str, optional
        The ending date until which to fetch the stock data. Format: YYYY-MM-DD.
    period : str, optional
        A string representing the period over which to fetch the stock data. Examples include '1mo', '3d', etc.
        If both `start_date` and `end_date` are not provided, then `period` must be provided.
    data : DataFrame, internal
        A pandas DataFrame containing the fetched stock data.

    Methods
    -------
    format_tickers():
        Formats the tickers list into a space-separated string for API calls.
    load_data():
        Fetches the stock data based on the provided parameters and stores it in the `data` attribute.
    plot_close():
        Plots the closing price for each ticker over the specified date range or period.

    Notes
    -----
    The constructor checks if either `start_date` or `end_date` is provided without the other. If so,
    and no `period` is provided, it raises a ValueError indicating that either a `period` parameter or both
    `start_date` and `end_date` parameters must be provided.
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    
    def __init__(self, tickers: list[str], start_date: Optional[str] = None, end_date: Optional[str] = None, period: Optional[str] = None):
        if start_date and not end_date:
            end_date = self.get_last_business_day()
        
        if not (start_date and end_date) and not period:
            raise ValueError("Please provide period parameter or start_date and end_date parameters")
                
        self.tickers = tickers
        self.ticker_list = tickers.copy()
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.data = None
        self.data_restructured = None
        
    @staticmethod
    def get_last_business_day():
        nyse = mcal.get_calendar('NYSE')
        today = datetime.now().date()
        last_business_day = nyse.valid_days(end_date=today, start_date=today - timedelta(days=10))[-1].date()
        return last_business_day.strftime('%Y-%m-%d')
    
    def convert_quotes_df(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics_dict = {}
    
        for ticker in df.columns.levels[0]:
            ticker_df = df[ticker]
            
            # Create Quote objects in bulk, handling NaN values
            quotes_list = [Quote(date=date, open=row['Open'], high=row['High'], 
                                 low=row['Low'], close=row['Close'], volume=row['Volume']) 
                           for date, row in ticker_df.iterrows()
                           if not (pd.isna(row['Open']) or pd.isna(row['High']) or 
                                   pd.isna(row['Low']) or pd.isna(row['Close']) or pd.isna(row['Volume']))]
    
            # Calculate trading metrics
            metrics_list = self.featurize_data(quotes_list)
    
            # Create DataFrame for the current ticker more efficiently
            ticker_data = pd.DataFrame({
                'Open': ticker_df['Open'],
                'High': ticker_df['High'],
                'Low': ticker_df['Low'],
                'Close': ticker_df['Close'],
                'Volume': ticker_df['Volume']
            }, index=ticker_df.index)
    
            # Add the features to the ticker DataFrame
            sma_df = pd.DataFrame(metrics_list).set_index('Date')
            ticker_data = pd.concat([ticker_data, sma_df], axis=1)
    
            metrics_dict[ticker] = ticker_data
    
        # Combine all ticker DataFrames into a multi-level DataFrame
        merged_df = pd.concat(metrics_dict, axis=1)
    
        return merged_df
    
    
    def featurize_data(self, quotes_list: list[Quote]):
        # Calculate all indicators at once
        results = {
            'sma': indicators.get_sma(quotes_list, lookback_periods=14),
            'obv': indicators.get_obv(quotes_list, sma_periods=14),
            'adl': indicators.get_adl(quotes_list, sma_periods=14),
            'adx': indicators.get_adx(quotes_list, lookback_periods=14),
            'aroon': indicators.get_aroon(quotes_list, lookback_periods=14),
            'macd': indicators.get_macd(quotes_list),
            'rsi': indicators.get_rsi(quotes_list, lookback_periods=14),
            'stoch': indicators.get_stoch(quotes_list, lookback_periods=14)
        }
    
        # Create a DataFrame from the results
        df = pd.DataFrame({
            'Date': [r.date for r in results['sma']],
            'sma_14': [r.sma for r in results['sma']],
            'obv_14': [r.obv for r in results['obv']],
            'adl_14': [r.adl for r in results['adl']],
            'adx_14': [r.adx for r in results['adx']],
            'aroon_14': [r.oscillator for r in results['aroon']],
            'macd': [r.macd for r in results['macd']],
            'rsi_14': [r.rsi for r in results['rsi']],
            'stoch_14': [r.oscillator for r in results['stoch']]
        })
    
        return df.to_dict('records')
        
    def format_tickers(self):
        self.tickers = " ".join(self.tickers)
    
    def load_data(self):
        self.format_tickers()
        
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.period is None:
                    self.data = yf.download(self.tickers, self.start_date, self.end_date, group_by="ticker")
                else:
                    self.data = yf.download(self.tickers, period=self.period, group_by="ticker")
                
                if self.data is None or self.data.empty:
                    raise ValueError("No data downloaded")
                
                break  # Success - exit retry loop
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (1 + random.random())  # Add jitter
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed to download data after {self.MAX_RETRIES} attempts") from e

        # Continue with data processing
        self.trim_to_shortest_history()
        self.data = self.convert_quotes_df(self.data)
        
        if len(self.data) > 183:
            self.data = self.data.iloc[183:]
            print(f"Removed first 183 days. New start date: {self.data.index[0]}")
        else:
            print(f"Warning: Data has less than 183 days. Keeping all available data.")
        
        self.data_restructured = self.data.stack(level=0).reset_index()
        self.data_restructured.columns = ['date', 'ticker'] + list(self.data_restructured.columns[2:])
        self.data_restructured['date'] = pd.to_datetime(self.data_restructured['date'])
        self.data_restructured.set_index('date', inplace=True)
        
        return self.data_restructured
        
    def plot_feature(self, feature: str):
        if self.data is None:
            self.data_restructured = self.load_data()
        for ticker in self.ticker_list:
            plt.figure(figsize=(16,10))
            #print(self.data[ticker])
            #display(self.data_restructured.groupby('ticker').get_group(ticker)[feature])
            plt.plot(self.data_restructured.groupby('ticker').get_group(ticker)[feature])
            plt.title(f"{ticker} {feature}")
            plt.grid()
            # enable mini grid lines
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
            plt.ylabel(f"{feature}")
            plt.xlabel("Date")
            plt.show()
            
    def get_data(self) -> pd.DataFrame:
        if self.data_restructured is None:
            self.load_data()
            
        #display(self.data_restructured.head())
        
        return self.data_restructured
            
    def trim_to_shortest_history(self):
        if self.data is None:
            raise ValueError("Data must be loaded before trimming.")
        """
        print("Before trimming:")
        for ticker in self.ticker_list:
            print(f"{ticker} starts at: {self.data[ticker].first_valid_index()}")
        """
        # Find the latest start date among all tickers (first non-NaN value)
        latest_start = max(self.data[ticker].first_valid_index() for ticker in self.ticker_list)
        #print(f"Latest start date: {latest_start}")

        # Trim all tickers to start from the latest start date
        self.data = self.data.loc[latest_start:]
        """
        print("After trimming:")
        for ticker in self.ticker_list:
            print(f"{ticker} starts at: {self.data[ticker].first_valid_index()}")
        """
        # Remove any remaining rows with NaN values
        self.data = self.data.dropna()

        """
        print("After removing NaNs:")
        for ticker in self.ticker_list:
            print(f"{ticker} starts at: {self.data[ticker].first_valid_index()}")
        """
if __name__ == "__main__":
    loader = StockLoader(tickers=['NVDA', "FTNT"], period='max')
    #loader.load_data(sktime_used=True)
    loader.plot_feature('Close')

