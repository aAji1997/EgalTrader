import os
from forecasting import StockAnalyzer

from IPython.display import display


from darts.models import TiDEModel
import joblib
import warnings

import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

class RLFramework:
    def __init__(self, tickers, period="5y"):
        self.analyzer = StockAnalyzer(tickers=tickers, period=period)
        self.tickers = tickers
        
    def get_darts_forecast(self, test_size: int = 0.3, fcst_len: int = 1):
        # Get the time series data and covariates
        ts_target_list, ts_covariates_list = self.analyzer.create_darts_timeseries()
        train_ts_list, val_ts_list, train_covariates_list, val_covariates_list, train_unscaled_close_list, val_unscaled_close_list, time_series_scaler, encoder = self.analyzer.transform_timeseries(ts_target_list, ts_covariates_list, test_size=test_size)
        

        
        model_params = joblib.load('trained_models/TiDE_optuna_best_params.joblib')
        model_tide = TiDEModel(**model_params)
        tide_model = model_tide.load("trained_models/TiDE_optuna_best_trained")
        #print(len(full_ts_list[0]))
        
        # Generate historical forecasts
        hist_fcst_params_train = {
            "series": train_ts_list,
            "past_covariates": train_covariates_list,
            "start": 35 / len(train_ts_list[0]),  # Ensure at least 35 days of prior data
            "forecast_horizon": fcst_len,
            "stride": 1,
            "retrain": False,
            "verbose": False
        }
        hist_fcst_params_val = {
            "series": val_ts_list,
            "past_covariates": val_covariates_list,
            "start": 35 / len(val_ts_list[0]),  # Ensure at least 35 days of prior data
            "forecast_horizon": fcst_len,
            "stride": 1,
            "retrain": False,
            "verbose": False
        }
        preds_train = tide_model.historical_forecasts(**hist_fcst_params_train)
        preds_val = tide_model.historical_forecasts(**hist_fcst_params_val)
        
        preds_train = encoder.inverse_transform(preds_train)
        preds_val = encoder.inverse_transform(preds_val)
        

        
        # Convert predictions to dataframe
        preds_train_list = []
        for pred in preds_train:
            static_covariates = pred.static_covariates
            #print(static_covariates["ticker"])
            ticker = static_covariates["ticker"].values.tolist()[0]
            pred_df = pred.pd_dataframe()
            pred_df["ticker"] = ticker
            #print(pred_df)
            preds_train_list.append(pred_df)
        
        preds_val_list = []
        for pred in preds_val:
            static_covariates = pred.static_covariates
            ticker = static_covariates["ticker"].values.tolist()[0]
            pred_df = pred.pd_dataframe()
            pred_df["ticker"] = ticker
            preds_val_list.append(pred_df)
        
        preds_train_df = pd.concat(preds_train_list)
        preds_val_df = pd.concat(preds_val_list)
        preds_train_df["Date"] = preds_train_df.index
        preds_val_df["Date"] = preds_val_df.index
        
        preds_train_df = pd.get_dummies(preds_train_df, columns=["ticker"])
        preds_val_df = pd.get_dummies(preds_val_df, columns=["ticker"])
        
        return preds_train_df, preds_val_df
    
    def get_hist_data(self):
        ts_target_list, ts_covariates_list, ts_unscaled_close_list = self.analyzer.create_darts_timeseries_rl()
        train_ts_list, val_ts_list, train_covariates_list, val_covariates_list, unscaled_close_train_list, unscaled_close_val_list, time_series_scaler, encoder = self.analyzer.transform_timeseries(ts_target_list, ts_covariates_list, ts_unscaled_close_list=ts_unscaled_close_list)
        
        train_ts_list = encoder.inverse_transform(train_ts_list)
        val_ts_list = encoder.inverse_transform(val_ts_list)
        
        train_covariates_list = encoder.inverse_transform(train_covariates_list)
        val_covariates_list = encoder.inverse_transform(val_covariates_list)
        
        ts_frame_list_train = []
        ts_frame_list_val = []
        for ts in train_ts_list:
            static_covariates = ts.static_covariates
            ticker = static_covariates["ticker"].values.tolist()[0]
            ts_df = ts.pd_dataframe()
            ts_df["ticker"] = ticker
            ts_frame_list_train.append(ts_df)
        for ts in val_ts_list:
            static_covariates = ts.static_covariates
            ticker = static_covariates["ticker"].values.tolist()[0]
            ts_df = ts.pd_dataframe()
            ts_df["ticker"] = ticker

            ts_frame_list_val.append(ts_df)
        
        ts_df_train = pd.concat(ts_frame_list_train)
        ts_df_train["Date"] = ts_df_train.index
        ts_df_val = pd.concat(ts_frame_list_val)
        ts_df_val["Date"] = ts_df_val.index

        covariates_frame_list_train = []
        covariates_frame_list_val = []
        for cov in train_covariates_list:
            static_covariates = cov.static_covariates
            ticker = static_covariates["ticker"].values.tolist()[0]

            cov_df = cov.pd_dataframe()
            cov_df["ticker"] = ticker
            covariates_frame_list_train.append(cov_df)
        for cov in val_covariates_list:
            static_covariates = cov.static_covariates
            ticker = static_covariates["ticker"].values.tolist()[0]
            cov_df = cov.pd_dataframe()
            cov_df["ticker"] = ticker
            covariates_frame_list_val.append(cov_df)
        cov_df_train = pd.concat(covariates_frame_list_train)
        cov_df_train["Date"] = cov_df_train.index
        cov_df_val = pd.concat(covariates_frame_list_val)
        cov_df_val["Date"] = cov_df_val.index
        
        # One-hot encode the 'ticker' column with actual ticker names
        ts_df_train = pd.get_dummies(ts_df_train, columns=["ticker"], prefix="ticker")
        ts_df_val = pd.get_dummies(ts_df_val, columns=["ticker"], prefix="ticker")
        cov_df_train = pd.get_dummies(cov_df_train, columns=["ticker"], prefix="ticker")
        cov_df_val = pd.get_dummies(cov_df_val, columns=["ticker"], prefix="ticker")
        
        merged_train_df = pd.merge(ts_df_train, cov_df_train, on=["Date"] + [col for col in ts_df_train.columns if col.startswith('ticker_')])
        merged_val_df = pd.merge(ts_df_val, cov_df_val, on=["Date"] + [col for col in ts_df_val.columns if col.startswith('ticker_')])

        # Transform unscaled closing prices into dataframes
        unscaled_close_train_df_list = []
        unscaled_close_val_df_list = []
        for unscaled_close, ticker in zip(unscaled_close_train_list, self.tickers):
            unscaled_close_df = unscaled_close.pd_dataframe()
            unscaled_close_df["ticker"] = ticker  # Add ticker information
            unscaled_close_train_df_list.append(unscaled_close_df)
        for unscaled_close, ticker in zip(unscaled_close_val_list, self.tickers):
            unscaled_close_df = unscaled_close.pd_dataframe()
            unscaled_close_df["ticker"] = ticker  # Add ticker information
            unscaled_close_val_df_list.append(unscaled_close_df)

        unscaled_close_train_df = pd.concat(unscaled_close_train_df_list)
        unscaled_close_train_df["Date"] = unscaled_close_train_df.index
        unscaled_close_val_df = pd.concat(unscaled_close_val_df_list)
        unscaled_close_val_df["Date"] = unscaled_close_val_df.index

        return merged_train_df, merged_val_df, unscaled_close_train_df, unscaled_close_val_df

    def fuse_hist_and_forecast(self):
        # Get historical and forecast data
        fcst_train_df, fcst_val_df = self.get_darts_forecast()
        hist_train_df, hist_val_df, hist_unscaled_close_train_df, hist_unscaled_close_val_df = self.get_hist_data()

        # Shift the forecasted closing price to align with the current day's metrics
        fcst_train_df['Forecast_Close'] = fcst_train_df['Close'].shift(-1)
        fcst_val_df['Forecast_Close'] = fcst_val_df['Close'].shift(-1)

        # Drop the last row as it will have NaN in 'Forecast_Close' after shifting
        fcst_train_df = fcst_train_df.dropna(subset=['Forecast_Close'])
        fcst_val_df = fcst_val_df.dropna(subset=['Forecast_Close'])

        # Emphasize forecasts with importance multiplier
        forecast_importance = 2.0
        fcst_train_df['Forecast_Close'] *= forecast_importance
        fcst_val_df['Forecast_Close'] *= forecast_importance

        # Add forecast-based features
        for df in [fcst_train_df, fcst_val_df]:
            # Calculate forecast return (predicted price change)
            df['Forecast_Return'] = (df['Forecast_Close'] - df['Close']) / df['Close']
            # Calculate forecast momentum (direction and magnitude)
            df['Forecast_Momentum'] = df['Forecast_Return'].rolling(window=5).mean().fillna(0)
            # Add forecast confidence (using rolling volatility as a proxy for uncertainty)
            df['Forecast_Confidence'] = 1 / (df['Forecast_Return'].rolling(window=10).std().fillna(0.01))
            # Normalize confidence to [0, 1]
            df['Forecast_Confidence'] = (df['Forecast_Confidence'] - df['Forecast_Confidence'].min()) / (df['Forecast_Confidence'].max() - df['Forecast_Confidence'].min() + 1e-6)

        # Merge the historical and forecast dataframes
        ticker_columns_train = [col for col in hist_train_df.columns if col.startswith('ticker_')]
        ticker_columns_val = [col for col in hist_val_df.columns if col.startswith('ticker_')]
        
        forecast_columns = ['Forecast_Close', 'Forecast_Return', 'Forecast_Momentum', 'Forecast_Confidence']
        
        merged_train_df = pd.merge(
            hist_train_df,
            fcst_train_df[['Date'] + forecast_columns + ticker_columns_train],
            on=['Date'] + ticker_columns_train
        )
        merged_val_df = pd.merge(
            hist_val_df,
            fcst_val_df[['Date'] + forecast_columns + ticker_columns_val],
            on=['Date'] + ticker_columns_val
        )

        return merged_train_df, merged_val_df, hist_unscaled_close_train_df, hist_unscaled_close_val_df
    
    def create_temporal_features(self):
        train_df, val_df, hist_unscaled_close_train_df, hist_unscaled_close_val_df = self.fuse_hist_and_forecast()

        # Extract basic temporal attributes
        for df in [train_df, val_df]:
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['weekday'] = df['Date'].dt.weekday  # Monday=0, Sunday=6

        # Define cyclic features and their maximum values
        cyclic_features = {
            'month': 12,
            'day': 31,        
            'weekday': 7
        }

        # Generate Sinusoidal and Fourier features
        for feature, max_val in cyclic_features.items():
            if feature == 'weekday':
                order = 1
            else:
                order = 2
            for i in range(1, order + 1):
                sin_col = f'{feature}_sin_{i}'
                cos_col = f'{feature}_cos_{i}'
                # Calculate the angular frequency
                angular_freq = 2 * np.pi * i / max_val
                for df in [train_df, val_df]:
                    df[sin_col] = np.sin(angular_freq * df[feature])
                    df[cos_col] = np.cos(angular_freq * df[feature])

        # Optionally, drop the original cyclic features if not needed
        train_df.drop(columns=list(cyclic_features.keys()), inplace=True)
        val_df.drop(columns=list(cyclic_features.keys()), inplace=True)
        
        # make the date column the index
        train_df = train_df.set_index('Date')
        val_df = val_df.set_index('Date')
        hist_unscaled_close_train_df = hist_unscaled_close_train_df.set_index('Date')
        hist_unscaled_close_val_df = hist_unscaled_close_val_df.set_index('Date')


        return train_df, val_df, hist_unscaled_close_train_df, hist_unscaled_close_val_df
    
    def scale_data(self, train_df, val_df):
        scaler = MinMaxScaler()
        
        
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        
        train_X_scaled = scaler.fit_transform(train_df_scaled)
        val_X_scaled = scaler.transform(val_df_scaled)
        
        train_X = pd.DataFrame(train_X_scaled, columns=train_df_scaled.columns, index=train_df.index)
        val_X = pd.DataFrame(val_X_scaled, columns=val_df_scaled.columns, index=val_df.index)
        

        
        return train_X, val_X
    
    def final_preprocessing(self):
        train_df, val_df, hist_unscaled_close_train_df, hist_unscaled_close_val_df = self.create_temporal_features()
        train_X, val_X = self.scale_data(train_df, val_df)
        
        # Restructure scaled observation data
        train_X_restructured = []
        val_X_restructured = []
        
        for df, restructured_list in [(train_X, train_X_restructured), (val_X, val_X_restructured)]:
            # Get unique dates
            dates = df.index.unique()
            
            for date in dates:
                date_data = df.loc[date]
                
                # Create a dict to store data for each ticker at this timestamp
                ticker_data = {}
                
                # Identify which rows belong to which tickers
                for ticker in self.tickers:
                    ticker_mask = date_data[f'ticker_{ticker}'] == 1
                    if isinstance(date_data, pd.Series):
                        ticker_row = date_data
                    else:
                        ticker_row = date_data[ticker_mask].iloc[0]
                    
                    # Store the row data without the ticker columns
                    ticker_cols = [col for col in ticker_row.index if not col.startswith('ticker_')]
                    ticker_data[ticker] = ticker_row[ticker_cols]
                
                # Create a DataFrame for this timestamp with MultiIndex
                timestamp_df = pd.DataFrame(ticker_data).T
                timestamp_df.index.name = 'ticker'
                timestamp_df = timestamp_df.assign(timestamp=date)
                restructured_list.append(timestamp_df)
        
        # Combine all timestamps into single MultiIndex DataFrames
        train_X_final = pd.concat(train_X_restructured, keys=[df.timestamp.iloc[0] for df in train_X_restructured], 
                                 names=['Date', 'ticker'])
        val_X_final = pd.concat(val_X_restructured, keys=[df.timestamp.iloc[0] for df in val_X_restructured], 
                               names=['Date', 'ticker'])
        
        # Restructure unscaled close price data similarly
        hist_unscaled_close_train_df = hist_unscaled_close_train_df.set_index(['ticker'], append=True)
        hist_unscaled_close_val_df = hist_unscaled_close_val_df.set_index(['ticker'], append=True)
        
        # Remove the timestamp column as it's now part of the index
        train_X_final = train_X_final.drop('timestamp', axis=1)
        val_X_final = val_X_final.drop('timestamp', axis=1)
        
        return train_X_final, val_X_final, hist_unscaled_close_train_df, hist_unscaled_close_val_df
    
    
    
if __name__ == "__main__":
    agent = RLFramework(tickers=['NVDA', 'FTNT'])
    train_X, val_X, hist_unscaled_close_train_df, hist_unscaled_close_val_df = agent.final_preprocessing()
    display(train_X.head())
    display(train_X.tail())
    display(hist_unscaled_close_train_df.head())
    
        
        
        