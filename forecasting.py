import os
import logging
from typing import Optional

from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from acquisition import StockLoader


from darts.timeseries import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from progress_tracker import progress_tracker
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler, StaticCovariatesTransformer
from darts.models import TiDEModel
from darts.metrics import smape
from optuna.pruners import HyperbandPruner
import optuna
#get darts SMAPELoss
from darts.utils.losses import SmapeLoss, MAELoss
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
import joblib
import warnings

# disable warnings
warnings.filterwarnings("ignore")



def generate_torch_kwargs():
    # run torch models on GPU, and disable progress bars for all model stages except training.
    stopper = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    return {
        "pl_trainer_kwargs": {
            "accelerator": "gpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True), stopper],
        }
    }



class StockAnalyzer:
    """
    A class designed to analyze stock data for a list of tickers over a specified date range or period.

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
        If both start_date and end_date are not provided, then period must be provided.
    data : DataFrame, internal
        A pandas DataFrame containing the fetched stock data.
    """
    def __init__(self, tickers: list[str], start_date: Optional[str] = None, end_date: Optional[str] = None, period: Optional[str] = None):
        self.tickers = tickers
        self.ticker_list = tickers.copy()
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.loader = StockLoader(self.tickers, self.start_date, self.end_date, self.period)
        self.data = self.loader.get_data()
        self.add_encoders = {
            'datetime_attribute': {'past': ['year', 'month', 'week'], 'future': ['year', 'month', 'week']}
        }

    def basic_plot(self, feature: str) -> None:
        self.loader.plot_feature(feature)

    def data_prep_sktime(self) -> None:
        # Create a dictionary to map tickers to unique integers
        ticker_to_int = {ticker: i for i, ticker in enumerate(self.data['ticker'].unique())}
        data = self.data.copy()

        # Convert the index to a PeriodIndex
        data.index = pd.PeriodIndex(data.index, freq='B')
        data["date"] = data.index

        # Convert tickers to integers
        data['ticker_int'] = data['ticker'].map(ticker_to_int)

        # Group by ticker_int and date, setting them as a MultiIndex
        data = data.set_index(['ticker_int', 'date'])

        # Ensure each ticker_int has all dates associated with it
        all_dates = pd.period_range(start=data.index.get_level_values('date').min(),
                                    end=data.index.get_level_values('date').max(),
                                    freq='B')

        # Reindex to ensure each ticker_int has all dates
        data = data.groupby(level=0).apply(lambda x: x.reindex(all_dates, level=1))

        # Remove the extra index level added by groupby-apply
        data.index = data.index.droplevel(0)

        # Store the ticker_to_int mapping for future reference
        self.ticker_to_int = ticker_to_int
        self.int_to_ticker = {v: k for k, v in ticker_to_int.items()}

        # Drop the ticker column
        data.drop('ticker', axis=1, inplace=True)

        y = data[['Close']]
        X = data.drop(columns=['Close'], axis=1)
        return X, y

    def create_darts_timeseries(self):
        # Identify columns to be used as covariates
        all_columns = self.data.columns
        covariate_columns = [col for col in all_columns if col not in ['Close', 'ticker']]
        self.num_covariates = len(covariate_columns)

        # Reset index to get the date as a column
        data_with_time = self.data.copy()
        data_with_time = data_with_time.reset_index()

        # Verify temporal alignment for all tickers
        grouped_data = data_with_time.groupby('ticker')
        start_dates = grouped_data['date'].min()
        end_dates = grouped_data['date'].max()

        # Check if all tickers have the same date range
        if not (start_dates == start_dates.iloc[0]).all() or not (end_dates == end_dates.iloc[0]).all():
            logging.warning("Not all tickers have the same date range:")
            for ticker in start_dates.index:
                logging.warning(f"{ticker}: {start_dates[ticker]} to {end_dates[ticker]}")

            # Find the latest start date and earliest end date
            latest_start = start_dates.max()
            earliest_end = end_dates.min()

            # Trim data to ensure all tickers have the same date range
            data_with_time = data_with_time[
                (data_with_time['date'] >= latest_start) &
                (data_with_time['date'] <= earliest_end)
            ]
            logging.info(f"Data trimmed to common date range: {latest_start} to {earliest_end}")

        # Verify the time index is monotonically increasing for each ticker
        for ticker, group in grouped_data:
            if not group['date'].is_monotonic_increasing:
                logging.error(f"Time index for {ticker} is not monotonically increasing")
                # Sort by time if needed
                group = group.sort_values('date')

        # Create darts timeseries for the target variable (Close) with Windows-compatible settings
        ts_target_list = TimeSeries.from_group_dataframe(
            df=data_with_time,
            group_cols=["ticker"],
            time_col="date",
            value_cols="Close",
            freq="B",
            n_jobs=1,  # Use single process on Windows
            fill_missing_dates=True
        )

        # Create darts timeseries for covariates with Windows-compatible settings
        ts_covariates_list = TimeSeries.from_group_dataframe(
            df=data_with_time,
            group_cols=["ticker"],
            time_col="date",
            value_cols=covariate_columns,
            freq="B",
            n_jobs=1,  # Use single process on Windows
            fill_missing_dates=True
        )

        # Fill missing values for both target and covariates
        filler = MissingValuesFiller()
        ts_target_list = filler.transform(ts_target_list)
        ts_covariates_list = filler.transform(ts_covariates_list)
        ts_covariates_list = [ts.add_holidays("US") for ts in ts_covariates_list]

        return ts_target_list, ts_covariates_list

    def create_darts_timeseries_rl(self):
        # Calculate return
        self.data['return'] = self.data['Close'] / self.data['Close'].shift(1)

        # Identify columns to be used as covariates
        all_columns = self.data.columns
        covariate_columns = [col for col in all_columns if col not in ['Close', 'ticker']]
        self.num_covariates = len(covariate_columns)

        # Reset index to get the date as a column
        data_with_time = self.data.reset_index()

        # Ensure we have the correct date column name
        if data_with_time.index.name == 'date':
            data_with_time = data_with_time.reset_index()

        # Verify temporal alignment for all tickers
        grouped_data = data_with_time.groupby('ticker')
        start_dates = grouped_data['date'].min()
        end_dates = grouped_data['date'].max()

        # Check if all tickers have the same date range
        if not (start_dates == start_dates.iloc[0]).all() or not (end_dates == end_dates.iloc[0]).all():
            logging.warning("Not all tickers have the same date range:")
            for ticker in start_dates.index:
                logging.warning(f"{ticker}: {start_dates[ticker]} to {end_dates[ticker]}")

            # Find the latest start date and earliest end date
            latest_start = start_dates.max()
            earliest_end = end_dates.min()

            # Trim data to ensure all tickers have the same date range
            data_with_time = data_with_time[
                (data_with_time['date'] >= latest_start) &
                (data_with_time['date'] <= earliest_end)
            ]
            logging.info(f"Data trimmed to common date range: {latest_start} to {earliest_end}")

        # Verify the time index is monotonically increasing for each ticker
        for ticker, group in grouped_data:
            if not group['date'].is_monotonic_increasing:
                logging.error(f"Time index for {ticker} is not monotonically increasing")
                # Sort by time if needed
                group = group.sort_values('date')


        # Create darts timeseries for the target variable (Close) with Windows-compatible settings
        ts_target_list = TimeSeries.from_group_dataframe(
            df=data_with_time,
            group_cols=["ticker"],
            time_col="date",
            value_cols="Close",
            freq="B",
            n_jobs=1,  # Use single process on Windows
            fill_missing_dates=True
        )

        # Create darts timeseries for covariates with Windows-compatible settings
        ts_covariates_list = TimeSeries.from_group_dataframe(
            df=data_with_time,
            group_cols=["ticker"],
            time_col="date",
            value_cols=covariate_columns,
            freq="B",
            n_jobs=1,  # Use single process on Windows
            fill_missing_dates=True
        )

        # Create darts timeseries for unscaled Close prices with Windows-compatible settings
        ts_unscaled_close_list = TimeSeries.from_group_dataframe(
            df=data_with_time,
            group_cols=["ticker"],
            time_col="date",
            value_cols="Close",
            freq="B",
            n_jobs=1,  # Use single process on Windows
            fill_missing_dates=True
        )

        # Fill missing values for both target and covariates
        filler = MissingValuesFiller()
        ts_target_list = filler.transform(ts_target_list)
        ts_covariates_list = filler.transform(ts_covariates_list)
        ts_unscaled_close_list = filler.transform(ts_unscaled_close_list)
        ts_covariates_list = [ts.add_holidays("US") for ts in ts_covariates_list]

        return ts_target_list, ts_covariates_list, ts_unscaled_close_list

    def transform_timeseries(self, ts_target_list, ts_covariates_list, test_size=0.3, ts_unscaled_close_list=None):
        # Split the data into train and test sets for each series
        train_ts_list = []
        val_ts_list = []
        train_covariates_list = []
        val_covariates_list = []
        train_unscaled_close_list = []
        val_unscaled_close_list = []

        # Check if ts_unscaled_close_list is provided
        if ts_unscaled_close_list is not None:
            for ts_target, ts_covariates, ts_unscaled_close in zip(ts_target_list, ts_covariates_list, ts_unscaled_close_list):
                train_ts, val_ts = ts_target.split_before(ts_target.time_index[-int(len(ts_target) * test_size)])
                train_covariates_ts, val_covariates_ts = ts_covariates.split_before(ts_covariates.time_index[-int(len(ts_covariates) * test_size)])
                train_unscaled_close, val_unscaled_close = ts_unscaled_close.split_before(ts_unscaled_close.time_index[-int(len(ts_unscaled_close) * test_size)])

                train_ts_list.append(train_ts)
                val_ts_list.append(val_ts)
                train_covariates_list.append(train_covariates_ts)
                val_covariates_list.append(val_covariates_ts)
                train_unscaled_close_list.append(train_unscaled_close)
                val_unscaled_close_list.append(val_unscaled_close)
        else:
            for ts_target, ts_covariates in zip(ts_target_list, ts_covariates_list):
                train_ts, val_ts = ts_target.split_before(ts_target.time_index[-int(len(ts_target) * test_size)])
                train_covariates_ts, val_covariates_ts = ts_covariates.split_before(ts_covariates.time_index[-int(len(ts_covariates) * test_size)])

                train_ts_list.append(train_ts)
                val_ts_list.append(val_ts)
                train_covariates_list.append(train_covariates_ts)
                val_covariates_list.append(val_covariates_ts)

        # Scale the time series
        time_series_scaler = Scaler()
        train_ts_list = time_series_scaler.fit_transform(train_ts_list)
        val_ts_list = time_series_scaler.transform(val_ts_list)

        # Scale the covariates
        covariates_scaler = Scaler()
        train_covariates_list = covariates_scaler.fit_transform(train_covariates_list)
        val_covariates_list = covariates_scaler.transform(val_covariates_list)

        # Categorically encode static covariates
        encoder = StaticCovariatesTransformer()
        train_covariates_list = encoder.fit_transform(train_covariates_list)
        val_covariates_list = encoder.transform(val_covariates_list)

        train_ts_list = encoder.fit_transform(train_ts_list)
        val_ts_list = encoder.transform(val_ts_list)

        return train_ts_list, val_ts_list, train_covariates_list, val_covariates_list, train_unscaled_close_list, val_unscaled_close_list, time_series_scaler, encoder

    def train_stock_model_darts(self, test_size=0.3, save_dir='trained_models', n_trials=20):
        # Initialize progress tracking
        import time
        progress_tracker.update_forecaster_progress(
            status="running",
            message="Preparing data for forecasting model...",
            progress=0.0,
            current_step=0,
            total_steps=n_trials + 1,  # n_trials + final training
            current_trial=0,
            total_trials=n_trials,
            start_time=time.time()
        )

        # Get the time series data and covariates
        ts_target_list, ts_covariates_list = self.create_darts_timeseries()

        progress_tracker.update_forecaster_progress(
            message="Transforming time series data...",
            progress=0.05
        )

        train_ts_list, val_ts_list, train_covariates_list, val_covariates_list, unscaled_close_train_list, unscaled_close_val_list, time_series_scaler, encoder = self.transform_timeseries(ts_target_list, ts_covariates_list, test_size)

        progress_tracker.update_forecaster_progress(
            message="Starting hyperparameter optimization...",
            progress=0.1
        )

        def objective(trial):
            # Update progress for this trial
            current_trial = trial.number + 1
            progress_tracker.update_forecaster_progress(
                current_trial=current_trial,
                current_step=current_trial,
                progress=0.1 + (0.7 * current_trial / n_trials),
                message=f"Hyperparameter optimization trial {current_trial}/{n_trials}..."
            )

            # Define the hyperparameters to tune
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            hidden_size = trial.suggest_int("hidden_size", 16, 128)
            num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 4)
            num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 4)
            decoder_output_dim = trial.suggest_int("decoder_output_dim", 16, 128)
            temporal_width_past = trial.suggest_int("temporal_width_past", 0, self.num_covariates)
            temporal_width_future = trial.suggest_int("temporal_width_future", 0, 4)
            temporal_decoder_hidden = trial.suggest_int("temporal_decoder_hidden", 16, 128)

            # Define the model
            model_tide = TiDEModel(
                input_chunk_length=35,
                output_chunk_length=1,
                dropout=dropout,
                hidden_size=hidden_size,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                decoder_output_dim=decoder_output_dim,
                temporal_width_past=temporal_width_past,
                temporal_width_future=temporal_width_future,
                temporal_decoder_hidden=temporal_decoder_hidden,
                use_layer_norm=True,
                use_static_covariates=True,
                batch_size=64,  # Fixed batch size
                n_epochs=100,
                optimizer_kwargs={"lr": lr},
                add_encoders=self.add_encoders,  # Pass the add_encoders dictionary
                random_state=42,
                force_reset=True,
                use_reversible_instance_norm=True,
                loss_fn=SmapeLoss(),
                **generate_torch_kwargs()
            )

            # Fit the model on multiple series
            model_tide.fit(
                series=train_ts_list,
                past_covariates=train_covariates_list,
                val_series=val_ts_list,
                val_past_covariates=val_covariates_list
            )

            # Perform backtesting
            backtest = model_tide.historical_forecasts(
                series=val_ts_list,
                past_covariates=val_covariates_list,
                start=0.7,
                forecast_horizon=1,
                stride=1,
                retrain=False,
                verbose=False
            )

            # Calculate SMAPE for backtesting results
            smape_values = smape(backtest, val_ts_list)
            smape_value = np.mean(smape_values)

            return smape_value

        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='minimize', pruner=HyperbandPruner())

        # Update progress before optimization
        progress_tracker.update_forecaster_progress(
            message="Running hyperparameter optimization...",
            progress=0.1
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Update progress after optimization
        progress_tracker.update_forecaster_progress(
            message="Hyperparameter optimization completed. Preparing final model...",
            progress=0.8,
            current_step=n_trials,
            current_trial=n_trials
        )

        # Get the best hyperparameters
        best_params = study.best_params
        best_score = study.best_value

        # Update progress with best score
        progress_tracker.update_forecaster_progress(
            best_score=best_score
        )
        model_params = {
            "model_name": "TiDE_optuna_best",
            "save_checkpoints": True,
            "input_chunk_length": 35,
            "output_chunk_length": 1,
            "dropout": best_params["dropout"],
            "hidden_size": best_params["hidden_size"],
            "num_encoder_layers": best_params["num_encoder_layers"],
            "num_decoder_layers": best_params["num_decoder_layers"],
            "decoder_output_dim": best_params["decoder_output_dim"],
            "temporal_width_past": best_params["temporal_width_past"],
            "temporal_width_future": best_params["temporal_width_future"],
            "temporal_decoder_hidden": best_params["temporal_decoder_hidden"],
            "use_layer_norm": True,
            "use_static_covariates": True,
            "use_reversible_instance_norm": True,
            "batch_size": 64,  # Fixed batch size
            "n_epochs": 100,
            "optimizer_kwargs": {"lr": best_params["lr"]},
            "add_encoders": self.add_encoders,  # Pass the add_encoders dictionary
            "random_state": 42,
            "force_reset": True,
            "loss_fn": SmapeLoss(),
            **generate_torch_kwargs()
        }
        # save model params using joblib
        joblib.dump(model_params, 'trained_models/TiDE_optuna_best_params.joblib')

        # Train the model with the best hyperparameters
        model_tide = TiDEModel(**model_params)
        #model_tide.save_checkpoints = True
        print("Fitting the model with the best hyperparameters...\n")

        # Update progress before final training
        progress_tracker.update_forecaster_progress(
            message="Training final model with best hyperparameters...",
            progress=0.85,
            current_step=n_trials + 1,
            total_steps=n_trials + 1
        )

        # Fit the model with the best hyperparameters
        model_tide.fit(
            series=train_ts_list,
            past_covariates=train_covariates_list,
            val_series=val_ts_list,
            val_past_covariates=val_covariates_list
        )

        print("Best Model training complete.\n")
        model_tide.save("trained_models/TiDE_optuna_best_trained")

        # Update progress after final training
        import time
        progress_tracker.update_forecaster_progress(
            status="completed",
            message="Forecasting model trained successfully!",
            progress=1.0,
            end_time=time.time()
        )
        model_tide = TiDEModel(**model_params)

        # Load the model for quick testing
        model_tide = model_tide.load("trained_models/TiDE_optuna_best_trained")

        hist_fcst_params = {
            "series": val_ts_list,
            "past_covariates": val_covariates_list,
            "start": 0.5,
            "forecast_horizon": 1,
            "stride": 1,
            "retrain": False,
            "verbose": False
        }
        hist_fcst = model_tide.historical_forecasts(last_points_only=True, **hist_fcst_params)
        # rescale historical forecast and validation data
        hist_fcst = time_series_scaler.inverse_transform(hist_fcst)
        val_ts_list = time_series_scaler.inverse_transform(val_ts_list)

        # plot both to compare
        for ts, fcst in zip(val_ts_list, hist_fcst):
            ts.plot(label="Actual")
            fcst.plot(label="Backtest Forecast 1 Day Ahead")

    def get_darts_forecast(self, test_size: float = 0.3, fcst_len: int = 1, save_hist_forecasts: bool = False):
        # Get the time series data and covariates
        ts_target_list, ts_covariates_list = self.create_darts_timeseries()
        train_ts_list, val_ts_list, train_covariates_list, val_covariates_list, unscaled_close_train_list, unscaled_close_val_list, time_series_scaler, encoder = self.transform_timeseries(ts_target_list, ts_covariates_list, test_size)
        model_params = joblib.load('trained_models/TiDE_optuna_best_params.joblib')
        model_tide = TiDEModel(**model_params)
        tide_model = model_tide.load("trained_models/TiDE_optuna_best_trained")

        if save_hist_forecasts:
            # Generate historical forecasts
            hist_fcst_params = {
                "series": val_ts_list,
                "past_covariates": val_covariates_list,
                "start": 0.5,
                "forecast_horizon": 1,
                "stride": 1,
                "retrain": False,
                "verbose": False
            }
            preds = tide_model.historical_forecasts(**hist_fcst_params)

            # Prepare actuals (validation data) for saving
            actuals = val_ts_list
        else:
            # Generate regular predictions
            preds = tide_model.predict(n=fcst_len, series=val_ts_list, past_covariates=val_covariates_list)

        # Scale predictions
        preds = time_series_scaler.inverse_transform(preds)
        preds = encoder.inverse_transform(preds)

        # Convert predictions to dataframe
        preds_list = []
        for pred in preds:
            static_covariates = pred.static_covariates
            ticker = static_covariates["ticker"].values.tolist()[0]
            pred_df = pred.pd_dataframe()
            pred_df["ticker"] = ticker
            pred_df.index.name = 'Date'  # Rename index to 'Date'
            preds_list.append(pred_df)

        preds_df = pd.concat(preds_list)

        if save_hist_forecasts:
            # Save historical forecasts
            preds_df.to_csv('./data/forecast.csv', index=True)

            # Prepare and save actuals
            actuals = time_series_scaler.inverse_transform(actuals)
            actuals = encoder.inverse_transform(actuals)
            actuals_list = []
            for actual in actuals:
                static_covariates = actual.static_covariates
                ticker = static_covariates["ticker"].values.tolist()[0]
                actual_df = actual.pd_dataframe()
                actual_df["ticker"] = ticker
                actual_df.index.name = 'Date'  # Rename index to 'Date'
                actuals_list.append(actual_df)

            actuals_df = pd.concat(actuals_list)

            # Trim actuals to align with historical forecasts start date
            min_forecast_date = preds_df.index.min()
            actuals_df = actuals_df[actuals_df.index >= min_forecast_date]


            actuals_df.to_csv('./data/actuals.csv', index=True)

        return preds_df

if __name__ == "__main__":
    tickers = ['FTNT', 'NVDA']
    analyzer = StockAnalyzer(tickers, start_date="2015-01-01")
    analyzer.train_stock_model_darts()
    