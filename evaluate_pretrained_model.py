import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import torch.nn.functional as F
import shutil
import json

from RL_training import RLTrainer
from RL_environment import PortfolioEnv
from portfolio_agent import PortfolioAgent
from forecasting import StockAnalyzer
from progress_tracker import progress_tracker


class ModelEvaluator:
    """
    Class for evaluating a pretrained RL model over a specific time period.
    """
    def __init__(
        self,
        model_path,
        tickers=None,
        start_date=None,
        end_date=None,
        initial_capital=10000,
        eval_episodes=10
    ):
        """
        Initialize the model evaluator.

        Args:
            model_path: Path to the pretrained model checkpoint
            tickers: List of ticker symbols to evaluate on
            start_date: Start date for evaluation period (format: 'YYYY-MM-DD')
            end_date: End date for evaluation period (format: 'YYYY-MM-DD')
            initial_capital: Initial investment capital
            eval_episodes: Number of evaluation episodes to run
        """
        self.model_path = model_path

        # If no tickers provided, use default ones
        self.tickers = tickers if tickers is not None else ['NVDA', 'FTNT']

        # Set evaluation period
        if start_date is None:
            # Default to 2 months ago from today
            end = datetime.now()
            start = end - timedelta(days=60)
            self.start_date = start.strftime('%Y-%m-%d')
            self.end_date = end.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            self.end_date = end_date

        self.initial_capital = initial_capital
        self.eval_episodes = eval_episodes

        # Initialize progress tracking
        progress_tracker.update_overall_progress(
            status="running",
            message="Initializing model evaluation...",
            progress=0.0,
            current_phase=3,  # Using phase 3 for evaluation
            start_time=time.time()
        )

        # Initialize environment and agent
        self._initialize_environment()

    def _initialize_environment(self):
        """Initialize the environment with the specified date range."""
        print(f"Initializing environment with tickers: {self.tickers}")
        print(f"Evaluation period: {self.start_date} to {self.end_date}")

        # Update progress
        progress_tracker.update_overall_progress(
            message="Preparing data for evaluation...",
            progress=0.1
        )

        # Initialize StockAnalyzer with the specified date range
        self.analyzer = StockAnalyzer(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Initialize environment
        self.env = PortfolioEnv(
            tickers=self.tickers,
            mode='eval',
            initial_capital=self.initial_capital
        )

        # Initialize agent
        self.agent = PortfolioAgent(
            self.env,
            buffer_size=5_000,
            train_batch_size=64,
            eval_batch_size=32
        )

        # Create a trainer instance for evaluation methods
        self.trainer = RLTrainer(
            eval_frequency=10,
            save_dir="memory",
            training_batch_size=64,
            eval_batch_size=32,
            rollout_episodes=self.eval_episodes,
            initial_capital=self.initial_capital
        )

        # Set the agent in the trainer
        self.trainer.env = self.env
        self.trainer.agent = self.agent

    def load_model(self):
        """Load the pretrained model."""
        print(f"Loading pretrained model from {self.model_path}")

        # Update progress
        progress_tracker.update_overall_progress(
            message="Loading pretrained model...",
            progress=0.3
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path)

            # Load model components
            self.agent.actor.load_state_dict(checkpoint['agent_state']['actor_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['agent_state']['critic_state_dict'])
            self.agent.meta_learner.load_state_dict(checkpoint['agent_state']['meta_learner_state_dict'])
            self.agent.market_encoder.load_state_dict(checkpoint['agent_state']['market_encoder_state_dict'])

            # Load market context if available
            if 'market_context' in checkpoint['agent_state']:
                self.agent.market_context = checkpoint['agent_state']['market_context']

            # Load adaptive parameters if available
            if 'adaptive_params' in checkpoint:
                self.agent.adaptive_params = checkpoint['adaptive_params']

            # Load adaptation bounds if available
            if 'adaptation_bounds' in checkpoint:
                self.agent.adaptation_bounds = checkpoint['adaptation_bounds']

            print("Successfully loaded pretrained model")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def evaluate(self):
        """
        Evaluate the pretrained model over the specified time period.

        Returns:
            Tuple containing (mean_score, sharpe_ratio, max_drawdown, evaluation_results)
        """
        # Update progress
        progress_tracker.update_overall_progress(
            message="Evaluating model performance...",
            progress=0.5
        )

        # Run evaluation using the trainer's evaluate method
        mean_score, sharpe_ratio, max_drawdown = self.trainer.evaluate(n_episodes=self.eval_episodes, final=True)

        # Get detailed evaluation results
        evaluation_results = {
            'mean_score': mean_score,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': self.trainer.episode_values if hasattr(self.trainer, 'episode_values') else [],
            'buy_and_hold_values': self.trainer.buy_and_hold_values if hasattr(self.trainer, 'buy_and_hold_values') else [],
            'final_eval_scores': self.trainer.final_eval_scores if hasattr(self.trainer, 'final_eval_scores') else []
        }

        # Update progress
        progress_tracker.update_overall_progress(
            status="completed",
            message="Model evaluation completed successfully!",
            progress=1.0,
            end_time=time.time()
        )

        return mean_score, sharpe_ratio, max_drawdown, evaluation_results


# Function to run evaluation from external scripts
def extract_tickers_from_model(model_path):
    """
    Extract the tickers that a model was trained on from the model checkpoint.

    Args:
        model_path: Path to the pretrained model checkpoint

    Returns:
        List of ticker symbols or None if not found
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path)

        # Check if the model has a metadata field with tickers
        if 'metadata' in checkpoint and 'tickers' in checkpoint['metadata']:
            return checkpoint['metadata']['tickers']

        # If not, try to infer from other parts of the checkpoint
        # Check if there's allocation history that might contain ticker information
        if 'allocation_history' in checkpoint and len(checkpoint['allocation_history']) > 0:
            # Allocation history might have ticker information
            allocation_entry = checkpoint['allocation_history'][0]
            if isinstance(allocation_entry, dict) and 'tickers' in allocation_entry:
                return allocation_entry['tickers']

        # Check if there's a history cache with ticker information
        if 'history_cache' in checkpoint and isinstance(checkpoint['history_cache'], dict):
            history_cache = checkpoint['history_cache']
            if 'tickers' in history_cache:
                return history_cache['tickers']

        # Try to find ticker information in the buffer if available
        if ('memory_state' in checkpoint and 'buffer' in checkpoint['memory_state'] and
            len(checkpoint['memory_state']['buffer']) > 0):
            # The buffer might contain state information with ticker data
            # This is a more complex extraction and might not be reliable
            pass

        # If we can't find ticker information in the checkpoint, check for a metadata file
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]

        # Check for different possible metadata file names
        possible_metadata_paths = [
            os.path.join(model_dir, f"{model_name}_metadata.json"),
            os.path.join(model_dir, "best_model_metadata.json") if model_name == "best_model" else None
        ]

        for metadata_path in possible_metadata_paths:
            if metadata_path and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if 'tickers' in metadata:
                        return metadata['tickers']
                except Exception as e:
                    print(f"Error reading metadata file {metadata_path}: {str(e)}")
                    continue

        # If we still can't find ticker information, return None
        return None

    except Exception as e:
        print(f"Error extracting tickers from model: {str(e)}")
        return None


def update_metadata_with_tickers(model_path, tickers, ticker_returns=None):
    """
    Update the metadata files associated with a model to use the specified tickers.

    Args:
        model_path: Path to the model file
        tickers: List of ticker symbols
        ticker_returns: Dictionary of ticker returns (optional)
    """
    try:
        # Get the model directory and name
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]

        # Find metadata files associated with this model
        metadata_files = []

        # Check for model-specific metadata file
        model_metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        if os.path.exists(model_metadata_path):
            metadata_files.append(model_metadata_path)

        # Check for best_model_metadata.json if this is the best model
        if model_name == "best_model":
            best_model_metadata_path = os.path.join(model_dir, "best_model_metadata.json")
            if os.path.exists(best_model_metadata_path):
                metadata_files.append(best_model_metadata_path)

        # If no metadata files found, create default ones
        if not metadata_files:
            # Create a default metadata file for this model
            if model_name == "best_model":
                metadata_files.append(os.path.join(model_dir, "best_model_metadata.json"))
            else:
                metadata_files.append(os.path.join(model_dir, f"{model_name}_metadata.json"))

        # If no ticker_returns provided, calculate them
        if ticker_returns is None:
            ticker_returns = {}
            try:
                # Initialize StockAnalyzer to get historical data
                analyzer = StockAnalyzer(tickers=tickers)

                # Calculate returns for each ticker
                for ticker in tickers:
                    try:
                        # Get historical data for the ticker
                        ticker_data = analyzer.data[analyzer.data['ticker'] == ticker]

                        if not ticker_data.empty:
                            # Calculate return based on first and last available prices
                            first_price = ticker_data.iloc[0]['Close']
                            last_price = ticker_data.iloc[-1]['Close']
                            pct_return = ((last_price - first_price) / first_price) * 100
                            ticker_returns[ticker] = round(pct_return, 2)
                        else:
                            ticker_returns[ticker] = 0.0
                    except Exception as e:
                        print(f"Error calculating return for {ticker}: {str(e)}")
                        ticker_returns[ticker] = 0.0
            except Exception as e:
                print(f"Error initializing StockAnalyzer: {str(e)}")
                # Set default returns if calculation fails
                for ticker in tickers:
                    ticker_returns[ticker] = 0.0

        # Update each metadata file
        for metadata_path in metadata_files:
            try:
                # Check if file exists
                if os.path.exists(metadata_path):
                    # Read existing metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Create new metadata with default values
                    metadata = {
                        "tickers": [],
                        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "initial_capital": 10000,
                        "model_version": "1.0"
                    }

                # Update tickers and ticker_returns
                metadata['tickers'] = tickers
                metadata['ticker_returns'] = ticker_returns

                # Write updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                print(f"Updated metadata file: {metadata_path}")
            except Exception as e:
                print(f"Error updating metadata file {metadata_path}: {str(e)}")

    except Exception as e:
        print(f"Error updating metadata with tickers: {str(e)}")

def evaluate_pretrained_model(
    model_path,
    tickers=None,
    start_date=None,
    end_date=None,
    initial_capital=10000,
    eval_episodes=10
):
    """
    Evaluate a pretrained model over a specific time period.

    Args:
        model_path: Path to the pretrained model checkpoint
        tickers: List of ticker symbols to evaluate on. If None, will try to use the tickers
                 the model was trained on.
        start_date: Start date for evaluation period (format: 'YYYY-MM-DD')
        end_date: End date for evaluation period (format: 'YYYY-MM-DD')
        initial_capital: Initial investment capital
        eval_episodes: Number of evaluation episodes to run

    Returns:
        Evaluation results dictionary
    """
    # If no tickers provided, try to extract them from the model
    if tickers is None:
        model_tickers = extract_tickers_from_model(model_path)
        if model_tickers:
            print(f"Using tickers from model: {model_tickers}")
            tickers = model_tickers
        else:
            print("Could not extract tickers from model, using default tickers")
            tickers = ['NVDA', 'FTNT']  # Default tickers
    else:
        # If tickers are provided and different from the model's tickers,
        # update the metadata files
        model_tickers = extract_tickers_from_model(model_path)
        if model_tickers and set(tickers) != set(model_tickers):
            print(f"Updating metadata files with new tickers: {tickers}")
            update_metadata_with_tickers(model_path, tickers)

    evaluator = ModelEvaluator(
        model_path=model_path,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        eval_episodes=eval_episodes
    )

    # Load the model
    if not evaluator.load_model():
        return {
            'success': False,
            'error': 'Failed to load model'
        }

    # Evaluate the model
    mean_score, sharpe_ratio, max_drawdown, results = evaluator.evaluate()

    # Add success flag to results
    results['success'] = True

    # Add the tickers used for evaluation to the results
    results['tickers'] = tickers

    return results



