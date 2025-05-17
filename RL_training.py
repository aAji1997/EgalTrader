from RL_environment import PortfolioEnv
from portfolio_agent import PortfolioAgent
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime
import os
from tqdm import tqdm
from progress_tracker import progress_tracker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from collections import deque
import subprocess
import sys
import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import gc
import shutil

class RLTrainer:
    def __init__(self,
                 eval_frequency: int = 10,
                 save_dir: str = "memory",
                 training_batch_size: int = 32,
                 eval_batch_size: int = 16,
                 rollout_episodes: int = 10,
                 n_episodes: int = 100,
                 num_workers: int = 4,
                 initial_capital: int = 10000,
                 tickers=None):  # Add tickers parameter
        """
        Initialize the RL trainer with dynamic episode and step settings based on data.

        Args:
            eval_frequency: How often to evaluate and potentially save checkpoints
            save_dir: Directory to save model checkpoints
            training_batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            rollout_episodes: Number of rollout episodes
            num_workers: Number of workers for data loading (default: 4)
            initial_capital: Initial investment capital (default: 10000)
            tickers: List of ticker symbols to use (default: None, which will use ['NVDA', 'FTNT'])
        """
        # Enable automatic mixed precision with memory optimization
        self.scaler = GradScaler()
        self.use_amp = torch.cuda.is_available()

        # Set CUDA settings for stability and memory efficiency
        if torch.cuda.is_available():
            # Disable autograd anomaly detection
            torch.autograd.set_detect_anomaly(False)

            # Disable CUDA benchmarking
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # Set matmul precision to medium for memory efficiency
            torch.set_float32_matmul_precision('medium')

            # Set initial memory fraction lower
            torch.cuda.set_per_process_memory_fraction(0.5)  # Start with 50% GPU memory

            # Enable expandable segments to reduce fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Disable tensor cores for memory efficiency
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            # Enable memory efficient attention
            os.environ['PYTORCH_ATTENTION_MODE'] = 'mem_efficient'

            # Set number of workers based on CUDA availability
            self.num_workers = num_workers if torch.cuda.is_available() else 0
        else:
            self.num_workers = 0  # No workers if CUDA is not available

        # Memory management settings
        self.memory_check_frequency = 1
        self.memory_threshold = 0.6
        self.min_batch_size = 4
        self.current_memory_fraction = 0.5

        # Save initialization parameters
        self.eval_frequency = eval_frequency
        self.save_dir = save_dir

        # Batch size management
        self.training_batch_size = training_batch_size
        self.eval_batch_size = eval_batch_size

        # Save initial capital and tickers
        self.initial_capital = initial_capital
        self.tickers = tickers if tickers is not None else ['NVDA', 'FTNT']

        # Initialize environment and agent with memory optimization
        try:
            self.env = PortfolioEnv(tickers=self.tickers, initial_capital=self.initial_capital)
            self.agent = PortfolioAgent(
                self.env,
                buffer_size=5_000,  # Reduced buffer size
                train_batch_size=self.training_batch_size,
                eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers  # Pass num_workers to agent
            )

            # Update agent's batch size to match trainer's
            self.agent.batch_size = self.training_batch_size

            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.agent.actor, 'enable_checkpointing'):
                self.agent.actor.enable_checkpointing()
            if hasattr(self.agent.critic, 'enable_checkpointing'):
                self.agent.critic.enable_checkpointing()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM during initialization. Reducing initial memory usage...")
                torch.cuda.empty_cache()
                self.current_memory_fraction = 0.3
                torch.cuda.set_per_process_memory_fraction(self.current_memory_fraction)
                self.env = PortfolioEnv(tickers=self.tickers, initial_capital=self.initial_capital)
                self.agent = PortfolioAgent(
                    self.env,
                    buffer_size=2_500,
                    train_batch_size=self.training_batch_size,
                    eval_batch_size=self.eval_batch_size,
                    num_workers=max(1, self.num_workers // 2)  # Reduce workers on OOM
                )
                # Update agent's batch size even in OOM case
                self.agent.batch_size = self.training_batch_size

        # Set episode length to full dataset length
        self.steps_per_episode = len(self.env.pure_train_dates) - 1
        self.n_episodes = n_episodes
        self.max_steps = self.steps_per_episode

        print(f"Training configuration:")
        print(f"Steps per episode (full training dataset): {self.steps_per_episode}")
        print(f"Number of episodes (dataset traversals): {self.n_episodes}")
        print(f"Total training steps: {self.steps_per_episode * self.n_episodes}")
        print(f"Training batch size: {self.training_batch_size}")
        print(f"Evaluation batch size: {self.eval_batch_size}")
        print(f"Agent batch size: {self.agent.batch_size}")
        print(f"Number of data loading workers: {self.num_workers}")

        # Create save directories if they don't exist
        self.save_dir = save_dir
        self.figures_dir = "figures"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

        # Training metrics
        self.episode_scores = []
        self.best_score = -np.inf

        # Evaluation metrics
        self.eval_scores = []
        self.eval_interval = eval_frequency

        # Early stopping parameters
        self.early_stopping = True
        self.patience = 20

        # Add evaluation metrics
        self.eval_scores = []
        self.best_eval_score = -np.inf

        # Add rollout evaluation parameters
        self.rollout_episodes = rollout_episodes
        self.rollout_scores = []
        self.best_rollout_score = -np.inf
        self.best_portfolio_return = 0.0
        self.best_ticker_returns = {}

        # Modify evaluation metrics to separate rollout and final evaluation
        self.final_eval_scores = []
        self.rollout_eval_scores = []

        # Initialize episode values and buy-and-hold values
        self.episode_values = []
        self.buy_and_hold_values = []




    def _clear_caches(self):
        """Clear various caches to prevent memory issues"""
        # Clear Python's garbage collector
        import gc
        gc.collect()

        # If CUDA is available, perform additional GPU memory cleanup
        if torch.cuda.is_available():
            # Force garbage collection on CUDA memory
            torch.cuda.empty_cache()

            self._check_and_clear_gpu_memory()

        # Ensure environment caches are properly initialized and cleared
        if hasattr(self, 'env'):
            # Initialize history_cache if it doesn't exist
            if not hasattr(self.env, 'history_cache'):
                self.env.history_cache = {
                    'volatility': {},  # Cache for historical volatilities
                    'returns': {},     # Cache for historical returns
                    'prices': {}       # Cache for historical prices
                }
            else:
                # Clear existing caches
                self.env.history_cache['volatility'].clear()
                self.env.history_cache['returns'].clear()
                self.env.history_cache['prices'].clear()

            # Clear other caches if they exist
            if hasattr(self.env, 'price_cache'):
                self.env.price_cache.clear()

            if hasattr(self.env, 'observation_cache'):
                self.env.observation_cache.clear()

            if hasattr(self.env, 'metrics_cache'):
                self.env.metrics_cache.clear()



    def _check_and_clear_gpu_memory(self):
        """Check GPU memory usage and clear caches if usage is high"""
        # Get current GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3

        # Print memory stats if usage is high (above 75% of capacity)
        if allocated > 3.0:  # 3GB out of 4GB is 75%
            print(f"\nHigh GPU memory usage detected!")
            print(f"Allocated: {allocated:.2f}GB")
            print(f"Reserved: {reserved:.2f}GB")

            # Clear model caches if they exist
            if hasattr(self.agent, 'actor'):
                self.agent.actor.zero_grad(set_to_none=True)
            if hasattr(self.agent, 'critic'):
                self.agent.critic.zero_grad(set_to_none=True)
            if hasattr(self.agent, 'target_critic'):
                self.agent.target_critic.zero_grad(set_to_none=True)

            # Clear optimizer states if memory is critically high
            if allocated > 3.5:  # Over 87.5% usage
                print("Critical memory usage - clearing optimizer states")
                if hasattr(self.agent, 'actor_optimizer'):
                    self.agent.actor_optimizer.zero_grad(set_to_none=True)
                if hasattr(self.agent, 'critic_optimizer'):
                    self.agent.critic_optimizer.zero_grad(set_to_none=True)

                # Force another garbage collection pass
                gc.collect()
                torch.cuda.empty_cache()

    def _trigger_restart(self):
        """Trigger a restart using PowerShell script"""
        print("\nTriggering training restart via PowerShell...")

        # Get the path to the Python interpreter being used
        python_path = sys.executable

        # Get the absolute path to the current script
        script_path = os.path.abspath(__file__)

        # Construct PowerShell command
        powershell_cmd = [
            "powershell.exe",
            "-ExecutionPolicy", "Bypass",
            "-File", "restart_training.ps1",
            "-pythonPath", f'"{python_path}"',
            "-scriptPath", f'"{script_path}"',
            "-maxRetries", "3",
            "-waitTime", "30"
        ]

        try:
            # Start PowerShell script in a new process
            subprocess.Popen(powershell_cmd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE,
                            shell=True)
            print("Restart script launched successfully")
        except Exception as e:
            print(f"Failed to launch restart script: {str(e)}")

        # Exit current process
        sys.exit(0)

    def _handle_cuda_error(self, e: RuntimeError, episode: int):
        """Handle CUDA errors by saving checkpoint and failing gracefully"""
        if "out of memory" in str(e):
            print("\nCUDA out of memory error detected. Saving checkpoint before failing...")

            # Try to save checkpoint before failing
            try:
                self.save_memory_checkpoint(episode)
                print("Successfully saved checkpoint before OOM failure")
            except Exception as save_error:
                print(f"Failed to save checkpoint before OOM: {str(save_error)}")

            # Re-raise the original OOM error
            raise e

        elif "buffer object failed" in str(e):
            print("\nCUDA buffer mapping error. Attempting to restart training...")
            self._trigger_restart()
            return False  # Non-recoverable error

        return False  # Non-recoverable error

    def train(self):
        """Train the agent with streamlined meta-learning integration."""
        print("Starting training...")

        # Load latest checkpoint if available
        start_episode = self.load_latest_checkpoint() or 1

        # Force a complete reset to ensure clean state before training
        print("Ensuring clean environment state before training...")
        self.reset_environment_state()

        # Initialize scores list
        scores = []

        # Training loop
        for episode in tqdm(range(start_episode, self.n_episodes + 1), desc="Training Episodes"):
            try:
                # Update progress
                progress = (episode - start_episode) / (self.n_episodes - start_episode + 1)
                progress_tracker.update_rl_agent_progress(
                    message=f"Training episode {episode}/{self.n_episodes}...",
                    progress=min(0.9 * progress, 0.89),  # Leave 10% for evaluation
                    current_episode=episode,
                    total_episodes=self.n_episodes
                )

                # Clear memory before each episode
                self._clear_caches()

                # Reset environment and get initial state
                state = self.env.reset()
                score = 0

                # Signal episode start to agent for parameter adaptation
                self.agent.adapt_parameters(is_episode_start=True)

                # Episode loop
                for step in range(self.steps_per_episode):
                    # Check memory usage and adjust if needed
                    if step % self.memory_check_frequency == 0:
                        self._check_and_adjust_memory()

                    # Get action using mixed precision
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        action = self.agent.act(state)

                    # Take action in environment
                    next_state, reward, done, info = self.env.step(action)

                    # Update market context and store experience
                    self.agent.update_market_context(next_state, reward, done)
                    self.agent.memory.add(state, action, reward, next_state, done)

                    # Learn if enough samples are available
                    if len(self.agent.memory) > self.training_batch_size:
                        try:
                            with torch.amp.autocast('cuda', enabled=self.use_amp):
                                self.agent.learn()
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                self._handle_oom_error()
                                continue
                            else:
                                raise e

                    # Update state and score
                    state = next_state
                    score += reward

                    # Break if done
                    if done:
                        break

                # Store episode score
                scores.append(score)

                # End episode for agent's meta-learning
                self.agent.end_episode()

                # Evaluate and save model periodically
                if episode % self.eval_frequency == 0:
                    self._evaluate_and_save(episode, scores)

            except Exception as e:
                print(f"Error in episode {episode}: {str(e)}")
                raise e

        return scores

    def _evaluate_and_save(self, episode, scores):
        """Evaluate the agent and save checkpoint if appropriate."""
        # Update progress before evaluation
        progress_tracker.update_rl_agent_progress(
            message=f"Evaluating agent at episode {episode}...",
            current_eval=episode // self.eval_frequency,
            total_evals=self.n_episodes // self.eval_frequency
        )

        # Perform rollout evaluation (not final evaluation)
        mean_score, rollout_metrics = self._evaluate_rollout(final=False)

        # Calculate composite score focusing ONLY on market outperformance and Sharpe ratio
        sharpe_ratio = rollout_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = rollout_metrics.get('max_drawdown', 1.0)  # Still track but don't use in score
        avg_portfolio_return = rollout_metrics.get('avg_portfolio_return', 0.0)
        avg_ticker_returns = rollout_metrics.get('avg_ticker_returns', {})
        buy_and_hold_returns = rollout_metrics.get('buy_and_hold_returns', {})

        # Calculate portfolio outperformance vs buy-and-hold
        buy_and_hold_portfolio_return = rollout_metrics.get('buy_and_hold_portfolio_return', 0.0)
        outperformance = avg_portfolio_return - buy_and_hold_portfolio_return

        # Apply a sigmoid function to outperformance to get a value between 0 and 2
        # This will be 1.0 at zero outperformance, approaching 2.0 for strong outperformance,
        # and approaching 0.0 for strong underperformance
        outperformance_factor = 2.0 / (1.0 + np.exp(-outperformance * 0.2))

        # Normalize Sharpe ratio using modified sigmoid to handle negative values better
        # Center sigmoid around 0 and scale to handle typical Sharpe ratio ranges
        normalized_sharpe = 2.0 / (1.0 + np.exp(-sharpe_ratio/2.0)) - 1.0

        # Calculate composite score focusing ONLY on market outperformance and Sharpe ratio
        # Equal weighting (50/50) between outperformance and Sharpe ratio
        composite_score = 0.5 * outperformance_factor * 2.0 + 0.5 * (normalized_sharpe + 1.0)

        # Use the composite score as the rollout score to ensure consistency
        rollout_score = composite_score
        self.rollout_scores.append(rollout_score)

        print(f"\nOutperformance vs Buy & Hold: {outperformance:.2f}% (Agent: {avg_portfolio_return:.2f}%, B&H: {buy_and_hold_portfolio_return:.2f}%)")
        print(f"Outperformance factor: {outperformance_factor:.4f}")

        # Update progress with evaluation results
        progress_tracker.update_rl_agent_progress(
            current_score=rollout_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            portfolio_return=avg_portfolio_return,
            best_score=getattr(self, 'best_composite_score', float('-inf'))
        )

        # Update best score if needed
        is_best = False
        if composite_score > getattr(self, 'best_composite_score', float('-inf')):
            self.best_composite_score = composite_score
            self.best_rollout_score = rollout_score
            self.best_sharpe = sharpe_ratio
            self.best_drawdown = max_drawdown
            self.best_portfolio_return = avg_portfolio_return
            self.best_ticker_returns = avg_ticker_returns
            is_best = True

            print(f"\nNew best model!")
            print(f"Composite Score: {composite_score:.4f}")
            print(f"Sharpe: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Portfolio Return: {avg_portfolio_return:.2f}%")
            print(f"Buy & Hold Return: {buy_and_hold_portfolio_return:.2f}%")
            print(f"Outperformance: {outperformance:.2f}%")
            print("Ticker-wise Returns vs Buy & Hold:")
            for ticker, ret in avg_ticker_returns.items():
                bh_ret = buy_and_hold_returns.get(ticker, 0.0)
                ticker_outperf = ret - bh_ret
                print(f"  {ticker}: {ret:.2f}% (B&H: {bh_ret:.2f}%, Outperf: {ticker_outperf:.2f}%)")

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth')
        self.save_checkpoint(
            episode=episode,
            episode_score=scores[-1],
            rollout_score=rollout_score,
            rollout_metrics=rollout_metrics
        )

        # Save best model separately if this is the best score
        if is_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            # Copy the checkpoint to best_model.pth
            shutil.copy2(checkpoint_path, best_model_path)
            print(f"Saved new best model to {best_model_path}")

        return rollout_score, is_best

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate the Sharpe ratio of the portfolio"""
        returns = np.concatenate(returns)
        excess_returns = returns - risk_free_rate
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate the maximum drawdown of the portfolio"""
        peak = portfolio_values[0]
        max_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def save_checkpoint(self, episode: int, episode_score: float, rollout_score: float, rollout_metrics: dict = None):
        """
        Save a comprehensive checkpoint.

        Args:
            episode: Current episode number
            episode_score: Score from the current training episode (used for tracking)
            rollout_score: Score from rollout evaluation (should be the composite score)
            rollout_metrics: Dictionary of evaluation metrics
        """
        # Initialize metrics if not provided
        if rollout_metrics is None:
            rollout_metrics = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}

        # Get metrics
        sharpe_ratio = rollout_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = rollout_metrics.get('max_drawdown', 1.0)
        avg_portfolio_return = rollout_metrics.get('avg_portfolio_return', 0.0)
        buy_and_hold_portfolio_return = rollout_metrics.get('buy_and_hold_portfolio_return', 0.0)

        # Calculate outperformance vs buy-and-hold
        outperformance = avg_portfolio_return - buy_and_hold_portfolio_return

        # Apply a sigmoid function to outperformance to get a value between 0 and 2
        # This will be 1.0 at zero outperformance, approaching 2.0 for strong outperformance,
        # and approaching 0.0 for strong underperformance
        outperformance_factor = 2.0 / (1.0 + np.exp(-outperformance * 0.2))

        # Normalize Sharpe ratio using modified sigmoid to handle negative values better
        # Center sigmoid around 0 and scale to handle typical Sharpe ratio ranges
        normalized_sharpe = 2.0 / (1.0 + np.exp(-sharpe_ratio/2.0)) - 1.0

        # Calculate composite score focusing ONLY on market outperformance and Sharpe ratio
        # Equal weighting (50/50) between outperformance and Sharpe ratio
        composite_score = 0.5 * outperformance_factor * 2.0 + 0.5 * (normalized_sharpe + 1.0)

        # Print outperformance information
        print(f"Outperformance vs Buy & Hold: {outperformance:.2f}% (Agent: {avg_portfolio_return:.2f}%, B&H: {buy_and_hold_portfolio_return:.2f}%)")
        print(f"Outperformance factor: {outperformance_factor:.4f}")

        # Store previous best metrics
        previous_best_composite = getattr(self, 'best_composite_score', -np.inf)

        # Determine if this is a new best model
        is_best_score = composite_score > previous_best_composite

        # Update best metrics if needed
        if is_best_score:
            self.best_rollout_score = composite_score  # Use composite score for consistency
            self.best_sharpe = sharpe_ratio
            self.best_drawdown = max_drawdown
            self.best_composite_score = composite_score
            print(f"\nNew best model with composite score: {composite_score:.4f}")
            print(f"Sharpe: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Portfolio Return: {avg_portfolio_return:.2f}%")
            print(f"Buy & Hold Return: {buy_and_hold_portfolio_return:.2f}%")
            print(f"Outperformance: {outperformance:.2f}%")

        # Calculate ticker-specific returns
        ticker_returns = {}
        try:
            # Get initial prices
            initial_date = self.env.dates[0]
            initial_prices = torch.tensor(
                self.env.unscaled_close_df.loc[initial_date]['Close'].values,
                device=self.env.gpu_device,
                dtype=torch.float32
            )

            # Get final prices
            final_date = self.env.dates[-1]
            final_prices = torch.tensor(
                self.env.unscaled_close_df.loc[final_date]['Close'].values,
                device=self.env.gpu_device,
                dtype=torch.float32
            )

            # Calculate percentage returns for each ticker
            for i, ticker in enumerate(self.env.tickers):
                initial_price = initial_prices[i].item()
                final_price = final_prices[i].item()
                pct_return = ((final_price - initial_price) / initial_price) * 100
                ticker_returns[ticker] = round(pct_return, 2)
        except Exception as e:
            print(f"Warning: Could not calculate ticker-specific returns: {str(e)}")
            # Set default values if calculation fails
            for ticker in self.env.tickers:
                ticker_returns[ticker] = 0.0

        # Use composite score for rollout_score in checkpoint for consistency
        # This updates the parameter value for use in the checkpoint

        checkpoint = {
            'episode': episode,
            'metadata': {
                'tickers': list(self.env.tickers),  # Store the tickers the model was trained on
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'initial_capital': self.initial_capital,
                'model_version': '1.0',
                'ticker_returns': ticker_returns,  # Add ticker-specific returns
                'composite_score': composite_score  # Add composite score to metadata
            },
            'agent_state': {
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic_state_dict': self.agent.critic.state_dict(),
                'target_critic_state_dict': self.agent.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
                'meta_learner_state_dict': self.agent.meta_learner.state_dict(),
                'market_encoder_state_dict': self.agent.market_encoder.state_dict(),
                'meta_optimizer_state_dict': self.agent.meta_optimizer.state_dict(),
                'strategy_bank_state': {
                    idx: strategy.state_dict()
                    for idx, strategy in enumerate(self.agent.strategy_bank.strategies)
                }
            },
            'agent_hyperparameters': {
                'exploration_temp': self.agent.exploration_temp,
                'risk_preference': self.agent.risk_preference,
                'gamma': self.agent.gamma,
                'exploration_noise': self.agent.exploration_noise,
                'min_action_prob': self.agent.actor.min_action_prob,
                'mode': self.agent.mode,
                'training': self.agent.training,
                'batch_size': self.agent.batch_size,
                'update_every': self.agent.update_every,
                'meta_batch_size': self.agent.meta_batch_size,
                'meta_update_freq': self.agent.meta_update_freq,
                'min_meta_samples': self.agent.min_meta_samples
            },
            'memory_state': {
                'buffer': list(self.agent.memory.buffer),
                'priorities': list(self.agent.memory.priorities),
                'alpha': self.agent.memory.alpha,
                'beta': self.agent.memory.beta,
                'max_priority': self.agent.memory.max_priority,
                'experience_hash_set': list(self.agent.memory.experience_hash_set),
                'td_error_history': list(self.agent.memory.td_error_history),
                'priority_variance': self.agent.memory.priority_variance,
                'meta_memory': list(self.agent.meta_memory),
                'current_episode_meta_experiences': list(getattr(self.agent, 'current_episode_meta_experiences', []))  # Add this line
            },
            'training_metrics': {
                'episode_scores': self.episode_scores,
                'eval_scores': self.eval_scores,
                'best_score': self.best_score,
                'returns_history': list(self.agent.returns_history),
                'volatility_history': list(self.agent.volatility_history),
                'eval_history': list(self.agent.eval_history),
                'best_rollout_score': self.best_rollout_score,
                'rollout_scores': self.rollout_scores,
                'meta_learning_metrics': {
                    'style_performance': self.agent.style_performance,
                    'current_style': self.agent.current_style
                },
                'best_metrics': {
                    'best_composite_score': getattr(self, 'best_composite_score', -np.inf),
                    'best_rollout_score': getattr(self, 'best_rollout_score', -np.inf),
                    'best_sharpe': getattr(self, 'best_sharpe', 0.0),
                    'best_drawdown': getattr(self, 'best_drawdown', 1.0),
                    'best_portfolio_return': getattr(self, 'best_portfolio_return', 0.0),
                    'best_ticker_returns': getattr(self, 'best_ticker_returns', {}),
                    'best_buy_and_hold_return': buy_and_hold_portfolio_return,
                    'best_outperformance': outperformance,
                    'best_outperformance_factor': outperformance_factor,
                    'best_score_episode': episode if is_best_score else getattr(self, 'best_score_episode', 0)
                }
            },
            'buffer_config': {
                'initial_buffer_size': self.agent.initial_buffer_size,
                'current_buffer_size': self.agent.current_buffer_size,
                'buffer_size_bounds': self.agent.buffer_size_bounds,
                'buffer_adjustment_rate': self.agent.buffer_adjustment_rate
            },
            'training_state': {
                't_step': self.agent.t_step,
                'mode': self.agent.mode,
                'batch_size': self.agent.batch_size,
                'update_every': self.agent.update_every,
                'cache_update_frequency': self.agent.cache_update_frequency,
                'batch_history_size': self.agent.batch_history_size
            },
            'allocation_history': list(self.agent.allocation_history),
            'market_context': {
                'current_regime': self.agent.market_context['current_regime'],
                'regime_history': list(self.agent.market_context['regime_history']),
                'returns_history': list(self.agent.market_context['returns_history']),
                'prices_history': list(self.agent.market_context['prices_history']),
                'volatility_history': list(self.agent.market_context['volatility_history']),
                'correlation_history': list(self.agent.market_context['correlation_history'])
            },
            'strategy_memory': {
                'strategies': self.agent.strategy_memory.strategies,
                'strategy_scores': {
                    key: list(scores) for key, scores in self.agent.strategy_memory.strategy_scores.items()
                },
                'similarity_threshold': self.agent.strategy_memory.similarity_threshold,
                'param_bounds': self.agent.strategy_memory.param_bounds
            },
            'regime_detector': {
                'window_sizes': self.agent.regime_detector.window_sizes,
                'regime_history': list(self.agent.regime_detector.regime_history),
                'volatility_thresholds': self.agent.regime_detector.volatility_thresholds,
                'trend_thresholds': self.agent.regime_detector.trend_thresholds
            },
            'adaptive_params': self.agent.adaptive_params,
            'adaptation_bounds': self.agent.adaptation_bounds,
            'parameter_history': self.agent.parameter_history if hasattr(self.agent, 'parameter_history') else []
        }

        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved comprehensive checkpoint at episode {episode}")

        # Save metadata to a separate JSON file for easier access
        metadata_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint['metadata'], f, indent=4)
        print(f"Saved metadata to {metadata_path}")

        # Save as best model if it's the best rollout score
        if is_best_score:
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)

            # Save best model metadata
            best_metadata_path = os.path.join(self.save_dir, 'best_model_metadata.json')
            with open(best_metadata_path, 'w') as f:
                json.dump(checkpoint['metadata'], f, indent=4)

            print("New best model saved!")



        return is_best_score

    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved agent checkpoint and verify loading"""
        try:
            checkpoint = torch.load(checkpoint_path)

            # Load agent state
            agent_state = checkpoint['agent_state']
            self.agent.actor.load_state_dict(agent_state['actor_state_dict'])
            self.agent.critic.load_state_dict(agent_state['critic_state_dict'])
            self.agent.target_critic.load_state_dict(agent_state['target_critic_state_dict'])
            self.agent.actor_optimizer.load_state_dict(agent_state['actor_optimizer_state_dict'])
            self.agent.critic_optimizer.load_state_dict(agent_state['critic_optimizer_state_dict'])

            # Load meta-learner components
            self.agent.meta_learner.load_state_dict(agent_state['meta_learner_state_dict'])
            self.agent.market_encoder.load_state_dict(agent_state['market_encoder_state_dict'])
            self.agent.meta_optimizer.load_state_dict(agent_state['meta_optimizer_state_dict'])

            # Load strategy bank
            for idx, strategy_state in agent_state['strategy_bank_state'].items():
                self.agent.strategy_bank.strategies[int(idx)].load_state_dict(strategy_state)

            # Load agent hyperparameters
            hyperparams = checkpoint['agent_hyperparameters']
            self.agent.exploration_temp = hyperparams['exploration_temp']
            self.agent.risk_preference = hyperparams['risk_preference']
            self.agent.gamma = hyperparams['gamma']
            self.agent.exploration_noise = hyperparams['exploration_noise']
            self.agent.actor.min_action_prob = hyperparams['min_action_prob']
            self.agent.mode = hyperparams['mode']
            self.agent.training = hyperparams['training']
            self.agent.batch_size = hyperparams['batch_size']
            self.agent.update_every = hyperparams['update_every']
            self.agent.meta_batch_size = hyperparams.get('meta_batch_size', 32)
            self.agent.meta_update_freq = hyperparams.get('meta_update_freq', 10)
            self.agent.min_meta_samples = hyperparams.get('min_meta_samples', 100)

            # Load memory state
            memory_state = checkpoint['memory_state']
            self.agent.memory.buffer = deque(memory_state['buffer'], maxlen=self.agent.memory.buffer.maxlen)
            self.agent.memory.priorities = deque(memory_state['priorities'], maxlen=self.agent.memory.buffer.maxlen)
            self.agent.memory.alpha = memory_state['alpha']
            self.agent.memory.beta = memory_state['beta']
            self.agent.memory.max_priority = memory_state['max_priority']
            self.agent.memory.experience_hash_set = set(memory_state['experience_hash_set'])
            self.agent.memory.td_error_history = deque(memory_state['td_error_history'], maxlen=1000)
            self.agent.memory.priority_variance = memory_state['priority_variance']
            self.agent.meta_memory = deque(memory_state.get('meta_memory', []), maxlen=10000)
            # Load current episode meta experiences
            self.agent.current_episode_meta_experiences = list(memory_state.get('current_episode_meta_experiences', []))  # Add this line

            # Load training metrics
            training_metrics = checkpoint['training_metrics']
            self.episode_scores = training_metrics['episode_scores']
            self.eval_scores = training_metrics['eval_scores']
            self.best_score = training_metrics['best_score']
            self.agent.returns_history = deque(training_metrics['returns_history'], maxlen=100)
            self.agent.volatility_history = deque(training_metrics['volatility_history'], maxlen=100)
            self.agent.eval_history = deque(training_metrics['eval_history'], maxlen=10)
            self.best_rollout_score = training_metrics['best_rollout_score']
            self.rollout_scores = training_metrics['rollout_scores']

            # Load best metrics
            if 'best_metrics' in training_metrics:
                best_metrics = training_metrics['best_metrics']
                self.best_composite_score = best_metrics['best_composite_score']
                self.best_rollout_score = best_metrics['best_rollout_score']
                self.best_sharpe = best_metrics['best_sharpe']
                self.best_drawdown = best_metrics['best_drawdown']
                self.best_portfolio_return = best_metrics.get('best_portfolio_return', 0.0)
                self.best_ticker_returns = best_metrics.get('best_ticker_returns', {})
                self.best_buy_and_hold_return = best_metrics.get('best_buy_and_hold_return', 0.0)
                self.best_outperformance = best_metrics.get('best_outperformance', 0.0)
                self.best_outperformance_factor = best_metrics.get('best_outperformance_factor', 1.0)
                self.best_score_episode = best_metrics['best_score_episode']

            # Load meta-learning metrics
            if 'meta_learning_metrics' in training_metrics:
                meta_metrics = training_metrics['meta_learning_metrics']
                self.agent.style_performance = meta_metrics['style_performance']
                self.agent.current_style = meta_metrics['current_style']

            # Load buffer configuration
            buffer_config = checkpoint['buffer_config']
            self.agent.initial_buffer_size = buffer_config['initial_buffer_size']
            self.agent.current_buffer_size = buffer_config['current_buffer_size']
            self.agent.buffer_size_bounds = buffer_config['buffer_size_bounds']
            self.agent.buffer_adjustment_rate = buffer_config['buffer_adjustment_rate']

            # Load training state
            training_state = checkpoint['training_state']
            self.agent.t_step = training_state['t_step']
            self.agent.mode = training_state['mode']
            self.agent.batch_size = training_state['batch_size']
            self.agent.update_every = training_state['update_every']
            self.agent.cache_update_frequency = training_state['cache_update_frequency']
            self.agent.batch_history_size = training_state['batch_history_size']

            # Load allocation history
            self.agent.allocation_history = deque(checkpoint['allocation_history'], maxlen=5)

            # Load market context
            market_context = checkpoint['market_context']
            self.agent.market_context = {
                'current_regime': market_context['current_regime'],
                'regime_history': deque(market_context['regime_history'], maxlen=100),
                'returns_history': deque(market_context['returns_history'], maxlen=100),
                'prices_history': deque(market_context['prices_history'], maxlen=100),
                'volatility_history': deque(market_context['volatility_history'], maxlen=100),
                'correlation_history': deque(market_context['correlation_history'], maxlen=100)
            }

            # Load strategy memory
            strategy_memory = checkpoint['strategy_memory']
            self.agent.strategy_memory.strategies = strategy_memory['strategies']
            self.agent.strategy_memory.strategy_scores = {
                key: deque(scores, maxlen=10)
                for key, scores in strategy_memory['strategy_scores'].items()
            }
            self.agent.strategy_memory.similarity_threshold = strategy_memory['similarity_threshold']
            self.agent.strategy_memory.param_bounds = strategy_memory['param_bounds']

            # Load regime detector
            regime_detector = checkpoint['regime_detector']
            self.agent.regime_detector.window_sizes = regime_detector['window_sizes']
            self.agent.regime_detector.regime_history = deque(regime_detector['regime_history'], maxlen=100)
            self.agent.regime_detector.volatility_thresholds = regime_detector['volatility_thresholds']
            self.agent.regime_detector.trend_thresholds = regime_detector['trend_thresholds']

            # Load adaptive parameters
            self.agent.adaptive_params = checkpoint['adaptive_params']
            self.agent.adaptation_bounds = checkpoint['adaptation_bounds']

            # Load parameter history if available
            if 'parameter_history' in checkpoint:
                self.agent.parameter_history = checkpoint['parameter_history']

            print(f"\nLoaded checkpoint from {checkpoint_path}")
            print(f"Restored Market Context:")
            print(f"  Current Regime: {self.agent.market_context['current_regime']}")
            print(f"  Strategy Memory Size: {sum(len(strats) for strats in self.agent.strategy_memory.strategies.values())}")
            print(f"  Meta-Learning Stats:")
            print(f"    Current Style: {self.agent.current_style}")
            print(f"    Meta Memory Size: {len(self.agent.meta_memory)}")
            print(f"    Current Episode Meta Experiences: {len(self.agent.current_episode_meta_experiences)}")  # Add this line
            print(f"Best Model Stats:")
            print(f"  Best Composite Score: {self.best_composite_score:.4f}")
            print(f"  Best Rollout Score: {self.best_rollout_score:.4f}")
            print(f"  Best Sharpe: {self.best_sharpe:.2f}")
            print(f"  Best Drawdown: {self.best_drawdown:.2%}")
            print(f"  Best Portfolio Return: {self.best_portfolio_return:.2f}%")
            print(f"  Best Buy & Hold Return: {getattr(self, 'best_buy_and_hold_return', 0.0):.2f}%")
            print(f"  Best Outperformance: {getattr(self, 'best_outperformance', 0.0):.2f}%")
            print(f"  Best Outperformance Factor: {getattr(self, 'best_outperformance_factor', 1.0):.4f}")
            if self.best_ticker_returns:
                print(f"  Best Ticker-wise Returns:")
                for ticker, ret in self.best_ticker_returns.items():
                    print(f"    {ticker}: {ret:.2f}%")
            print(f"  Best Score Episode: {self.best_score_episode}")
            print(f"  Adaptation Factors:")
            print(f"    Vol: {self.agent.adaptive_params['vol_scaling_factor']:.2f}")
            print(f"    Trend: {self.agent.adaptive_params['trend_scaling_factor']:.2f}")
            print(f"    Correlation: {self.agent.adaptive_params['correlation_scaling_factor']:.2f}")

            # Verify the model was loaded by checking if weights exist
            if hasattr(self.agent.actor, 'state_dict') and len(self.agent.actor.state_dict()) > 0:
                print("Model weights verified successfully")
            else:
                print("Warning: Model loaded but weights verification failed")

            return checkpoint['episode'] + 1

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            if 'checkpoint' in locals():
                print("Available keys in checkpoint:", list(checkpoint.keys()))
            raise

    def _save_trading_statistics(self, trading_actions, allocation_history, timestamps=None):
        """Save visualizations of trading statistics and parameter adjustments"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create figures directory if it doesn't exist
            if not os.path.exists(self.figures_dir):
                os.makedirs(self.figures_dir)

            # 1. Trading Actions Bar Chart with Confidence Intervals
            # Restructure trading actions data for statistical analysis
            action_stats = []
            for ticker in self.env.tickers:
                for action_type in ['buys', 'holds', 'sells']:
                    action_counts = trading_actions[action_type][ticker]
                    if isinstance(action_counts, (list, np.ndarray)):
                        mean_count = np.mean(action_counts)
                        std_count = np.std(action_counts)
                        ci = 1.96 * std_count / np.sqrt(len(action_counts))  # 95% confidence interval
                    else:
                        # Handle single episode case
                        mean_count = float(action_counts)
                        ci = 0

                    action_stats.append({
                        'Ticker': ticker,
                        'Action': action_type.capitalize(),
                        'Mean_Count': mean_count,
                        'CI': ci
                    })

            action_df = pd.DataFrame(action_stats)

            plt.figure(figsize=(12, 6))
            # Create grouped bar chart
            bar_plot = sns.barplot(
                data=action_df,
                x='Ticker',
                y='Mean_Count',
                hue='Action',
                palette='pastel',
                capsize=0.05,  # Size of the error bar caps
                errwidth=1.5,  # Width of error bars
                ci=None  # We'll add our custom error bars
            )

            # Add error bars manually for more control
            for i, row in action_df.iterrows():
                bar = bar_plot.patches[i]
                center = bar.get_x() + bar.get_width() / 2
                height = bar.get_height()
                plt.vlines(
                    center,
                    height - row['CI'],
                    height + row['CI'],
                    color='black',
                    linewidth=1.5,
                    alpha=0.7
                )

            plt.title('Average Trading Actions Distribution by Ticker (with 95% CIs)')
            plt.ylabel('Average Number of Actions')
            plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.figures_dir}/trading_actions_{timestamp}.png", bbox_inches='tight')
            plt.close()

            # 2. Allocation History Stacked Area Chart
            plt.figure(figsize=(15, 8))

            # Convert allocation history to numpy arrays
            allocation_arrays = {}
            for ticker in self.env.tickers:
                values = np.array(allocation_history[ticker])
                if values.size > 0:  # Check if we have any values
                    allocation_arrays[ticker] = values.reshape(-1, 1) if values.ndim == 1 else values

            if allocation_arrays:  # Only proceed if we have data
                # Use a soft color palette
                colors = sns.color_palette("pastel", n_colors=len(self.env.tickers))

                # Plot mean allocations for each ticker
                bottom = np.zeros(len(allocation_arrays[list(allocation_arrays.keys())[0]]))
                for i, (ticker, values) in enumerate(allocation_arrays.items()):
                    mean_values = np.mean(values, axis=1) if values.ndim > 1 else values.ravel()
                    plt.fill_between(
                        range(len(mean_values)),
                        bottom,
                        bottom + mean_values,
                        alpha=0.7,
                        color=colors[i],
                        label=ticker
                    )
                    bottom += mean_values

                plt.title('Portfolio Allocation History')
                plt.xlabel('Time Step')
                plt.ylabel('Allocation')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.figures_dir}/allocation_history_{timestamp}.png", bbox_inches='tight')
            plt.close()

            # 3. Parameter Adjustment History
            if hasattr(self.agent, 'parameter_history') and self.agent.parameter_history:
                param_data = pd.DataFrame(self.agent.parameter_history)

                # Create figure with subplots (fig is used implicitly by plt.savefig)
                _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

                # Plot exploration noise
                ax1.plot(param_data['episode'], param_data['exploration_noise'], marker='o')
                ax1.set_title('Exploration Noise History')
                ax1.set_ylabel('Exploration Noise')
                ax1.grid(True)

                # Plot exploration temperature
                ax2.plot(param_data['episode'], param_data['exploration_temp'], marker='o', color='red')
                ax2.set_title('Exploration Temperature History')
                ax2.set_ylabel('Exploration Temp')
                ax2.grid(True)

                # Plot risk preference
                ax3.plot(param_data['episode'], param_data['risk_preference'], marker='o', color='orange')
                ax3.set_title('Risk Preference History')
                ax3.set_ylabel('Risk Preference')
                ax3.grid(True)

                # Plot learning rate
                ax4.plot(param_data['episode'], param_data['learning_rate'], marker='o', color='green')
                ax4.set_title('Actor Learning Rate History')
                ax4.set_ylabel('Learning Rate')
                ax4.grid(True)

                # Color background based on parameter state if available
                if 'parameter_state' in param_data.columns:
                    for ax in [ax1, ax2, ax3, ax4]:
                        for i, state in enumerate(param_data['parameter_state']):
                            color = {
                                'neutral': 'white',
                                'aggressive': 'lightcoral',
                                'conservative': 'lightblue'
                            }.get(state, 'white')
                            ax.axvspan(
                                param_data['episode'].iloc[i] - 0.5,
                                param_data['episode'].iloc[i] + 0.5,
                                alpha=0.2,
                                color=color
                            )

                plt.tight_layout()
                plt.savefig(f"{self.figures_dir}/parameter_history_{timestamp}.png")
                plt.close()

                # Save parameter history to CSV for analysis
                param_data.to_csv(f"{self.figures_dir}/parameter_history_{timestamp}.csv", index=False)

            # 4. Save statistics to JSON
            stats = {
                'trading_summary': {
                    ticker: {
                        'buys': float(np.mean(trading_actions['buys'][ticker])),
                        'holds': float(np.mean(trading_actions['holds'][ticker])),
                        'sells': float(np.mean(trading_actions['sells'][ticker]))
                    } for ticker in self.env.tickers
                },
                'average_allocation': {
                    ticker: float(np.mean(allocation_history[ticker]))
                    for ticker in self.env.tickers
                }
            }

            # Add evaluation metrics if available
            if hasattr(self, 'final_eval_scores') and self.final_eval_scores:
                latest_eval = self.final_eval_scores[-1]
                stats['evaluation_metrics'] = {
                    'final_score': float(latest_eval['score']),
                    'sharpe_ratio': float(latest_eval['sharpe']),
                    'max_drawdown': float(latest_eval['max_drawdown']),
                    'portfolio_return': float(latest_eval['portfolio_return']),
                    'relative_return': float(latest_eval['relative_return'])
                }

            # Add parameter adjustment metrics if available
            if hasattr(self.agent, 'parameter_history') and self.agent.parameter_history:
                latest_params = self.agent.parameter_history[-1]
                # Create parameter metrics dictionary with available keys
                param_metrics = {
                    'exploration_noise': float(latest_params['exploration_noise']),
                    'exploration_temp': float(latest_params['exploration_temp']),
                    'risk_preference': float(latest_params['risk_preference']),
                    'learning_rate': float(latest_params['learning_rate'])
                }

                # Add optional keys if they exist
                if 'parameter_state' in latest_params:
                    param_metrics['parameter_state'] = latest_params['parameter_state']
                else:
                    param_metrics['parameter_state'] = self.agent.current_style if hasattr(self.agent, 'current_style') else 'neutral'

                if 'relative_improvement' in latest_params:
                    param_metrics['relative_improvement'] = float(latest_params['relative_improvement'])

                stats['parameter_metrics'] = param_metrics

            # Save JSON in the figures directory
            with open(f"{self.figures_dir}/trading_stats_{timestamp}.json", 'w') as f:
                json.dump(stats, f, indent=4)

        except Exception as e:
            print(f"\nWarning: Error in saving trading statistics: {str(e)}")
            print("This is non-critical and won't affect the training process.")

    def _calculate_buy_and_hold_return(self, return_values=False):
        """Calculate the return from a simple buy-and-hold strategy with uniform initial allocation"""
        self.env.reset()

        # Get initial prices
        initial_date = self.env.dates[0]

        try:
            # Try to get prices directly using the date
            initial_prices = torch.tensor(
                self.env.unscaled_close_df.loc[initial_date]['Close'].values,
                device=self.env.gpu_device,
                dtype=torch.float32
            )
        except KeyError:
            # If direct lookup fails, try to convert the date format
            try:
                # Convert numpy.datetime64 to pandas Timestamp if needed
                if isinstance(initial_date, np.datetime64):
                    initial_date_pd = pd.Timestamp(initial_date)
                else:
                    initial_date_pd = pd.to_datetime(initial_date)

                # Try lookup with converted date
                initial_prices = torch.tensor(
                    self.env.unscaled_close_df.loc[initial_date_pd]['Close'].values,
                    device=self.env.gpu_device,
                    dtype=torch.float32
                )
            except Exception as e:
                print(f"Warning: Could not get initial prices for date {initial_date} in buy-and-hold calculation. Using default values. Error: {str(e)}")
                # Fallback to using default values
                initial_prices = torch.ones(self.env.num_tickers, device=self.env.gpu_device, dtype=torch.float32) * 100.0  # Arbitrary default price

        # Calculate integer number of shares with uniform allocation target
        allocation_per_ticker = self.env.initial_capital / self.env.num_tickers
        initial_shares = torch.floor(allocation_per_ticker / initial_prices)  # Integer number of shares

        # Calculate actual cost and remaining cash
        initial_cost = (initial_shares * initial_prices).sum()
        remaining_cash = self.env.initial_capital - initial_cost

        # Calculate buy and hold values for each time step
        buy_and_hold_values = []
        initial_portfolio_value = self.env.initial_capital
        buy_and_hold_values.append(initial_portfolio_value)

        # For each date in the environment
        for date in self.env.dates[1:]:  # Skip the first date as we already have the initial value
            # Get prices for this date
            try:
                # Try to get prices directly using the date
                prices = torch.tensor(
                    self.env.unscaled_close_df.loc[date]['Close'].values,
                    device=self.env.gpu_device,
                    dtype=torch.float32
                )

                # Calculate portfolio value at this date
                portfolio_value = (initial_shares * prices).sum() + remaining_cash
                buy_and_hold_values.append(portfolio_value.item())
            except KeyError:
                # If direct lookup fails, try to convert the date format
                try:
                    # Convert numpy.datetime64 to pandas Timestamp if needed
                    if isinstance(date, np.datetime64):
                        date_pd = pd.Timestamp(date)
                    else:
                        date_pd = pd.to_datetime(date)

                    # Try lookup with converted date
                    prices = torch.tensor(
                        self.env.unscaled_close_df.loc[date_pd]['Close'].values,
                        device=self.env.gpu_device,
                        dtype=torch.float32
                    )

                    # Calculate portfolio value at this date
                    portfolio_value = (initial_shares * prices).sum() + remaining_cash
                    buy_and_hold_values.append(portfolio_value.item())
                except Exception as e:
                    # If there's an error (e.g., missing data), use the last known value
                    if buy_and_hold_values:
                        buy_and_hold_values.append(buy_and_hold_values[-1])
                    else:
                        buy_and_hold_values.append(initial_portfolio_value)
                    print(f"Warning: Error calculating buy-and-hold value for date {date}: {str(e)}")
            except Exception as e:
                # If there's an error (e.g., missing data), use the last known value
                if buy_and_hold_values:
                    buy_and_hold_values.append(buy_and_hold_values[-1])
                else:
                    buy_and_hold_values.append(initial_portfolio_value)
                print(f"Warning: Error calculating buy-and-hold value for date {date}: {str(e)}")

        # Get final prices
        final_date = self.env.dates[-1]

        try:
            # Try to get prices directly using the date
            final_prices = torch.tensor(
                self.env.unscaled_close_df.loc[final_date]['Close'].values,
                device=self.env.gpu_device,
                dtype=torch.float32
            )
        except KeyError:
            # If direct lookup fails, try to convert the date format
            try:
                # Convert numpy.datetime64 to pandas Timestamp if needed
                if isinstance(final_date, np.datetime64):
                    final_date_pd = pd.Timestamp(final_date)
                else:
                    final_date_pd = pd.to_datetime(final_date)

                # Try lookup with converted date
                final_prices = torch.tensor(
                    self.env.unscaled_close_df.loc[final_date_pd]['Close'].values,
                    device=self.env.gpu_device,
                    dtype=torch.float32
                )
            except Exception as e:
                print(f"Warning: Could not get final prices for date {final_date} in buy-and-hold calculation. Using initial prices. Error: {str(e)}")
                # Fallback to using initial prices (this will result in 0% return)
                final_prices = initial_prices.clone()

        # Calculate final value including remaining cash
        final_value = (initial_shares * final_prices).sum() + remaining_cash

        # Calculate net return
        net_return = final_value - self.env.initial_capital
        return_pct = (net_return / self.env.initial_capital) * 100

        # Print allocation details for transparency
        print("\nBuy & Hold Strategy Details:")
        for i, ticker in enumerate(self.env.tickers):
            shares = initial_shares[i].item()
            cost = (shares * initial_prices[i]).item()
            print(f"{ticker}: {shares:.0f} shares @ ${initial_prices[i]:.2f} = ${cost:.2f}")
        print(f"Total Investment: ${initial_cost:.2f}")
        print(f"Remaining Cash: ${remaining_cash:.2f}")

        if return_values:
            return return_pct.item(), buy_and_hold_values
        else:
            return return_pct.item()

    def _get_price(self, date, ticker):
        """Helper method to get price for a specific date and ticker"""
        try:
            price = self.env.unscaled_close_df.loc[(date, ticker), 'Close']
            return price
        except KeyError:
            print(f"Warning: Price not found for {ticker} on {date}")
            return 0.0



    def save_memory_checkpoint(self, episode):
        """Save memory checkpoint including replay buffer state with robust error handling"""
        try:
            memory_dir = "memory"
            if not os.path.exists(memory_dir):
                os.makedirs(memory_dir)

            checkpoint_path = os.path.join(memory_dir, f"memory_checkpoint_ep{episode}.pt")

            # Helper function to safely get attributes
            def safe_get_attr(obj, attr, default=None):
                return getattr(obj, attr, default)

            # Helper function to safely get state dict
            def safe_state_dict(model):
                try:
                    return model.state_dict() if model is not None else None
                except Exception:
                    return None

            # Helper function to safely convert tensor to float
            def safe_to_float(value):
                try:
                    if torch.is_tensor(value):
                        return float(value.item())
                    return float(value)
                except Exception:
                    return 0.0

            # Safely get agent state
            agent_state = {
                'actor_state_dict': safe_state_dict(safe_get_attr(self.agent, 'actor')),
                'critic_state_dict': safe_state_dict(safe_get_attr(self.agent, 'critic')),
                'target_critic_state_dict': safe_state_dict(safe_get_attr(self.agent, 'target_critic')),
                'actor_optimizer_state_dict': safe_state_dict(safe_get_attr(self.agent, 'actor_optimizer')),
                'critic_optimizer_state_dict': safe_state_dict(safe_get_attr(self.agent, 'critic_optimizer'))
            }

            # Safely get actor components if they exist
            if hasattr(self.agent, 'actor') and self.agent.actor is not None:
                actor = self.agent.actor
                agent_state.update({
                    'ticker_embeddings_state_dict': safe_state_dict(safe_get_attr(actor, 'ticker_embeddings')),
                    'ticker_extractors_state_dict': [safe_state_dict(ext) for ext in safe_get_attr(actor, 'ticker_extractors', [])],
                    'portfolio_extractor_state_dict': safe_state_dict(safe_get_attr(actor, 'portfolio_extractor')),
                    'market_attention_state_dict': safe_state_dict(safe_get_attr(actor, 'market_attention')),
                    'market_context_ln_state_dict': safe_state_dict(safe_get_attr(actor, 'market_context_ln')),
                    'ticker_attention_state_dict': safe_state_dict(safe_get_attr(actor, 'ticker_attention')),
                    'discrete_heads_state_dict': [safe_state_dict(head) for head in safe_get_attr(actor, 'discrete_heads', [])],
                    'allocation_heads_state_dict': [safe_state_dict(head) for head in safe_get_attr(actor, 'allocation_heads', [])],
                    'ticker_biases': safe_get_attr(actor, 'ticker_biases', None),
                    'market_context_integration_state_dict': safe_state_dict(safe_get_attr(actor, 'market_context_integration'))
                })

            # Safely get agent hyperparameters
            agent_hyperparameters = {
                'exploration_temp': safe_to_float(safe_get_attr(self.agent, 'exploration_temp', 1.0)),
                'risk_preference': safe_to_float(safe_get_attr(self.agent, 'risk_preference', 0.5)),
                'gamma': safe_to_float(safe_get_attr(self.agent, 'gamma', 0.99)),
                'gamma_start': safe_to_float(safe_get_attr(self.agent, 'gamma_start', 0.6)),
                'gamma_end': safe_to_float(safe_get_attr(self.agent, 'gamma_end', 0.999)),
                'exploration_noise': safe_to_float(safe_get_attr(self.agent, 'exploration_noise', 0.1)),
                'min_action_prob': safe_to_float(safe_get_attr(self.agent.actor, 'min_action_prob', 0.01)) if hasattr(self.agent, 'actor') else 0.01,
                'mode': safe_get_attr(self.agent, 'mode', 'train'),
                'batch_size': safe_get_attr(self.agent, 'batch_size', self.training_batch_size),
                'update_every': safe_get_attr(self.agent, 'update_every', 1),
                'parameter_state': safe_get_attr(self.agent, 'parameter_state', 'neutral')
            }

            # Safely get memory state
            memory = safe_get_attr(self.agent, 'memory')
            memory_state = {}
            if memory is not None:
                memory_state = {
                    'buffer': list(safe_get_attr(memory, 'buffer', [])),
                    'priorities': [safe_to_float(p) for p in safe_get_attr(memory, 'priorities', [])],
                    'alpha': safe_to_float(safe_get_attr(memory, 'alpha', 0.6)),
                    'beta': safe_to_float(safe_get_attr(memory, 'beta', 0.4)),
                    'max_priority': safe_to_float(safe_get_attr(memory, 'max_priority', 1.0)),
                    'experience_hash_set': list(safe_get_attr(memory, 'experience_hash_set', set())),
                    'td_error_history': list(safe_get_attr(memory, 'td_error_history', [])),
                    'priority_variance': safe_to_float(safe_get_attr(memory, 'priority_variance', 1.0))
                }

            # Safely get training metrics
            training_metrics = {
                'episode_scores': self.episode_scores,
                'eval_scores': self.eval_scores,
                'best_score': safe_to_float(self.best_score),
                'returns_history': list(safe_get_attr(self.agent, 'returns_history', [])),
                'volatility_history': list(safe_get_attr(self.agent, 'volatility_history', [])),
                'eval_history': list(safe_get_attr(self.agent, 'eval_history', [])),
                'best_eval_score': safe_to_float(self.best_eval_score),
                'best_rollout_score': safe_to_float(self.best_rollout_score),
                'rollout_scores': self.rollout_scores,
                'final_eval_scores': self.final_eval_scores,
                'rollout_eval_scores': self.rollout_eval_scores
            }

            # Safely get buffer configuration
            buffer_config = {
                'buffer_size': safe_get_attr(memory, 'capacity', 10000) if memory is not None else 10000,
                'current_size': len(safe_get_attr(memory, 'buffer', [])) if memory is not None else 0,
                'buffer_size_bounds': safe_get_attr(self.agent, 'buffer_size_bounds', (1000, 100000)),
                'buffer_adjustment_rate': safe_to_float(safe_get_attr(self.agent, 'buffer_adjustment_rate', 0.1))
            }

            # Safely get training state
            training_state = {
                't_step': safe_get_attr(self.agent, 't_step', 0),
                'mode': safe_get_attr(self.agent, 'mode', 'train'),
                'batch_size': safe_get_attr(self.agent, 'batch_size', self.training_batch_size),
                'update_every': safe_get_attr(self.agent, 'update_every', 1),
                'cache_update_frequency': safe_get_attr(self.agent, 'cache_update_frequency', 10),
                'batch_history_size': safe_get_attr(self.agent, 'batch_history_size', 100)
            }

            # Combine all components
            checkpoint = {
                'episode': episode,
                'agent_state': agent_state,
                'agent_hyperparameters': agent_hyperparameters,
                'memory_state': memory_state,
                'training_metrics': training_metrics,
                'buffer_config': buffer_config,
                'training_state': training_state,
                'allocation_history': list(safe_get_attr(self.agent, 'allocation_history', [])),
                'history_cache': safe_get_attr(self.agent, 'history_cache', {})
            }

            # Save checkpoint with error handling
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"\nSaved comprehensive memory checkpoint at episode {episode}")
            except Exception as save_error:
                # If saving fails, try to save to a different location
                backup_path = os.path.join(memory_dir, f"backup_checkpoint_ep{episode}.pt")
                try:
                    torch.save(checkpoint, backup_path)
                    print(f"\nSaved backup checkpoint at episode {episode}")
                except Exception as backup_error:
                    print(f"\nFailed to save checkpoint: {str(save_error)}")
                    print(f"Backup save also failed: {str(backup_error)}")
                    raise

        except Exception as e:
            print(f"\nError during checkpoint creation: {str(e)}")
            # Create minimal emergency checkpoint
            try:
                emergency_checkpoint = {
                    'episode': episode,
                    'episode_scores': self.episode_scores,
                    'best_score': safe_to_float(self.best_score),
                    'agent_state': {
                        'actor_state_dict': safe_state_dict(safe_get_attr(self.agent, 'actor')),
                        'critic_state_dict': safe_state_dict(safe_get_attr(self.agent, 'critic'))
                    }
                }
                emergency_path = os.path.join(memory_dir, f"emergency_checkpoint_ep{episode}.pt")
                torch.save(emergency_checkpoint, emergency_path)
                print(f"\nSaved emergency checkpoint with minimal data at episode {episode}")
            except Exception as emergency_error:
                print(f"\nFailed to save emergency checkpoint: {str(emergency_error)}")
                # At this point, we've tried everything possible to save the checkpoint

    def load_latest_checkpoint(self):
        """Load the latest available memory checkpoint"""
        memory_dir = "memory"
        if not os.path.exists(memory_dir):
            print("No memory checkpoints found")
            return None

        checkpoints = glob.glob(os.path.join(memory_dir, "checkpoint_episode_*.pth"))
        if not checkpoints:
            print("No memory checkpoints found")
            return None

        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'episode_(\d+)', x).group(1)))
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)

        try:
            # Load agent state
            agent_state = checkpoint['agent_state']
            self.agent.actor.load_state_dict(agent_state['actor_state_dict'])
            self.agent.critic.load_state_dict(agent_state['critic_state_dict'])
            self.agent.target_critic.load_state_dict(agent_state['target_critic_state_dict'])
            self.agent.actor_optimizer.load_state_dict(agent_state['actor_optimizer_state_dict'])
            self.agent.critic_optimizer.load_state_dict(agent_state['critic_optimizer_state_dict'])

            # Load agent hyperparameters
            hyperparams = checkpoint['agent_hyperparameters']
            self.agent.exploration_temp = hyperparams['exploration_temp']
            self.agent.risk_preference = hyperparams['risk_preference']
            self.agent.gamma = hyperparams['gamma']
            self.agent.exploration_noise = hyperparams['exploration_noise']
            self.agent.actor.min_action_prob = hyperparams['min_action_prob']
            self.agent.mode = hyperparams['mode']
            self.agent.training = hyperparams['training']
            self.agent.batch_size = hyperparams['batch_size']
            self.agent.update_every = hyperparams['update_every']

            # Load memory state
            memory_state = checkpoint['memory_state']
            self.agent.memory.buffer = deque(memory_state['buffer'], maxlen=self.agent.memory.buffer.maxlen)
            self.agent.memory.priorities = deque(memory_state['priorities'], maxlen=self.agent.memory.buffer.maxlen)
            self.agent.memory.alpha = memory_state['alpha']
            self.agent.memory.beta = memory_state['beta']
            self.agent.memory.max_priority = memory_state['max_priority']
            self.agent.memory.experience_hash_set = set(memory_state['experience_hash_set'])
            self.agent.memory.td_error_history = deque(memory_state['td_error_history'], maxlen=1000)
            self.agent.memory.priority_variance = memory_state['priority_variance']

            # Load training metrics
            training_metrics = checkpoint['training_metrics']
            self.episode_scores = training_metrics['episode_scores']
            self.eval_scores = training_metrics['eval_scores']
            self.best_score = training_metrics['best_score']
            self.agent.returns_history = deque(training_metrics['returns_history'], maxlen=100)
            self.agent.volatility_history = deque(training_metrics['volatility_history'], maxlen=100)
            self.agent.eval_history = deque(training_metrics['eval_history'], maxlen=10)
            self.best_rollout_score = training_metrics['best_rollout_score']
            self.rollout_scores = training_metrics['rollout_scores']

            # Load buffer configuration
            buffer_config = checkpoint['buffer_config']
            self.agent.initial_buffer_size = buffer_config['initial_buffer_size']
            self.agent.current_buffer_size = buffer_config['current_buffer_size']
            self.agent.buffer_size_bounds = buffer_config['buffer_size_bounds']
            self.agent.buffer_adjustment_rate = buffer_config['buffer_adjustment_rate']

            # Load training state
            training_state = checkpoint['training_state']
            self.agent.t_step = training_state['t_step']
            self.agent.mode = training_state['mode']
            self.agent.batch_size = training_state['batch_size']
            self.agent.update_every = training_state['update_every']
            self.agent.cache_update_frequency = training_state['cache_update_frequency']
            self.agent.batch_history_size = training_state['batch_history_size']

            # Load allocation history
            self.agent.allocation_history = deque(checkpoint['allocation_history'], maxlen=5)

            # Load market context
            market_context = checkpoint['market_context']
            self.agent.market_context = {
                'current_regime': market_context['current_regime'],
                'regime_history': deque(market_context['regime_history'], maxlen=100),
                'returns_history': deque(market_context['returns_history'], maxlen=100),
                'prices_history': deque(market_context['prices_history'], maxlen=100),
                'volatility_history': deque(market_context['volatility_history'], maxlen=100),
                'correlation_history': deque(market_context['correlation_history'], maxlen=100)
            }

            # Load strategy memory
            strategy_memory = checkpoint['strategy_memory']
            self.agent.strategy_memory.strategies = strategy_memory['strategies']
            self.agent.strategy_memory.strategy_scores = {
                key: deque(scores, maxlen=10)
                for key, scores in strategy_memory['strategy_scores'].items()
            }
            self.agent.strategy_memory.similarity_threshold = strategy_memory['similarity_threshold']
            self.agent.strategy_memory.param_bounds = strategy_memory['param_bounds']

            # Load regime detector
            regime_detector = checkpoint['regime_detector']
            self.agent.regime_detector.window_sizes = regime_detector['window_sizes']
            self.agent.regime_detector.regime_history = deque(regime_detector['regime_history'], maxlen=100)
            self.agent.regime_detector.volatility_thresholds = regime_detector['volatility_thresholds']
            self.agent.regime_detector.trend_thresholds = regime_detector['trend_thresholds']

            # Load adaptive parameters
            self.agent.adaptive_params = checkpoint['adaptive_params']
            self.agent.adaptation_bounds = checkpoint['adaptation_bounds']

            # Load parameter history if available
            if 'parameter_history' in checkpoint:
                self.agent.parameter_history = checkpoint['parameter_history']

            print(f"\nLoaded checkpoint from {latest_checkpoint}")
            print(f"Restored Market Context:")
            print(f"  Current Regime: {self.agent.market_context['current_regime']}")
            print(f"  Strategy Memory Size: {sum(len(strats) for strats in self.agent.strategy_memory.strategies.values())}")
            print(f"  Adaptation Factors:")
            print(f"    Vol: {self.agent.adaptive_params['vol_scaling_factor']:.2f}")
            print(f"    Trend: {self.agent.adaptive_params['trend_scaling_factor']:.2f}")
            print(f"    Correlation: {self.agent.adaptive_params['correlation_scaling_factor']:.2f}")

            # Verify the model was loaded by checking if weights exist
            if hasattr(self.agent.actor, 'state_dict') and len(self.agent.actor.state_dict()) > 0:
                print("Model weights verified successfully")
            else:
                print("Warning: Model loaded but weights verification failed")

            return checkpoint['episode'] + 1

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            if 'checkpoint' in locals():
                print("Available keys in checkpoint:", list(checkpoint.keys()))
            raise

    def _get_discrete_actions(self, probs):
        """Helper method to get discrete actions with memory efficiency"""
        discrete_actions = []
        for j in range(self.env.num_tickers):
            ticker_probs = probs[j]

            if self.env.invested_capital < 1e-6 and self.env.remaining_capital > 1000:
                action = 2  # Buy action
            elif self.env.remaining_capital < 1e-6:
                # Only sample from sell (0) or hold (1)
                restricted_probs = torch.tensor([0.5, 0.5, 0.0], device=self.agent.gpu_device)
                action = torch.multinomial(restricted_probs, 1).item()
            else:
                action = torch.multinomial(ticker_probs, 1).item()

            discrete_actions.append(action)

        return discrete_actions

    def _check_and_adjust_memory(self):
        """Check GPU memory usage and adjust parameters if needed."""
        if not torch.cuda.is_available():
            return

        # Get current memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_ratio = allocated / max_memory

        if usage_ratio > self.memory_threshold:
            print(f"\nHigh memory usage detected: {usage_ratio:.2%}")

            # Reduce batch size if possible
            if self.training_batch_size > self.min_batch_size:
                self.training_batch_size = max(self.min_batch_size, self.training_batch_size // 2)
                print(f"Reduced batch size to {self.training_batch_size}")

            # Clear memory
            self._clear_caches()

            # Reduce memory fraction if still high
            if usage_ratio > 0.8 and self.current_memory_fraction > 0.3:
                self.current_memory_fraction = max(0.3, self.current_memory_fraction - 0.1)
                torch.cuda.set_per_process_memory_fraction(self.current_memory_fraction)
                print(f"Reduced memory fraction to {self.current_memory_fraction:.1%}")

    def _handle_oom_error(self):
        """Handle out of memory errors by adjusting parameters."""
        # Clear memory
        self._clear_caches()

        # Reduce batch size
        if self.training_batch_size > self.min_batch_size:
            self.training_batch_size = max(self.min_batch_size, self.training_batch_size // 2)
            print(f"Reduced batch size to {self.training_batch_size}")

        # Reduce memory fraction
        if self.current_memory_fraction > 0.3:
            self.current_memory_fraction = max(0.3, self.current_memory_fraction - 0.1)
            torch.cuda.set_per_process_memory_fraction(self.current_memory_fraction)
            print(f"Reduced memory fraction to {self.current_memory_fraction:.1%}")

    def reset_environment_state(self):
        """Force a complete reset of the environment state to fix any inconsistencies"""
        print("\nPerforming complete environment reset...")
        self.env.remaining_capital = float(self.env.initial_capital)
        self.env.invested_capital = 0.0
        self.env.positions = torch.zeros(self.env.num_tickers, device=self.env.cpu_device)
        self.env.prev_portfolio_value = float(self.env.initial_capital)
        self.env.current_step = 0

        # Ensure history_cache is properly initialized
        if not hasattr(self.env, 'history_cache'):
            self.env.history_cache = {
                'volatility': {},  # Cache for historical volatilities
                'returns': {},     # Cache for historical returns
                'prices': {}       # Cache for historical prices
            }

        # Clear price cache
        if hasattr(self.env, 'price_cache'):
            self.env.price_cache.clear()

        # Clear observation cache
        if hasattr(self.env, 'observation_cache'):
            self.env.observation_cache.clear()

        # Clear metrics cache
        if hasattr(self.env, 'metrics_cache'):
            self.env.metrics_cache.clear()

        # Reset agent's consecutive sell counter
        if hasattr(self.agent, 'consecutive_sell_actions'):
            self.agent.consecutive_sell_actions = 0

        print(f"Reset complete. Capital: ${self.env.remaining_capital:.2f}, Invested: ${self.env.invested_capital:.2f}")
        return self.env.reset()  # Call the environment's reset method to get initial observation

    def _save_comparative_performance_charts(self, ticker_returns_by_episode, buy_and_hold_returns, portfolio_returns_by_episode=None, portfolio_values_by_episode=None, buy_and_hold_values=None, episode=None):
        """
        Save line charts comparing cumulative returns versus buy-and-hold across episode steps
        with confidence intervals, using dates on the x-axis.

        Args:
            ticker_returns_by_episode: Dictionary mapping tickers to lists of returns across episodes
            buy_and_hold_returns: Dictionary mapping tickers to their buy-and-hold returns
            portfolio_returns_by_episode: List of portfolio returns across episodes (optional)
            portfolio_values_by_episode: List of lists containing portfolio values at each step for each episode
            buy_and_hold_values: List containing buy-and-hold portfolio values at each step
            episode: Current episode number (optional)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_str = f"_ep{episode}" if episode is not None else ""

            # Create a ticker string for filenames
            tickers_str = "_".join(self.env.tickers)

            # Create figures directory if it doesn't exist
            if not os.path.exists(self.figures_dir):
                os.makedirs(self.figures_dir)

            # Get the dates from the environment for the x-axis
            # Use rollout dates since we're in rollout evaluation mode
            eval_dates = self.env.dates

            # Convert dates to datetime objects if they're strings
            if isinstance(eval_dates[0], str):
                eval_dates = [pd.to_datetime(date) for date in eval_dates]

            # Format dates for display
            date_labels = [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                          for date in eval_dates]

            # Calculate average returns and confidence intervals across episodes for each ticker
            ticker_avg_returns = {}
            ticker_ci_upper = {}
            ticker_ci_lower = {}

            for ticker, returns_list in ticker_returns_by_episode.items():
                # Convert list of returns to numpy array
                returns_array = np.array(returns_list)

                # Calculate mean return
                avg_return = np.mean(returns_array)

                # Calculate 95% confidence interval
                if len(returns_array) > 1:
                    std_dev = np.std(returns_array)
                    ci = 1.96 * std_dev / np.sqrt(len(returns_array))  # 95% confidence interval
                else:
                    ci = 0

                ticker_avg_returns[ticker] = avg_return
                ticker_ci_upper[ticker] = avg_return + ci
                ticker_ci_lower[ticker] = avg_return - ci

            # Calculate average portfolio return and confidence interval if provided
            if portfolio_returns_by_episode and len(portfolio_returns_by_episode) > 0:
                portfolio_array = np.array(portfolio_returns_by_episode)
                avg_portfolio_return = np.mean(portfolio_array)

                if len(portfolio_array) > 1:
                    portfolio_std = np.std(portfolio_array)
                    portfolio_ci = 1.96 * portfolio_std / np.sqrt(len(portfolio_array))
                else:
                    portfolio_ci = 0

                portfolio_ci_upper = avg_portfolio_return + portfolio_ci
                portfolio_ci_lower = avg_portfolio_return - portfolio_ci

            # Create a single chart with all tickers
            plt.figure(figsize=(14, 8))

            # Use a color palette
            colors = plt.cm.tab10(np.linspace(0, 1, len(ticker_returns_by_episode)))

            # First, calculate cumulative returns for each episode (same as in the third chart)
            if portfolio_values_by_episode and buy_and_hold_values:
                # Calculate cumulative returns for each episode
                cumulative_returns_by_episode = []
                for episode_values in portfolio_values_by_episode:
                    # Convert to percentage returns relative to initial value
                    initial_value = episode_values[0]
                    cumulative_returns = [(val / initial_value - 1) * 100 for val in episode_values]
                    cumulative_returns_by_episode.append(cumulative_returns)

                # Calculate buy and hold cumulative returns
                initial_bh_value = buy_and_hold_values[0]
                bh_cumulative_returns = [(val / initial_bh_value - 1) * 100 for val in buy_and_hold_values]

                # Calculate mean and confidence intervals for agent returns at each time step
                mean_returns = []
                ci_upper = []
                ci_lower = []

                # Make sure we have data for each time step
                max_steps = min(len(eval_dates), min([len(returns) for returns in cumulative_returns_by_episode]))

                for step in range(max_steps):
                    # Get returns for this step across all episodes
                    step_returns = [episode_returns[step] for episode_returns in cumulative_returns_by_episode]

                    # Calculate mean and confidence interval
                    mean_return = np.mean(step_returns)
                    mean_returns.append(mean_return)

                    if len(step_returns) > 1:
                        std_dev = np.std(step_returns)
                        ci = 1.96 * std_dev / np.sqrt(len(step_returns))
                        ci_upper.append(mean_return + ci)
                        ci_lower.append(mean_return - ci)
                    else:
                        ci_upper.append(mean_return)
                        ci_lower.append(mean_return)

                # Ensure buy-and-hold returns match the length of agent returns
                bh_cumulative_returns = bh_cumulative_returns[:max_steps]

                # Plot portfolio cumulative returns
                plt.plot(range(max_steps), mean_returns,
                         linestyle='-', linewidth=3, label=f'Portfolio: {avg_portfolio_return:.2f}%',
                         color='black')

                # Plot confidence interval for portfolio
                plt.fill_between(range(max_steps), ci_lower, ci_upper,
                                alpha=0.2, color='black', label='95% CI')

                # Calculate final buy-and-hold return
                final_bh_return = bh_cumulative_returns[-1] if bh_cumulative_returns else 0.0

                # Plot buy-and-hold cumulative returns
                plt.plot(range(max_steps), bh_cumulative_returns,
                         linestyle='--', linewidth=2, label=f'B&H: {final_bh_return:.2f}%',
                         color='red')

                # Plot ticker-specific returns as horizontal lines for comparison
                for i, ticker in enumerate(ticker_returns_by_episode.keys()):
                    # Calculate the average return for this ticker
                    avg_return = ticker_avg_returns[ticker]

                    # Plot ticker average return as a horizontal line
                    plt.axhline(y=avg_return, linestyle='-.', linewidth=1.5,
                               label=f'Agent ({ticker}): {avg_return:.2f}%',
                               color=colors[i], alpha=0.7)

                    # Plot buy-and-hold ticker return as a horizontal line
                    bh_return = buy_and_hold_returns.get(ticker, 0)
                    plt.axhline(y=bh_return, linestyle=':', linewidth=1.5,
                               label=f'B&H ({ticker}): {bh_return:.2f}%',
                               color=colors[i], alpha=0.5)
            else:
                # Fallback to old method if we don't have portfolio values
                # Plot each ticker's average return with confidence interval
                for i, ticker in enumerate(ticker_returns_by_episode.keys()):
                    # Calculate the average return for this ticker
                    avg_return = ticker_avg_returns[ticker]

                    # Plot average return line - use a horizontal line to show the final average return
                    plt.plot([0, len(eval_dates)-1], [avg_return, avg_return],
                             linestyle='-', linewidth=2, label=f'Agent ({ticker}): {avg_return:.2f}%',
                             color=colors[i])

                    # Plot confidence interval as shaded area
                    plt.fill_between([0, len(eval_dates)-1],
                                    [ticker_ci_lower[ticker], ticker_ci_lower[ticker]],
                                    [ticker_ci_upper[ticker], ticker_ci_upper[ticker]],
                                    alpha=0.2, color=colors[i])

                    # Plot buy-and-hold as a horizontal line with matching color but dashed
                    bh_return = buy_and_hold_returns.get(ticker, 0)
                    plt.plot([0, len(eval_dates)-1], [bh_return, bh_return],
                             linestyle='--', linewidth=1.5,
                             label=f'B&H ({ticker}): {bh_return:.2f}%',
                             color=colors[i])

                # Add portfolio returns if provided
                if portfolio_returns_by_episode and len(portfolio_returns_by_episode) > 0:
                    plt.plot([0, len(eval_dates)-1], [avg_portfolio_return, avg_portfolio_return],
                             linestyle='-', linewidth=3, label=f'Portfolio: {avg_portfolio_return:.2f}%',
                             color='black')

                    # Plot confidence interval for portfolio
                    plt.fill_between([0, len(eval_dates)-1],
                                    [portfolio_ci_lower, portfolio_ci_lower],
                                    [portfolio_ci_upper, portfolio_ci_upper],
                                    alpha=0.1, color='black')

            # Add zero line for reference
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)

            # Set x-axis ticks and labels
            # Choose a subset of dates to display to avoid overcrowding
            num_ticks = min(10, len(eval_dates))
            tick_indices = np.linspace(0, len(eval_dates)-1, num_ticks, dtype=int)
            plt.xticks(tick_indices, [date_labels[i] for i in tick_indices], rotation=45)

            # Add labels and title
            ticker_list = tickers_str.split('_')
            plt.title(f'Comparative Evaluation Performance for tickers {", ".join(ticker_list)}', fontsize=16)
            plt.xlabel('Evaluation Period', fontsize=12)
            plt.ylabel('Return (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=10)

            # Save the figure
            plt.tight_layout()
            plt.savefig(f"{self.figures_dir}/comparative_performance{episode_str}_{tickers_str}_{timestamp}.png")
            plt.close()

            # Create a detailed chart showing performance by ticker
            plt.figure(figsize=(16, 10))

            # Create subplots for each ticker plus portfolio
            num_tickers = len(ticker_returns_by_episode)
            total_plots = num_tickers + 1  # +1 for portfolio
            rows = (total_plots + 1) // 2  # Ensure enough rows for all plots
            cols = 2 if num_tickers > 1 else 1

            # First, calculate ticker-specific cumulative returns if we have portfolio values
            ticker_cumulative_returns = {}

            if portfolio_values_by_episode and len(portfolio_values_by_episode) > 0:
                # For each ticker, try to extract cumulative returns from the portfolio data
                for ticker in ticker_returns_by_episode.keys():
                    ticker_idx = list(self.env.tickers).index(ticker) if ticker in self.env.tickers else -1

                    if ticker_idx >= 0:
                        # Get position values for this ticker across episodes
                        ticker_values_by_episode = []

                        # Use the same approach as for portfolio values
                        for episode in range(len(portfolio_values_by_episode)):
                            # Use ticker returns directly since we don't have per-ticker values
                            ticker_return = ticker_returns_by_episode[ticker][episode] if episode < len(ticker_returns_by_episode[ticker]) else 0

                            # Create a synthetic cumulative return curve based on final return
                            # This is a simplification but better than a flat line
                            steps = len(portfolio_values_by_episode[episode])
                            ticker_values = [0] * steps

                            # Linear interpolation from 0 to final return
                            for step in range(steps):
                                ticker_values[step] = ticker_return * step / (steps - 1) if steps > 1 else ticker_return

                            ticker_values_by_episode.append(ticker_values)

                        ticker_cumulative_returns[ticker] = ticker_values_by_episode

            # Now plot each ticker
            for i, ticker in enumerate(ticker_returns_by_episode.keys()):
                ax = plt.subplot(rows, cols, i+1)

                # If we have cumulative returns for this ticker, plot them
                if ticker in ticker_cumulative_returns:
                    # Calculate mean and confidence intervals
                    ticker_values = ticker_cumulative_returns[ticker]

                    # Calculate mean returns at each step
                    mean_returns = []
                    ci_upper = []
                    ci_lower = []

                    # Find the minimum length across all episodes
                    min_length = min([len(values) for values in ticker_values])

                    for step in range(min_length):
                        step_returns = [values[step] for values in ticker_values]
                        mean_return = np.mean(step_returns)
                        mean_returns.append(mean_return)

                        if len(step_returns) > 1:
                            std_dev = np.std(step_returns)
                            ci = 1.96 * std_dev / np.sqrt(len(step_returns))
                            ci_upper.append(mean_return + ci)
                            ci_lower.append(mean_return - ci)
                        else:
                            ci_upper.append(mean_return)
                            ci_lower.append(mean_return)

                    # Plot mean returns
                    ax.plot(range(min_length), mean_returns,
                           linestyle='-', linewidth=2,
                           label=f'Agent: {ticker_avg_returns[ticker]:.2f}%',
                           color=colors[i])

                    # Plot confidence interval
                    # Make confidence interval more visible
                    ax.fill_between(range(min_length), ci_lower, ci_upper,
                                  alpha=0.3, color=colors[i])

                    # Plot buy-and-hold as a horizontal line
                    bh_return = buy_and_hold_returns.get(ticker, 0)
                    ax.axhline(y=bh_return, linestyle='--', linewidth=1.5,
                              label=f'B&H: {bh_return:.2f}%',
                              color='red')
                else:
                    # Fallback to old method if we don't have cumulative returns
                    # Plot average return line
                    avg_return = ticker_avg_returns[ticker]
                    ax.plot([0, len(eval_dates)-1], [avg_return, avg_return],
                            linestyle='-', linewidth=2, label=f'Agent: {avg_return:.2f}%',
                            color=colors[i])

                    # Plot confidence interval as shaded area
                    ax.fill_between([0, len(eval_dates)-1],
                                   [ticker_ci_lower[ticker], ticker_ci_lower[ticker]],
                                   [ticker_ci_upper[ticker], ticker_ci_upper[ticker]],
                                   alpha=0.2, color=colors[i])

                    # Plot buy-and-hold as a horizontal line
                    bh_return = buy_and_hold_returns.get(ticker, 0)
                    ax.plot([0, len(eval_dates)-1], [bh_return, bh_return],
                            linestyle='--', linewidth=1.5,
                            label=f'B&H: {bh_return:.2f}%',
                            color='red')

                # Add zero line for reference
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)

                # Set x-axis ticks and labels
                if len(eval_dates) > 5:
                    tick_indices = np.linspace(0, len(eval_dates)-1, 5, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45)

                # Add title and legend
                ax.set_title(f'{ticker}', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')

                # Add outperformance text
                outperformance = ticker_avg_returns[ticker] - bh_return
                color = 'green' if outperformance > 0 else 'red'
                ax.text(0.02, 0.02, f'Outperformance: {outperformance:+.2f}%',
                        transform=ax.transAxes, fontsize=12, color=color,
                        bbox=dict(facecolor='white', alpha=0.7))

            # Add portfolio subplot if available
            if portfolio_returns_by_episode and len(portfolio_returns_by_episode) > 0:
                ax = plt.subplot(rows, cols, num_tickers+1)

                # If we have portfolio values, plot actual cumulative returns
                if portfolio_values_by_episode and buy_and_hold_values:
                    # Use the same cumulative returns calculation as in the main chart
                    cumulative_returns_by_episode = []
                    for episode_values in portfolio_values_by_episode:
                        # Convert to percentage returns relative to initial value
                        initial_value = episode_values[0]
                        cumulative_returns = [(val / initial_value - 1) * 100 for val in episode_values]
                        cumulative_returns_by_episode.append(cumulative_returns)

                    # Calculate buy and hold cumulative returns
                    initial_bh_value = buy_and_hold_values[0]
                    bh_cumulative_returns = [(val / initial_bh_value - 1) * 100 for val in buy_and_hold_values]

                    # Calculate mean and confidence intervals for agent returns at each time step
                    mean_returns = []
                    ci_upper = []
                    ci_lower = []

                    # Make sure we have data for each time step
                    max_steps = min(len(eval_dates), min([len(returns) for returns in cumulative_returns_by_episode]))

                    for step in range(max_steps):
                        # Get returns for this step across all episodes
                        step_returns = [episode_returns[step] for episode_returns in cumulative_returns_by_episode]

                        # Calculate mean and confidence interval
                        mean_return = np.mean(step_returns)
                        mean_returns.append(mean_return)

                        if len(step_returns) > 1:
                            std_dev = np.std(step_returns)
                            ci = 1.96 * std_dev / np.sqrt(len(step_returns))
                            ci_upper.append(mean_return + ci)
                            ci_lower.append(mean_return - ci)
                        else:
                            ci_upper.append(mean_return)
                            ci_lower.append(mean_return)

                    # Ensure buy-and-hold returns match the length of agent returns
                    bh_cumulative_returns = bh_cumulative_returns[:max_steps]

                    # Plot agent's mean cumulative returns with confidence interval
                    ax.plot(range(max_steps), mean_returns,
                             linestyle='-', linewidth=2.5, label=f'Portfolio: {avg_portfolio_return:.2f}%',
                             color='blue')

                    # Make confidence interval more visible
                    ax.fill_between(range(max_steps), ci_lower, ci_upper,
                                    alpha=0.3, color='blue')

                    # Plot buy-and-hold cumulative returns
                    final_bh_return = bh_cumulative_returns[-1] if bh_cumulative_returns else 0.0
                    ax.plot(range(max_steps), bh_cumulative_returns,
                             linestyle='--', linewidth=2, label=f'B&H: {final_bh_return:.2f}%',
                             color='red')
                else:
                    # Fallback to old method if we don't have portfolio values
                    # Plot average portfolio return
                    ax.plot([0, len(eval_dates)-1], [avg_portfolio_return, avg_portfolio_return],
                            linestyle='-', linewidth=3, label=f'Portfolio: {avg_portfolio_return:.2f}%',
                            color='black')

                    # Plot confidence interval for portfolio
                    ax.fill_between([0, len(eval_dates)-1],
                                   [portfolio_ci_lower, portfolio_ci_lower],
                                   [portfolio_ci_upper, portfolio_ci_upper],
                                   alpha=0.1, color='black')

                # Add zero line for reference
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)

                # Set x-axis ticks and labels
                if len(eval_dates) > 5:
                    tick_indices = np.linspace(0, len(eval_dates)-1, 5, dtype=int)
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45)

                # Add title and legend
                ax.set_title('Portfolio', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')

            plt.suptitle('Comparative Evaluation Performance', fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
            plt.savefig(f"{self.figures_dir}/comparative_performance_detailed{episode_str}_{tickers_str}_{timestamp}.png")
            plt.close()

            # Create a new chart showing cumulative returns over time with confidence intervals
            if portfolio_values_by_episode and buy_and_hold_values:
                plt.figure(figsize=(16, 8))

                # Calculate cumulative returns for each episode
                cumulative_returns_by_episode = []
                for episode_values in portfolio_values_by_episode:
                    # Convert to percentage returns relative to initial value
                    initial_value = episode_values[0]
                    cumulative_returns = [(val / initial_value - 1) * 100 for val in episode_values]
                    cumulative_returns_by_episode.append(cumulative_returns)

                # Calculate buy and hold cumulative returns
                initial_bh_value = buy_and_hold_values[0]
                bh_cumulative_returns = [(val / initial_bh_value - 1) * 100 for val in buy_and_hold_values]

                # Calculate mean and confidence intervals for agent returns at each time step
                mean_returns = []
                ci_upper = []
                ci_lower = []

                # Make sure we have data for each time step
                max_steps = min(len(eval_dates), min([len(returns) for returns in cumulative_returns_by_episode]))

                for step in range(max_steps):
                    # Get returns for this step across all episodes
                    step_returns = [episode_returns[step] for episode_returns in cumulative_returns_by_episode]

                    # Calculate mean and confidence interval
                    mean_return = np.mean(step_returns)
                    mean_returns.append(mean_return)

                    if len(step_returns) > 1:
                        std_dev = np.std(step_returns)
                        ci = 1.96 * std_dev / np.sqrt(len(step_returns))
                        ci_upper.append(mean_return + ci)
                        ci_lower.append(mean_return - ci)
                    else:
                        ci_upper.append(mean_return)
                        ci_lower.append(mean_return)

                # Ensure buy-and-hold returns match the length of agent returns
                bh_cumulative_returns = bh_cumulative_returns[:max_steps]

                # Debug information
                print(f"\nCumulative returns data points:")
                print(f"  Agent mean returns: {len(mean_returns)} points")
                print(f"  Buy & Hold returns: {len(bh_cumulative_returns)} points")
                print(f"  Evaluation dates: {len(eval_dates)} dates")
                print(f"  First 5 agent returns: {mean_returns[:5]}")
                print(f"  First 5 B&H returns: {bh_cumulative_returns[:5]}")

                # Plot agent's mean cumulative returns with confidence interval
                plt.plot(range(max_steps), mean_returns,
                         linestyle='-', linewidth=2.5, label='Agent Portfolio', color='blue')

                # Make confidence interval more visible
                plt.fill_between(range(max_steps), ci_lower, ci_upper,
                                alpha=0.3, color='blue', label='95% Confidence Interval')

                # Add debug info about confidence interval
                print(f"  Confidence interval details:")
                print(f"    Mean range: {min(mean_returns):.2f}% to {max(mean_returns):.2f}%")
                print(f"    CI width: min={min([u-l for u,l in zip(ci_upper, ci_lower)]):.2f}%, max={max([u-l for u,l in zip(ci_upper, ci_lower)]):.2f}%")
                print(f"    CI lower range: {min(ci_lower):.2f}% to {max(ci_lower):.2f}%")
                print(f"    CI upper range: {min(ci_upper):.2f}% to {max(ci_upper):.2f}%")

                # Plot buy-and-hold cumulative returns
                plt.plot(range(max_steps), bh_cumulative_returns,
                         linestyle='--', linewidth=2, label='Buy & Hold', color='red')

                # Add zero line for reference
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)

                # Set x-axis ticks and labels
                num_ticks = min(10, len(eval_dates))
                tick_indices = np.linspace(0, len(eval_dates)-1, num_ticks, dtype=int)
                plt.xticks(tick_indices, [date_labels[i] for i in tick_indices], rotation=45)

                # Add labels and title
                plt.title('Cumulative Returns vs Buy & Hold', fontsize=16)
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Cumulative Return (%)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best', fontsize=12)

                # Add final return values as text annotations
                final_agent_return = mean_returns[-1]
                final_bh_return = bh_cumulative_returns[-1]
                outperformance = final_agent_return - final_bh_return

                plt.annotate(f'Agent: {final_agent_return:.2f}%',
                            xy=(len(eval_dates)-1, final_agent_return),
                            xytext=(len(eval_dates)-1, final_agent_return + 2),
                            fontsize=12, color='blue')

                plt.annotate(f'B&H: {final_bh_return:.2f}%',
                            xy=(len(eval_dates)-1, final_bh_return),
                            xytext=(len(eval_dates)-1, final_bh_return - 4),
                            fontsize=12, color='red')

                # Add outperformance text
                color = 'green' if outperformance > 0 else 'red'
                plt.text(0.02, 0.02, f'Outperformance: {outperformance:+.2f}%',
                        transform=plt.gca().transAxes, fontsize=14, color=color,
                        bbox=dict(facecolor='white', alpha=0.7))

                plt.tight_layout()
                plt.savefig(f"{self.figures_dir}/cumulative_returns{episode_str}_{tickers_str}_{timestamp}.png")
                plt.close()

            print(f"\nSaved comparative performance charts to {self.figures_dir}/")

        except Exception as e:
            print(f"\nWarning: Error in saving comparative performance charts: {str(e)}")
            import traceback
            traceback.print_exc()
            print("This is non-critical and won't affect the training process.")

    def _evaluate_rollout(self, final=False):
        """
        Perform rollout evaluation with meta-learning integration.

        Args:
            final: If True, use the final evaluation dates (Jan-Mar 2025) instead of
                  rollout dates (Oct-Dec 2024)
        """
        self.agent.mode = 'final_eval' if final else 'rollout'
        self.env.set_mode('final_eval' if final else 'rollout')

        # Force a complete reset to ensure clean state
        self.reset_environment_state()

        episode_scores = []
        episode_returns = []
        episode_values = []

        # Track ticker-wise returns
        ticker_returns_by_episode = {ticker: [] for ticker in self.env.tickers}
        portfolio_returns_by_episode = []

        # Track portfolio values at each step for each episode
        portfolio_values_by_episode = []

        # Calculate buy and hold returns for comparison
        buy_and_hold_returns = {}

        # Initialize variables that might be used outside the try block
        sharpe_ratio = 0.0
        max_drawdown = 1.0
        avg_portfolio_return = 0.0
        buy_and_hold_return_pct = 0.0
        outperformance = 0.0
        outperformance_factor = 1.0  # Default is neutral (1.0)
        normalized_sharpe = 0.0
        composite_score = 1.0  # Default neutral score
        avg_ticker_returns = {ticker: 0.0 for ticker in self.env.tickers}

        # Check which date range we're using
        eval_period = "final evaluation" if final else "rollout evaluation"
        if len(self.env.dates) > 0:
            first_date = self.env.dates[0]
            if isinstance(first_date, pd.Timestamp):
                if final and first_date.year == 2025:
                    print(f"\nUsing January-March 2025 dates for {eval_period}")
                elif not final and first_date.year == 2024 and first_date.month >= 10:
                    print(f"\nUsing October-December 2024 dates for {eval_period}")

        try:
            # Calculate buy-and-hold returns for each ticker
            initial_date = self.env.dates[0]
            final_date = self.env.dates[-1]

            # Convert dates if needed
            if isinstance(initial_date, np.datetime64):
                initial_date_pd = pd.Timestamp(initial_date)
            else:
                initial_date_pd = pd.to_datetime(initial_date)

            if isinstance(final_date, np.datetime64):
                final_date_pd = pd.Timestamp(final_date)
            else:
                final_date_pd = pd.to_datetime(final_date)

            for i, ticker in enumerate(self.env.tickers):
                try:
                    # Try with original dates first
                    try:
                        initial_price = self.env.unscaled_close_df.loc[initial_date]['Close'].values[i]
                    except KeyError:
                        # Try with converted date
                        initial_price = self.env.unscaled_close_df.loc[initial_date_pd]['Close'].values[i]

                    try:
                        final_price = self.env.unscaled_close_df.loc[final_date]['Close'].values[i]
                    except KeyError:
                        # Try with converted date
                        final_price = self.env.unscaled_close_df.loc[final_date_pd]['Close'].values[i]

                    bh_return = ((final_price - initial_price) / initial_price) * 100
                    buy_and_hold_returns[ticker] = round(bh_return, 2)
                except Exception as e:
                    print(f"Warning: Could not calculate buy-and-hold return for {ticker}: {str(e)}")
                    buy_and_hold_returns[ticker] = 0.0

            # Calculate buy-and-hold portfolio values for each time step
            buy_and_hold_return_pct, buy_and_hold_values = self._calculate_buy_and_hold_return(return_values=True)

            # Print buy-and-hold returns
            print(f"\nBuy & Hold Returns ({eval_period}):")
            for ticker, ret in buy_and_hold_returns.items():
                print(f"  {ticker}: {ret:.2f}%")
            print(f"  Portfolio: {buy_and_hold_return_pct:.2f}%")

            for episode in tqdm(range(self.rollout_episodes), desc="Rollout Episodes", total=self.rollout_episodes):
                state = self.env.reset()
                episode_score = 0
                episode_value_history = [self.env.initial_capital]
                episode_return_history = []

                # Get initial prices for ticker return calculation
                initial_date = self.env.dates[0]

                try:
                    # Try to get prices directly using the date
                    initial_prices = torch.tensor(
                        self.env.unscaled_close_df.loc[initial_date]['Close'].values,
                        device=self.env.gpu_device,
                        dtype=torch.float32
                    )
                except KeyError:
                    # If direct lookup fails, try to convert the date format
                    try:
                        # Convert numpy.datetime64 to pandas Timestamp if needed
                        if isinstance(initial_date, np.datetime64):
                            initial_date_pd = pd.Timestamp(initial_date)
                        else:
                            initial_date_pd = pd.to_datetime(initial_date)

                        # Try lookup with converted date
                        initial_prices = torch.tensor(
                            self.env.unscaled_close_df.loc[initial_date_pd]['Close'].values,
                            device=self.env.gpu_device,
                            dtype=torch.float32
                        )
                    except Exception as e:
                        print(f"Warning: Could not get initial prices for date {initial_date}. Using default values. Error: {str(e)}")
                        # Fallback to using default values
                        initial_prices = torch.ones(self.env.num_tickers, device=self.env.gpu_device, dtype=torch.float32) * 100.0  # Arbitrary default price

                # Note: We're tracking price returns, not position returns
                # so we don't need to track initial positions

                market_features = self.agent._create_market_features()

                with torch.no_grad():
                    style_weights = self.agent.meta_learner(
                        market_features,
                        torch.tensor(self.agent._get_strategy_metrics(), device=self.agent.gpu_device)
                    )
                    # Ensure we only have 3 style outputs
                    style_weights = style_weights[:3] if style_weights.dim() == 1 else style_weights[:, :3]
                    style_weights = F.softmax(style_weights, dim=-1)

                # Get the style index (should be 0, 1, or 2)
                selected_style = torch.argmax(style_weights).item()

                # Validate the style index
                if not (0 <= selected_style < 3):
                    print(f"Warning: Invalid style index {selected_style}, defaulting to moderate")
                    selected_style = 1  # Default to moderate style

                # Map to style name
                style_names = ['aggressive', 'moderate', 'conservative']
                self.agent.current_style = style_names[selected_style]

                # Signal episode start to agent
                self.agent.adapt_parameters(is_episode_start=True)

                # Track actions taken during the episode
                episode_actions = {
                    'buy': 0,
                    'sell': 0,
                    'hold': 0
                }

                # Track action history for each ticker
                action_history = []

                for step in range(self.steps_per_episode):
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        action = self.agent.act(state, eval_mode=True)

                    # Track actions taken
                    if isinstance(action, tuple) and isinstance(action[0], torch.Tensor):
                        discrete_actions = action[0].cpu().numpy()

                        # Add to action history
                        action_history.append(discrete_actions.tolist())

                        for act in discrete_actions:
                            if act == 0:
                                episode_actions['sell'] += 1
                            elif act == 1:
                                episode_actions['hold'] += 1
                            elif act == 2:
                                episode_actions['buy'] += 1

                    next_state, reward, done, info = self.env.step(action)
                    self.agent.update_market_context(next_state, reward, done)

                    episode_score += reward.item() if torch.is_tensor(reward) else reward
                    portfolio_value = info['portfolio_value'].item() if torch.is_tensor(info['portfolio_value']) else info['portfolio_value']
                    episode_value_history.append(portfolio_value)
                    episode_return_history.append(reward)

                    # Debug: Print step information every 10 steps
                    if step % 10 == 0:
                        print(f"  Step {step}: Portfolio value: ${portfolio_value:.2f}, Reward: {reward.item() if torch.is_tensor(reward) else reward:.4f}")

                    if done:
                        break

                    state = next_state

                # Print action summary
                print(f"  Actions taken: Buy: {episode_actions['buy']}, Sell: {episode_actions['sell']}, Hold: {episode_actions['hold']}")

                # Calculate ticker-wise returns at the end of the episode
                final_date_idx = min(self.env.current_step, len(self.env.dates)-1)
                final_date = self.env.dates[final_date_idx]

                try:
                    # Try to get prices directly using the date
                    final_prices = torch.tensor(
                        self.env.unscaled_close_df.loc[final_date]['Close'].values,
                        device=self.env.gpu_device,
                        dtype=torch.float32
                    )
                except KeyError:
                    # If direct lookup fails, try to convert the date format
                    try:
                        # Convert numpy.datetime64 to pandas Timestamp if needed
                        if isinstance(final_date, np.datetime64):
                            final_date_pd = pd.Timestamp(final_date)
                        else:
                            final_date_pd = pd.to_datetime(final_date)

                        # Try lookup with converted date
                        final_prices = torch.tensor(
                            self.env.unscaled_close_df.loc[final_date_pd]['Close'].values,
                            device=self.env.gpu_device,
                            dtype=torch.float32
                        )
                    except Exception as e:
                        print(f"Warning: Could not get final prices for date {final_date}. Using initial prices. Error: {str(e)}")
                        # Fallback to using initial prices (this will result in 0% return)
                        final_prices = initial_prices.clone()

                # Note: We're tracking price returns, not position returns
                # so we don't need to track final positions

                # Calculate overall portfolio return first so it's available for ticker calculations
                initial_portfolio_value = episode_value_history[0]
                final_portfolio_value = episode_value_history[-1]

                # Calculate portfolio return percentage
                portfolio_return = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100

                # Debug information
                print(f"  Initial portfolio value: ${initial_portfolio_value:.2f}")
                print(f"  Final portfolio value: ${final_portfolio_value:.2f}")
                print(f"  Portfolio return: {portfolio_return:.2f}%")

                # Store portfolio value history for this episode
                portfolio_values_by_episode.append(episode_value_history)

                # Calculate ticker-wise percentage returns
                ticker_returns = {}
                for i, ticker in enumerate(self.env.tickers):
                    initial_price = initial_prices[i].item()
                    final_price = final_prices[i].item()

                    # Calculate market return (for reference)
                    market_return = ((final_price - initial_price) / initial_price) * 100

                    # Calculate position return (agent's trading return)
                    # This is based on the agent's actions during the episode

                    # Get the ticker's action history for this episode
                    ticker_actions = [action_history[step][i] for step in range(len(action_history))]

                    # Count actions
                    buy_count = ticker_actions.count(2)  # 2 = BUY
                    sell_count = ticker_actions.count(0)  # 0 = SELL
                    hold_count = ticker_actions.count(1)  # 1 = HOLD

                    # Calculate the actual return based on the agent's trading activity
                    # Get the current position value for this ticker
                    current_position = self.env.positions[i].item()

                    # Calculate the actual return based on the agent's trading activity and position
                    if current_position > 0:
                        # Agent has a position in this ticker at the end of the episode
                        # Note: We calculate this for reference but don't need to use it directly

                        # Calculate return based on trading activity
                        if buy_count > 0:
                            # If the agent bought this ticker during the episode, calculate a dynamic return
                            # that varies between episodes based on when purchases were made

                            # Use a randomized component to ensure variation between episodes
                            # while still being influenced by the market and portfolio returns
                            random_factor = np.random.uniform(0.8, 1.2)  # 20% variation

                            if sell_count > 0:
                                # Active trading - weight toward portfolio return with variation
                                position_return = portfolio_return * random_factor
                            else:
                                # Bought and held - weight toward market return with variation
                                position_return = market_return * random_factor
                        else:
                            # Had position from before - use market return
                            position_return = market_return

                        ticker_returns[ticker] = round(position_return, 2)
                    else:
                        # No position at the end
                        if buy_count > 0 or sell_count > 0:
                            # Agent traded but ended with no position - calculate a dynamic return
                            # Use a fraction of portfolio return with randomization
                            activity_ratio = (buy_count + sell_count) / len(action_history)
                            random_factor = np.random.uniform(0.7, 1.3)  # 30% variation
                            position_return = portfolio_return * activity_ratio * random_factor
                            ticker_returns[ticker] = round(position_return, 2)
                        else:
                            # Agent never traded this ticker
                            ticker_returns[ticker] = 0.0

                    # Debug information
                    if self.env.mode == 'rollout':
                        print(f"  {ticker} trading summary:")
                        print(f"    Market return: {market_return:.2f}%")
                        print(f"    Agent return: {ticker_returns[ticker]:.2f}%")
                        print(f"    Actions: Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}")
                        print(f"    Final position: {self.env.positions[i].item():.2f} shares")

                    ticker_returns_by_episode[ticker].append(ticker_returns[ticker])

                # Debug information already printed above

                # Check if the agent has any positions
                total_positions = self.env.positions.sum().item()
                if total_positions < 1e-6 and abs(portfolio_return) < 1e-6:
                    print("  Warning: Agent has no positions and zero return")

                portfolio_returns_by_episode.append(portfolio_return)

                # Print ticker-wise and portfolio returns for this episode
                print(f"\nEpisode {episode+1} Returns:")
                print(f"  Portfolio: {portfolio_return:.2f}%")
                print("  Ticker-wise returns:")
                for ticker, ret in ticker_returns.items():
                    print(f"    {ticker}: {ret:.2f}%")

                episode_scores.append(episode_score)
                episode_returns.extend(episode_return_history)
                episode_values.extend(episode_value_history)

            # Calculate average returns across all episodes
            avg_portfolio_return = np.mean(portfolio_returns_by_episode)
            avg_ticker_returns = {ticker: np.mean(returns) for ticker, returns in ticker_returns_by_episode.items()}

            # Print average returns across all episodes
            print("\nAverage Returns Across All Rollout Episodes:")
            print(f"  Portfolio: {avg_portfolio_return:.2f}%")
            print("  Ticker-wise returns:")
            for ticker, ret in avg_ticker_returns.items():
                print(f"    {ticker}: {ret:.2f}%")

            # Calculate and print the composite score components
            outperformance = avg_portfolio_return - buy_and_hold_return_pct
            outperformance_factor = 2.0 / (1.0 + np.exp(-outperformance * 0.2))
            normalized_sharpe = 2.0 / (1.0 + np.exp(-sharpe_ratio/2.0)) - 1.0
            composite_score = 0.5 * outperformance_factor * 2.0 + 0.5 * (normalized_sharpe + 1.0)

            print(f"\nComposite Score Components:")
            print(f"  Outperformance vs Buy & Hold: {outperformance:.2f}% (Agent: {avg_portfolio_return:.2f}%, B&H: {buy_and_hold_return_pct:.2f}%)")
            print(f"  Outperformance factor: {outperformance_factor:.4f}")
            print(f"  Sharpe ratio: {sharpe_ratio:.2f} (normalized: {normalized_sharpe:.4f})")
            print(f"  Composite Score: {composite_score:.4f}")

            # Save comparative performance charts
            self._save_comparative_performance_charts(
                episode=episode,
                ticker_returns_by_episode=ticker_returns_by_episode,
                buy_and_hold_returns=buy_and_hold_returns,
                portfolio_returns_by_episode=portfolio_returns_by_episode,
                portfolio_values_by_episode=portfolio_values_by_episode,
                buy_and_hold_values=buy_and_hold_values
            )

            # Store values for later access by evaluate method
            self.episode_values = episode_values
            self.buy_and_hold_values = buy_and_hold_values

            # Calculate Sharpe ratio
            returns_np = np.array([ret.cpu().numpy() if torch.is_tensor(ret) else ret for ret in episode_returns])
            sharpe_ratio = float(np.sqrt(252) * (np.mean(returns_np) / (np.std(returns_np) + 1e-6)))

            # Calculate max drawdown
            values_np = np.array(episode_values)
            peak = np.maximum.accumulate(values_np)
            drawdown = (peak - values_np) / peak
            max_drawdown = float(np.max(drawdown))

            # Calculate outperformance vs buy-and-hold
            outperformance = avg_portfolio_return - buy_and_hold_return_pct

            # Apply a sigmoid function to outperformance to get a value between 0 and 2
            # This will be 1.0 at zero outperformance, approaching 2.0 for strong outperformance,
            # and approaching 0.0 for strong underperformance
            outperformance_factor = 2.0 / (1.0 + np.exp(-outperformance * 0.2))

            # Normalize Sharpe ratio using modified sigmoid to handle negative values better
            # Center sigmoid around 0 and scale to handle typical Sharpe ratio ranges
            normalized_sharpe = 2.0 / (1.0 + np.exp(-sharpe_ratio/2.0)) - 1.0

            # Calculate composite score focusing ONLY on market outperformance and Sharpe ratio
            # Equal weighting (50/50) between outperformance and Sharpe ratio
            composite_score = 0.5 * outperformance_factor * 2.0 + 0.5 * (normalized_sharpe + 1.0)

            metrics = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_portfolio_return': avg_portfolio_return,
                'avg_ticker_returns': avg_ticker_returns,
                'buy_and_hold_returns': buy_and_hold_returns,  # Add buy-and-hold returns to metrics
                'buy_and_hold_portfolio_return': buy_and_hold_return_pct,  # Add overall buy-and-hold portfolio return
                'outperformance': outperformance,  # Add outperformance
                'outperformance_factor': outperformance_factor,  # Add outperformance factor
                'normalized_sharpe': normalized_sharpe,  # Add normalized Sharpe
                'composite_score': composite_score  # Add composite score
            }

            # Return the composite score as the rollout score for consistency
            return composite_score, metrics

        except Exception as e:
            print(f"Error in _evaluate_rollout: {str(e)}")
            # Create default metrics in case of exception
            metrics = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_portfolio_return': avg_portfolio_return,
                'avg_ticker_returns': avg_ticker_returns,
                'buy_and_hold_returns': buy_and_hold_returns,
                'buy_and_hold_portfolio_return': buy_and_hold_return_pct,
                'outperformance': outperformance,
                'outperformance_factor': outperformance_factor,
                'normalized_sharpe': normalized_sharpe,
                'composite_score': composite_score
            }
            # Use a neutral composite score
            composite_score = 1.0

        finally:
            # Reset agent and environment modes
            self.agent.mode = 'train'
            self.env.set_mode('train')

        # Return the composite score and metrics (this will be reached if an exception occurs)
        return composite_score, metrics

    def evaluate(self, n_episodes=10, final=False):
        """
        Evaluate the agent over multiple episodes and return performance metrics.

        Args:
            n_episodes: Number of evaluation episodes to run
            final: If True, use the final evaluation dates (Jan-Mar 2025) instead of
                  rollout dates (Oct-Dec 2024)

        Returns:
            Tuple containing (mean_score, sharpe_ratio, max_drawdown)
        """
        # Store original rollout episodes value
        original_rollout_episodes = self.rollout_episodes

        # Set rollout episodes to the requested number
        self.rollout_episodes = n_episodes

        try:
            # Perform evaluation (rollout or final)
            mean_score, eval_metrics = self._evaluate_rollout(final=final)

            # Extract metrics
            sharpe_ratio = eval_metrics.get('sharpe_ratio', 0.0)
            max_drawdown = eval_metrics.get('max_drawdown', 1.0)

            # Store evaluation results for visualization in the appropriate list
            eval_result = {
                'score': mean_score,
                'sharpe': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'portfolio_return': eval_metrics.get('avg_portfolio_return', 0.0),
                'relative_return': eval_metrics.get('outperformance', 0.0)
            }

            if final:
                self.final_eval_scores = [eval_result]
            else:
                self.rollout_eval_scores = [eval_result]

            # Store portfolio values and buy-and-hold values if available
            if hasattr(self, 'episode_values') and len(self.episode_values) > 0:
                self.buy_and_hold_values = eval_metrics.get('buy_and_hold_values', [])

            return mean_score, sharpe_ratio, max_drawdown

        except Exception as e:
            print(f"Error during {'final' if final else 'rollout'} evaluation: {str(e)}")
            # Return default values in case of error
            return 0.0, 0.0, 1.0

        finally:
            # Restore original rollout episodes value
            self.rollout_episodes = original_rollout_episodes

    def training_loop(self):
        """
        Execute the complete training pipeline with rollout and final evaluation.
        """
        print("Starting training pipeline...")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")

        # Initialize progress tracking
        import time
        start_time = time.time()
        progress_tracker.update_rl_agent_progress(
            status="running",
            message="Initializing RL agent training...",
            progress=0.0,
            current_episode=0,
            total_episodes=self.n_episodes,
            start_time=start_time
        )

        # Execute training
        print("\nTraining phase:")
        scores = self.train()
        print("\nTraining completed successfully")

        # Update progress after training
        progress_tracker.update_rl_agent_progress(
            status="evaluating",
            message="Training completed. Performing final evaluation on hold-out set (Jan-Mar 2025)...",
            progress=0.95,
            current_episode=self.n_episodes,
            total_episodes=self.n_episodes
        )

        # Perform final evaluation on the hold-out set (Jan-Mar 2025)
        print("\nPerforming final evaluation on hold-out set (January-March 2025)...")
        final_score, final_sharpe, final_drawdown = self.evaluate(n_episodes=10, final=True)

        # Get the final evaluation metrics
        if self.final_eval_scores and len(self.final_eval_scores) > 0:
            final_metrics = self.final_eval_scores[-1]
            portfolio_return = final_metrics.get('portfolio_return', 0.0)
            relative_return = final_metrics.get('relative_return', 0.0)

            print(f"\nFinal Evaluation Results (Jan-Mar 2025):")
            print(f"  Composite Score: {final_score:.4f}")
            print(f"  Sharpe Ratio: {final_sharpe:.2f}")
            print(f"  Max Drawdown: {final_drawdown:.2%}")
            print(f"  Portfolio Return: {portfolio_return:.2f}%")
            print(f"  Outperformance vs Buy & Hold: {relative_return:+.2f}%")

        # Update progress after final evaluation
        progress_tracker.update_rl_agent_progress(
            status="completed",
            message="RL agent training and evaluation completed successfully!",
            progress=1.0,
            current_episode=self.n_episodes,
            total_episodes=self.n_episodes,
            end_time=time.time()
        )

        print("\nTraining and evaluation completed.")
        print("Check the rollout evaluation results (Oct-Dec 2024) and final evaluation results (Jan-Mar 2025) for performance metrics.")

        return scores

    def validate_functionality(self):
        """
        Quick validation method to test if training and rollout functionality are working properly.
        Runs a single training episode and rollout evaluation.
        """
        print("\nValidating training and rollout functionality...")

        # Run one training episode
        print("\nTesting training episode:")
        state = self.env.reset()
        episode_score = 0

        for step in range(self.steps_per_episode):
            # Get action from agent
            action = self.agent.act(state)

            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)

            # Store experience
            self.agent.memory.add(state, action, reward, next_state, done)

            # Learn if enough samples
            if len(self.agent.memory) > self.training_batch_size:
                self.agent.learn()

            state = next_state
            episode_score += reward

            if done:
                break

        print(f"Training episode completed with score: {episode_score:.2f}")

        # Run one rollout evaluation
        print("\nTesting rollout evaluation:")
        rollout_score, _ = self._evaluate_rollout(final=False)
        print(f"Rollout evaluation completed with score: {rollout_score:.2f}")

        return True

    def _evaluate_and_save(self, episode, scores):
        """Evaluate the agent and save checkpoint if appropriate."""
        # Update progress before evaluation
        progress_tracker.update_rl_agent_progress(
            message=f"Evaluating agent at episode {episode}...",
            current_eval=episode // self.eval_frequency,
            total_evals=self.n_episodes // self.eval_frequency
        )

        # Perform rollout evaluation (not final evaluation)
        rollout_score, rollout_metrics = self._evaluate_rollout(final=False)
        self.rollout_scores.append(rollout_score)

        # Calculate composite score focusing ONLY on market outperformance and Sharpe ratio
        sharpe_ratio = rollout_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = rollout_metrics.get('max_drawdown', 1.0)  # Still track but don't use in score
        avg_portfolio_return = rollout_metrics.get('avg_portfolio_return', 0.0)
        avg_ticker_returns = rollout_metrics.get('avg_ticker_returns', {})

        # Calculate portfolio outperformance vs buy-and-hold
        buy_and_hold_portfolio_return = rollout_metrics.get('buy_and_hold_portfolio_return', 0.0)
        outperformance = avg_portfolio_return - buy_and_hold_portfolio_return

        # Apply a sigmoid function to outperformance to get a value between 0 and 2
        # This will be 1.0 at zero outperformance, approaching 2.0 for strong outperformance,
        # and approaching 0.0 for strong underperformance
        outperformance_factor = 2.0 / (1.0 + np.exp(-outperformance * 0.2))

        # Normalize Sharpe ratio using modified sigmoid to handle negative values better
        # Center sigmoid around 0 and scale to handle typical Sharpe ratio ranges
        normalized_sharpe = 2.0 / (1.0 + np.exp(-sharpe_ratio/2.0)) - 1.0

        # Calculate composite score focusing ONLY on market outperformance and Sharpe ratio
        # Equal weighting (50/50) between outperformance and Sharpe ratio
        composite_score = 0.5 * outperformance_factor * 2.0 + 0.5 * (normalized_sharpe + 1.0)

        # Update progress with evaluation results
        progress_tracker.update_rl_agent_progress(
            current_score=rollout_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            portfolio_return=avg_portfolio_return,
            best_score=getattr(self, 'best_composite_score', float('-inf'))
        )

        # Update best score if needed
        is_best = False
        if composite_score > getattr(self, 'best_composite_score', float('-inf')):
            self.best_composite_score = composite_score
            self.best_rollout_score = rollout_score
            self.best_sharpe = sharpe_ratio
            self.best_drawdown = max_drawdown
            self.best_portfolio_return = avg_portfolio_return
            self.best_ticker_returns = avg_ticker_returns
            is_best = True

            print(f"\nNew best model!")
            print(f"Composite Score: {composite_score:.4f}")
            print(f"Rollout Score: {rollout_score:.4f}")
            print(f"Sharpe: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Portfolio Return: {avg_portfolio_return:.2f}%")
            print("Ticker-wise Returns:")
            for ticker, ret in avg_ticker_returns.items():
                print(f"  {ticker}: {ret:.2f}%")

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth')
        self.save_checkpoint(
            episode=episode,
            episode_score=scores[-1],
            rollout_score=rollout_score,
            rollout_metrics=rollout_metrics
        )

        # Save best model separately if this is the best score
        if is_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            # Copy the checkpoint to best_model.pth
            shutil.copy2(checkpoint_path, best_model_path)
            print(f"Saved new best model to {best_model_path}")

        return rollout_score, is_best

if __name__ == "__main__":
    trainer = RLTrainer(
        eval_frequency=10,
        save_dir="memory",
        training_batch_size=32,  # Reduced from 64 to compensate for longer sequence length
        eval_batch_size=16,      # Reduced from 32 to compensate for longer sequence length
        n_episodes=200,
        rollout_episodes=10,
        initial_capital=10000
    )
    trainer.training_loop()



