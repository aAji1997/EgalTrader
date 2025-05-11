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
                 n_episodes: int = 1000,
                 num_workers: int = 4,
                 initial_capital: int = 10000):  # Add initial_capital parameter
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

        # Save initial capital
        self.initial_capital = initial_capital

        # Initialize environment and agent with memory optimization
        try:
            self.env = PortfolioEnv(initial_capital=self.initial_capital)
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
                self.env = PortfolioEnv(initial_capital=self.initial_capital)
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

        # Modify evaluation metrics to separate rollout and final evaluation
        self.final_eval_scores = []
        self.rollout_eval_scores = []




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
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        action = self.agent.act(state)

                    # Take action in environment
                    next_state, reward, done, info = self.env.step(action)

                    # Update market context and store experience
                    self.agent.update_market_context(next_state, reward, done)
                    self.agent.memory.add(state, action, reward, next_state, done)

                    # Learn if enough samples are available
                    if len(self.agent.memory) > self.training_batch_size:
                        try:
                            with torch.cuda.amp.autocast(enabled=self.use_amp):
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

    def evaluate(self, n_episodes: int = 50):
        """Evaluate with meta-learning integration and comprehensive performance tracking."""
        # Clear caches before evaluation
        self._clear_caches()

        # Load the best model if it exists
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("\nLoading best model for evaluation...")
            try:
                checkpoint = torch.load(best_model_path)
                self.agent.actor.load_state_dict(checkpoint['agent_state']['actor_state_dict'])
                self.agent.critic.load_state_dict(checkpoint['agent_state']['critic_state_dict'])
                self.agent.meta_learner.load_state_dict(checkpoint['agent_state']['meta_learner_state_dict'])
                self.agent.market_encoder.load_state_dict(checkpoint['agent_state']['market_encoder_state_dict'])
                print("Successfully loaded best model")
            except Exception as e:
                print(f"Warning: Could not load best model: {str(e)}")
                print("Proceeding with current model")
        else:
            print("\nNo best model checkpoint found. Using current model.")

        self.agent.mode = 'eval'
        self.env.set_mode('eval')
        eval_scores = []
        portfolio_values = []

        # Track trading actions across episodes
        trading_actions = {
            'buys': {ticker: [] for ticker in self.env.tickers},
            'sells': {ticker: [] for ticker in self.env.tickers},
            'holds': {ticker: [] for ticker in self.env.tickers}
        }
        allocation_history = {ticker: [] for ticker in self.env.tickers}
        timestamps = []

        # Track market regimes and strategy performance
        regime_performance = {}
        regime_transitions = []

        # Track meta-learning performance
        meta_learning_metrics = {
            'style_selections': [],
            'style_performance': {},
            'adaptation_history': [],
            'market_conditions': [],
            'regime_style_mapping': {},  # Track which styles work best in which regimes
            'style_transition_impact': []  # Track impact of style changes
        }

        # Calculate buy and hold baseline return
        buy_and_hold_return = self._calculate_buy_and_hold_return()
        print(f"\nBuy & Hold Baseline Return: {buy_and_hold_return:.2f}%")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")

        for episode in tqdm(range(n_episodes), desc="Evaluation Episodes", total=n_episodes):
            state = self.env.reset()
            episode_score = 0
            episode_values = [self.env.initial_capital]
            episode_returns = []
            episode_actions = {
                'buys': {ticker: 0 for ticker in self.env.tickers},
                'sells': {ticker: 0 for ticker in self.env.tickers},
                'holds': {ticker: 0 for ticker in self.env.tickers}
            }

            # Create initial market features
            market_features = self.agent._create_market_features()


            # Get meta-learner predictions for initial state
            with torch.no_grad():
                strategy_metrics = self.agent._get_strategy_metrics()  # Shape: [1, 9]
                style_weights = self.agent.meta_learner(
                    market_features,
                    strategy_metrics
                )
                # Get style probabilities
                style_weights = F.softmax(style_weights, dim=-1)
                selected_style = torch.argmax(style_weights[0]).item()
                self.agent.current_style = self.agent.style_names[selected_style]

            # Store initial style selection
            meta_learning_metrics['style_selections'].append({
                'episode': episode,
                'step': 0,
                'selected_style': self.agent.current_style,
                'style_weights': style_weights.cpu().numpy(),
                'market_regime': self.agent.market_context['current_regime'],
                'portfolio_value': self.env.initial_capital
            })

            # Signal episode start to agent
            self.agent.adapt_parameters(is_episode_start=True)

            for step in range(self.steps_per_episode):
                # Get actions from the agent
                discrete_probs, allocation_probs = self.agent.actor(state.unsqueeze(0))

                # Ensure allocation_probs has the right shape
                if allocation_probs.dim() == 1:
                    allocation_probs = allocation_probs.unsqueeze(0)

                # Sample discrete actions for each ticker
                discrete_actions = []
                for j in range(self.env.num_tickers):
                    # Ensure probs has correct shape [num_actions]
                    probs = discrete_probs[0, j] if discrete_probs.dim() == 3 else discrete_probs[j]

                    # Force buy if no positions and significant capital
                    if self.env.invested_capital < 1e-6 and self.env.remaining_capital > 1000:
                        action = 2  # Buy action
                    # Prohibit buy if no remaining capital
                    elif self.env.remaining_capital < 1e-6:
                        # Only sample from sell (0) or hold (1)
                        probs = torch.tensor([0.5, 0.5, 0.0], device=self.agent.gpu_device)
                        action = torch.multinomial(probs, 1).item()
                    else:
                        # Ensure probs is properly shaped and normalized
                        if probs.dim() != 1:
                            probs = probs.view(-1)  # Flatten to 1D
                        probs = F.softmax(probs, dim=0)  # Ensure valid probability distribution
                        action = torch.multinomial(probs, 1).item()
                    discrete_actions.append(action)

                    # Track actions
                    ticker = self.env.tickers[j]
                    if action == 0:  # Sell
                        episode_actions['sells'][ticker] += 1
                    elif action == 1:  # Hold
                        episode_actions['holds'][ticker] += 1
                    else:  # Buy
                        episode_actions['buys'][ticker] += 1

                # Track allocations
                for j, ticker in enumerate(self.env.tickers):
                    alloc_prob = allocation_probs[0, j] if allocation_probs.dim() == 2 else allocation_probs[j]
                    allocation_history[ticker].append(alloc_prob.item())

                if step == 0:  # Only track timestamps once per episode
                    timestamps.append(self.env.dates[step])

                # Take step with actions
                actions = (
                    torch.tensor(discrete_actions, device=self.agent.gpu_device),
                    allocation_probs[0] if allocation_probs.dim() == 2 else allocation_probs
                )
                next_state, reward, done, info = self.env.step(actions)

                # Update market context with actual reward
                self.agent.update_market_context(state, reward, done)

                # Track regime transitions and performance
                current_regime = self.agent.market_context['current_regime']
                if not regime_transitions or regime_transitions[-1] != current_regime:
                    regime_transitions.append(current_regime)

                    # Re-evaluate style selection on regime change
                    market_features = self.agent._create_market_features()

                    with torch.no_grad():
                        regime_encoding, _ = self.agent.market_encoder(market_features)


                        strategy_metrics = self.agent._get_strategy_metrics()  # Shape: [1, 9]

                        style_weights = self.agent.meta_learner(
                            market_features,
                            strategy_metrics
                        )


                        # Ensure we only have 3 style outputs and properly normalize them
                        style_weights = style_weights[:3] if style_weights.dim() == 1 else style_weights[:, :3]

                        style_weights = F.softmax(style_weights, dim=-1)


                    # Update style selection
                    old_style = self.agent.current_style
                    old_portfolio_value = info['portfolio_value'].item() if isinstance(info['portfolio_value'], torch.Tensor) else info['portfolio_value']
                    selected_style = torch.argmax(style_weights).item()


                    # No need for validation since we've constrained the output
                    self.agent.current_style = self.agent.style_names[selected_style]

                    # Store style transition with impact tracking
                    if old_style != self.agent.current_style:
                        meta_learning_metrics['style_transition_impact'].append({
                            'episode': episode,
                            'step': step,
                            'old_style': old_style,
                            'new_style': self.agent.current_style,
                            'market_regime': current_regime,
                            'portfolio_value_before': old_portfolio_value,
                            'style_weights': style_weights.cpu().numpy()
                        })

                # Update episode metrics
                episode_score += reward.item() if isinstance(reward, torch.Tensor) else reward
                portfolio_value = info['portfolio_value'].item() if isinstance(info['portfolio_value'], torch.Tensor) else info['portfolio_value']
                portfolio_values.append(portfolio_value)
                episode_values.append(portfolio_value)
                episode_returns.append(reward.detach() if isinstance(reward, torch.Tensor) else reward)

                # Track regime performance
                if current_regime not in regime_performance:
                    regime_performance[current_regime] = {
                        'returns': [],
                        'volatility': [],
                        'sharpe': [],
                        'drawdown': [],
                        'style_performance': {}  # Track performance by style within regime
                    }

                if len(episode_values) > 1:
                    regime_return = (episode_values[-1] - episode_values[-2]) / episode_values[-2]
                    regime_performance[current_regime]['returns'].append(regime_return)

                    # Track style performance within regime
                    if self.agent.current_style not in regime_performance[current_regime]['style_performance']:
                        regime_performance[current_regime]['style_performance'][self.agent.current_style] = {
                            'returns': [],
                            'sharpe': [],
                            'drawdown': []
                        }

                    style_perf = regime_performance[current_regime]['style_performance'][self.agent.current_style]
                    style_perf['returns'].append(regime_return)

                    # Calculate regime-specific metrics
                    regime_returns = np.array([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in regime_performance[current_regime]['returns']])
                    if len(regime_returns) > 1:
                        regime_performance[current_regime]['volatility'].append(np.std(regime_returns) * np.sqrt(252))
                        regime_performance[current_regime]['sharpe'].append(
                            np.sqrt(252) * (np.mean(regime_returns) / (np.std(regime_returns) + 1e-6))
                        )

                        # Calculate regime-specific drawdown
                        regime_values = np.array(episode_values)
                        peak = np.maximum.accumulate(regime_values)
                        drawdown = (peak - regime_values) / peak
                        regime_performance[current_regime]['drawdown'].append(np.max(drawdown))

                        # Calculate style-specific metrics within regime
                        style_returns = np.array([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in style_perf['returns']])
                        style_perf['sharpe'].append(
                            np.sqrt(252) * (np.mean(style_returns) / (np.std(style_returns) + 1e-6))
                        )
                        style_perf['drawdown'].append(np.max(drawdown))

                # Store adaptation history
                meta_learning_metrics['adaptation_history'].append({
                    'episode': episode,
                    'step': step,
                    'exploration_noise': float(self.agent.exploration_noise),
                    'exploration_temp': float(self.agent.exploration_temp),
                    'risk_preference': float(self.agent.risk_preference),
                    'learning_rate': float(self.agent.actor_optimizer.param_groups[0]['lr']),
                    'vol_scaling': float(self.agent.adaptive_params['vol_scaling_factor']),
                    'trend_scaling': float(self.agent.adaptive_params['trend_scaling_factor']),
                    'correlation_scaling': float(self.agent.adaptive_params['correlation_scaling_factor'])
                })

                # Store market conditions
                meta_learning_metrics['market_conditions'].append({
                    'episode': episode,
                    'step': step,
                    'regime': current_regime,
                    'volatility': float(np.mean(list(self.agent.market_context['volatility_history'])[-5:])) if len(self.agent.market_context['volatility_history']) >= 5 else 0.0,
                    'trend': float(np.mean(list(self.agent.market_context['returns_history'])[-20:])) if len(self.agent.market_context['returns_history']) >= 20 else 0.0,
                    'correlation': float(np.mean(list(self.agent.market_context['correlation_history'])[-5:])) if len(self.agent.market_context['correlation_history']) >= 5 else 0.0
                })

                if done:
                    break

                state = next_state

            # Store episode actions
            for action_type in ['buys', 'sells', 'holds']:
                for ticker in self.env.tickers:
                    trading_actions[action_type][ticker].append(episode_actions[action_type][ticker])

            eval_scores.append(episode_score)

            # Track style performance
            if self.agent.current_style not in meta_learning_metrics['style_performance']:
                meta_learning_metrics['style_performance'][self.agent.current_style] = {
                    'scores': [],
                    'returns': [],
                    'sharpe': [],
                    'drawdown': []
                }

            style_perf = meta_learning_metrics['style_performance'][self.agent.current_style]
            style_perf['scores'].append(episode_score)

            # Calculate style-specific metrics
            if len(episode_returns) > 1:
                returns_np = np.array([ret.cpu().numpy() if torch.is_tensor(ret) else ret for ret in episode_returns])
                style_perf['returns'].extend(returns_np)
                style_perf['sharpe'].append(
                    float(np.sqrt(252) * (np.mean(returns_np) / (np.std(returns_np) + 1e-6)))
                )

                # Calculate drawdown
                values_np = np.array(episode_values)
                peak = np.maximum.accumulate(values_np)
                drawdown = (peak - values_np) / peak
                style_perf['drawdown'].append(float(np.max(drawdown)))

        # Convert lists to numpy arrays after ensuring all elements are CPU scalars
        eval_scores_np = np.array(eval_scores)
        portfolio_values_np = np.array(portfolio_values)

        # Calculate evaluation metrics
        mean_score = np.mean(eval_scores_np)

        # Calculate Sharpe ratio
        returns = np.diff(portfolio_values_np) / portfolio_values_np[:-1]
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / (np.std(returns) + 1e-6))

        # Calculate maximum drawdown
        peak = portfolio_values_np[0]
        max_drawdown = 0
        for value in portfolio_values_np:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate final return
        final_return = ((portfolio_values_np[-1] - portfolio_values_np[0]) / portfolio_values_np[0]) * 100
        relative_return = final_return - buy_and_hold_return

        # Analyze regime-style relationships
        for regime, metrics in regime_performance.items():
            style_metrics = metrics['style_performance']
            if style_metrics:
                best_style = max(style_metrics.items(),
                               key=lambda x: np.mean(x[1]['sharpe']) if x[1]['sharpe'] else -np.inf)[0]
                meta_learning_metrics['regime_style_mapping'][regime] = {
                    'best_style': best_style,
                    'style_metrics': {
                        style: {
                            'mean_sharpe': float(np.mean(data['sharpe'])) if data['sharpe'] else 0.0,
                            'mean_return': float(np.mean(data['returns'])) if data['returns'] else 0.0,
                            'mean_drawdown': float(np.mean(data['drawdown'])) if data['drawdown'] else 0.0
                        }
                        for style, data in style_metrics.items()
                    }
                }

        # Analyze style transition impact
        for i, transition in enumerate(meta_learning_metrics['style_transition_impact']):
            if i < len(meta_learning_metrics['style_transition_impact']) - 1:  # Check if there's a next transition
                next_transition = meta_learning_metrics['style_transition_impact'][i + 1]
                transition['portfolio_value_after'] = next_transition['portfolio_value_before']
                transition['return'] = (transition['portfolio_value_after'] - transition['portfolio_value_before']) / transition['portfolio_value_before']
            else:
                # For the last transition, use the final portfolio value
                transition['portfolio_value_after'] = episode_values[-1]
                transition['return'] = (transition['portfolio_value_after'] - transition['portfolio_value_before']) / transition['portfolio_value_before']

        print("\nEvaluation Results:")
        print(f"Mean Score: {mean_score:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Portfolio Return: {final_return:.2f}%")
        print(f"Return vs Buy & Hold: {relative_return:+.2f}%")
        print(f"Final Market Regime: {self.agent.market_context['current_regime']}")
        print(f"Strategy Memory Size: {sum(len(strats) for strats in self.agent.strategy_memory.strategies.values())}")

        # Print regime performance statistics
        print("\nRegime Performance:")
        for regime, metrics in regime_performance.items():
            if metrics['returns']:
                regime_returns = np.array([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in metrics['returns']])
                regime_sharpe = np.sqrt(252) * (np.mean(regime_returns) / (np.std(regime_returns) + 1e-6))
                print(f"\n{regime}:")
                print(f"  Mean Return: {np.mean(regime_returns):.4f}")
                print(f"  Volatility: {np.std(regime_returns):.4f}")
                print(f"  Sharpe Ratio: {regime_sharpe:.2f}")
                print(f"  Best Style: {meta_learning_metrics['regime_style_mapping'][regime]['best_style']}")
                print("  Style Performance:")
                for style, style_metrics in metrics['style_performance'].items():
                    if style_metrics['returns']:
                        style_returns = np.array([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in style_metrics['returns']])
                        print(f"    {style}:")
                        print(f"      Mean Return: {np.mean(style_returns):.4f}")
                        print(f"      Mean Sharpe: {np.mean(style_metrics['sharpe']):.2f}")
                        print(f"      Mean Drawdown: {np.mean(style_metrics['drawdown']):.2%}")

        # Print meta-learning performance
        print("\nMeta-Learning Performance:")
        for style, metrics in meta_learning_metrics['style_performance'].items():
            if metrics['scores']:
                print(f"\n{style.capitalize()}:")
                print(f"  Mean Score: {np.mean(metrics['scores']):.4f}")
                print(f"  Mean Sharpe: {np.mean(metrics['sharpe']):.2f}")
                print(f"  Mean Drawdown: {np.mean(metrics['drawdown']):.2%}")
                print(f"  Selection Frequency: {sum(s['selected_style'] == style for s in meta_learning_metrics['style_selections']) / len(meta_learning_metrics['style_selections']):.2%}")

        # Print style transition impact
        print("\nStyle Transition Impact:")
        if meta_learning_metrics['style_transition_impact']:
            for transition in meta_learning_metrics['style_transition_impact']:
                if 'return' in transition:
                    print(f"\n{transition['old_style']} -> {transition['new_style']} in {transition['market_regime']}:")
                    print(f"  Return: {transition['return']:.2%}")
        else:
            print("  No style transitions occurred during evaluation")

        # Store evaluation results
        self.final_eval_scores.append({
            'episode': len(self.episode_scores),
            'score': float(mean_score),
            'sharpe': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'portfolio_return': float(final_return),
            'relative_return': float(relative_return),
            'market_regime': self.agent.market_context['current_regime'],
            'regime_transitions': regime_transitions,
            'regime_performance': {
                regime: {
                    'mean_return': float(np.mean([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in metrics['returns']])) if metrics['returns'] else 0.0,
                    'volatility': float(np.std([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in metrics['returns']])) if metrics['returns'] else 0.0,
                    'sharpe': float(np.sqrt(252) * (np.mean([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in metrics['returns']]) / (np.std([ret.detach().cpu().numpy() if torch.is_tensor(ret) else ret for ret in metrics['returns']]) + 1e-6))) if metrics['returns'] else 0.0,
                    'best_style': meta_learning_metrics['regime_style_mapping'][regime]['best_style'],
                    'style_metrics': meta_learning_metrics['regime_style_mapping'][regime]['style_metrics']
                }
                for regime, metrics in regime_performance.items()
            },
            'meta_learning_metrics': meta_learning_metrics
        })

        # Save trading statistics visualization
        self._save_trading_statistics(trading_actions, allocation_history, timestamps)

        # Reset mode after evaluation
        self.agent.mode = 'train'
        self.env.set_mode('train')

        # Clear caches after evaluation
        self._clear_caches()

        return mean_score, sharpe_ratio, max_drawdown

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
        """Save a comprehensive checkpoint."""
        # Initialize metrics if not provided
        if rollout_metrics is None:
            rollout_metrics = {'sharpe_ratio': 0.0, 'max_drawdown': 1.0}

        # Get metrics
        sharpe_ratio = rollout_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = rollout_metrics.get('max_drawdown', 1.0)

        # Normalize rollout score to [-1, 1] range using tanh
        normalized_rollout = np.tanh(rollout_score)

        # Normalize Sharpe ratio using modified sigmoid to handle negative values better
        # Center sigmoid around 0 and scale to handle typical Sharpe ratio ranges
        normalized_sharpe = 2.0 / (1.0 + np.exp(-sharpe_ratio/2.0)) - 1.0

        # Exponential penalty for drawdown that becomes more severe as drawdown increases
        # This creates a stronger penalty for larger drawdowns
        drawdown_penalty = np.exp(3.0 * max_drawdown) - 1.0  # Exponential scaling
        drawdown_factor = 1.0 / (1.0 + drawdown_penalty)  # Convert to [0, 1] range

        # Calculate composite score with balanced weighting:
        # - 40% weight on returns (normalized rollout score)
        # - 40% weight on risk-adjusted returns (normalized Sharpe)
        # - 20% weight on drawdown protection (exponential penalty)
        composite_score = (0.4 * normalized_rollout +
                          0.4 * normalized_sharpe +
                          0.2 * drawdown_factor)

        # Scale final score to [0, 100] range for better interpretability
        composite_score = 50 * (composite_score + 1.0)  # Maps [-1, 1] to [0, 100]

        # Store previous best metrics
        previous_best_composite = getattr(self, 'best_composite_score', -np.inf)

        # Determine if this is a new best model
        is_best_score = composite_score > previous_best_composite

        # Update best metrics if needed
        if is_best_score:
            self.best_rollout_score = rollout_score
            self.best_sharpe = sharpe_ratio
            self.best_drawdown = max_drawdown
            self.best_composite_score = composite_score
            print(f"\nNew best model with composite score: {composite_score:.4f}")
            print(f"Score: {rollout_score:.4f}")
            print(f"Sharpe: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")

        checkpoint = {
            'episode': episode,
            'metadata': {
                'tickers': list(self.env.tickers),  # Store the tickers the model was trained on
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'initial_capital': self.initial_capital,
                'model_version': '1.0'
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

    def _save_trading_statistics(self, trading_actions, allocation_history, timestamps):
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

                # Create figure with subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

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
                ax4.plot(param_data['episode'], param_data['actor_lr'], marker='o', color='green')
                ax4.set_title('Actor Learning Rate History')
                ax4.set_ylabel('Learning Rate')
                ax4.grid(True)

                # Color background based on parameter state
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
                stats['parameter_metrics'] = {
                    'exploration_noise': float(latest_params['exploration_noise']),
                    'exploration_temp': float(latest_params['exploration_temp']),
                    'risk_preference': float(latest_params['risk_preference']),
                    'actor_lr': float(latest_params['actor_lr']),
                    'parameter_state': latest_params['parameter_state'],
                    'relative_improvement': float(latest_params['relative_improvement'])
                }

            # Save JSON in the figures directory
            with open(f"{self.figures_dir}/trading_stats_{timestamp}.json", 'w') as f:
                json.dump(stats, f, indent=4)

        except Exception as e:
            print(f"\nWarning: Error in saving trading statistics: {str(e)}")
            print("This is non-critical and won't affect the training process.")

    def _calculate_buy_and_hold_return(self):
        """Calculate the return from a simple buy-and-hold strategy with uniform initial allocation"""
        self.env.reset()

        # Get initial prices
        initial_date = self.env.dates[0]
        initial_prices = torch.tensor(
            self.env.unscaled_close_df.loc[initial_date]['Close'].values,
            device=self.env.gpu_device,
            dtype=torch.float32
        )

        # Calculate integer number of shares with uniform allocation target
        allocation_per_ticker = self.env.initial_capital / self.env.num_tickers
        initial_shares = torch.floor(allocation_per_ticker / initial_prices)  # Integer number of shares

        # Calculate actual cost and remaining cash
        initial_cost = (initial_shares * initial_prices).sum()
        remaining_cash = self.env.initial_capital - initial_cost

        # Get final prices
        final_date = self.env.dates[-1]
        final_prices = torch.tensor(
            self.env.unscaled_close_df.loc[final_date]['Close'].values,
            device=self.env.gpu_device,
            dtype=torch.float32
        )

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

    def _evaluate_rollout(self):
        """Perform rollout evaluation with meta-learning integration."""
        self.agent.mode = 'rollout'
        self.env.set_mode('rollout')

        episode_scores = []
        episode_returns = []
        episode_values = []

        try:
            for episode in tqdm(range(self.rollout_episodes), desc="Rollout Episodes", total=self.rollout_episodes):
                state = self.env.reset()
                episode_score = 0
                episode_value_history = [self.env.initial_capital]
                episode_return_history = []

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


                self.agent.adapt_parameters(is_episode_start=True)

                for step in range(self.steps_per_episode):
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        action = self.agent.act(state, eval_mode=True)

                    next_state, reward, done, info = self.env.step(action)
                    self.agent.update_market_context(next_state, reward, done)

                    episode_score += reward.item() if torch.is_tensor(reward) else reward
                    portfolio_value = info['portfolio_value'].item() if torch.is_tensor(info['portfolio_value']) else info['portfolio_value']
                    episode_value_history.append(portfolio_value)
                    episode_return_history.append(reward)

                    if done:
                        break

                    state = next_state

                episode_scores.append(episode_score)
                episode_returns.extend(episode_return_history)
                episode_values.extend(episode_value_history)

            mean_score = float(np.mean(episode_scores))
            returns_np = np.array([ret.cpu().numpy() if torch.is_tensor(ret) else ret for ret in episode_returns])
            sharpe_ratio = float(np.sqrt(252) * (np.mean(returns_np) / (np.std(returns_np) + 1e-6)))

            values_np = np.array(episode_values)
            peak = np.maximum.accumulate(values_np)
            drawdown = (peak - values_np) / peak
            max_drawdown = float(np.max(drawdown))

            metrics = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }

            return mean_score, metrics

        finally:
            self.agent.mode = 'train'
            self.env.set_mode('train')

    def training_loop(self):
        """
        Execute the complete training pipeline including final evaluation.
        """
        print("Starting training pipeline...")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")

        # Initialize progress tracking
        import time
        progress_tracker.update_rl_agent_progress(
            status="running",
            message="Initializing RL agent training...",
            progress=0.0,
            current_episode=0,
            total_episodes=self.n_episodes,
            start_time=time.time()
        )

        # Execute training
        print("\nTraining phase:")
        scores = self.train()
        print("\nTraining completed successfully")

        # Update progress before evaluation
        progress_tracker.update_rl_agent_progress(
            message="Training completed. Starting final evaluation...",
            progress=0.9,
            current_episode=self.n_episodes,
            total_episodes=self.n_episodes
        )

        # Perform final evaluation
        print("\nStarting final evaluation phase...")
        mean_score, sharpe_ratio, max_drawdown = self.evaluate(n_episodes=50)  # Using 50 episodes for thorough evaluation

        # Update progress after evaluation
        progress_tracker.update_rl_agent_progress(
            status="completed",
            message="RL agent training and evaluation completed successfully!",
            progress=1.0,
            current_score=mean_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            end_time=time.time()
        )

        print("\nFinal Evaluation Results:")
        print(f"Mean Evaluation Score: {mean_score:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")

        return scores, (mean_score, sharpe_ratio, max_drawdown)

    def validate_functionality(self):
        """
        Quick validation method to test if training, rollout, and evaluation are working properly.
        Runs a single training episode, rollout evaluation, and two evaluation episodes.
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
            next_state, reward, done, info = self.env.step(action)

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
        rollout_score, rollout_metrics = self._evaluate_rollout()
        print(f"Rollout evaluation completed with score: {rollout_score:.2f}")

        # Run evaluation test with two episodes
        print("\nTesting evaluation functionality (2 episodes):")
        mean_score, sharpe_ratio, max_drawdown = self.evaluate(n_episodes=2)
        print(f"Evaluation test completed:")
        print(f"Mean Score: {mean_score:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

        return True

    def _evaluate_and_save(self, episode, scores):
        """Evaluate the agent and save checkpoint if appropriate."""
        # Update progress before evaluation
        progress_tracker.update_rl_agent_progress(
            message=f"Evaluating agent at episode {episode}...",
            current_eval=episode // self.eval_frequency,
            total_evals=self.n_episodes // self.eval_frequency
        )

        # Perform rollout evaluation
        rollout_score, rollout_metrics = self._evaluate_rollout()
        self.rollout_scores.append(rollout_score)

        # Calculate composite score considering multiple metrics
        sharpe_ratio = rollout_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = rollout_metrics.get('max_drawdown', 1.0)
        composite_score = rollout_score * (1 + sharpe_ratio) * (1 - max_drawdown)

        # Update progress with evaluation results
        progress_tracker.update_rl_agent_progress(
            current_score=rollout_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            best_score=getattr(self, 'best_composite_score', float('-inf'))
        )

        # Update best score if needed
        is_best = False
        if composite_score > getattr(self, 'best_composite_score', float('-inf')):
            self.best_composite_score = composite_score
            self.best_rollout_score = rollout_score
            self.best_sharpe = sharpe_ratio
            self.best_drawdown = max_drawdown
            is_best = True
            print(f"\nNew best model!")
            print(f"Composite Score: {composite_score:.4f}")
            print(f"Rollout Score: {rollout_score:.4f}")
            print(f"Sharpe: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")

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
        training_batch_size=64,
        eval_batch_size=32,
        rollout_episodes=10,
        initial_capital=10000
    )
    trainer.training_loop()