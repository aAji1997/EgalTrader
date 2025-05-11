import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Tuple
import torch
from collections import deque

from RL_framework import RLFramework


class PortfolioEnv(gym.Env):
    def __init__(self, tickers=['INTC', 'HPE'], mode='train', initial_capital=10_000):
        # Initialize live trading attributes first
        self.is_live_trading = False
        self.live_portfolio_state = {
            'positions': None,
            'remaining_capital': None,
            'invested_capital': None,
            'prev_portfolio_value': None
        }
        
        self.initial_capital = initial_capital
        self.rl_framework = RLFramework(tickers=tickers)
        self.train_df, self.val_df, self.unscaled_close_train_df, self.unscaled_close_val_df = self.rl_framework.final_preprocessing()
        
        # Sort and index DataFrames properly
        self.train_df = self.train_df.sort_index()
        self.val_df = self.val_df.sort_index()
        self.unscaled_close_train_df = self.unscaled_close_train_df.sort_index()
        self.unscaled_close_val_df = self.unscaled_close_val_df.sort_index()
        
        # Create sorted indices
        self.train_df = self.train_df.sort_index(level=['Date', 'ticker'])
        self.val_df = self.val_df.sort_index(level=['Date', 'ticker'])
        self.unscaled_close_train_df = self.unscaled_close_train_df.sort_index(level=['Date', 'ticker'])
        self.unscaled_close_val_df = self.unscaled_close_val_df.sort_index(level=['Date', 'ticker'])
        
        # Split training data into train and rollout sets (80-20 split)
        train_dates = self.train_df.index.get_level_values('Date').unique()
        split_idx = int(len(train_dates) * 0.8)
        
        self.pure_train_dates = train_dates[:split_idx]
        self.rollout_dates = train_dates[split_idx:]
        self.val_dates = self.val_df.index.get_level_values('Date').unique()
        
        # Initialize mode-specific dates
        self.dates = self.pure_train_dates  # Default to training mode
        self.data_df = self.train_df
        self.unscaled_close_df = self.unscaled_close_train_df
        
        print(f"Data splits:")
        print(f"Training dates: {len(self.pure_train_dates)}")
        print(f"Rollout dates: {len(self.rollout_dates)}")
        print(f"Validation dates: {len(self.val_dates)}")
        
        # Set mode (train or eval)
        self.mode = mode
        
        # Get tickers from MultiIndex
        self.tickers = self.train_df.index.get_level_values('ticker').unique()
        self.num_tickers = len(self.tickers)
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.tickers)}
        
        # Portfolio constraints
        self.max_position_size = 0.5
        self.min_position_size = 0.0
        self.max_leverage = 1.0
        
        # Add scaling parameters for portfolio features
        self.max_portfolio_value = self.initial_capital * self.max_leverage
        self.min_portfolio_value = 0
        
        # Action space remains the same
        self.action_space = Tuple((
            MultiDiscrete([3] * self.num_tickers),
            Box(
                low=self.min_position_size,
                high=self.max_position_size,
                shape=(self.num_tickers,),
                dtype=np.float32
            )
        ))
        
        # Define observation space
        num_features = len(self.train_df.columns)
        num_portfolio_features = 2  # invested_capital, remaining_capital
        total_features = self.num_tickers * num_features + num_portfolio_features
        
        print(f"Initializing environment:")
        print(f"Number of tickers: {self.num_tickers}")
        print(f"Features per ticker: {num_features}")
        print(f"Portfolio features: {num_portfolio_features}")
        print(f"Total features: {total_features}")
        
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )
        
        # Store these for validation
        self.num_features_per_ticker = num_features
        self.total_features = total_features
        
        # Add device handling
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        # Initialize price cache on CPU
        self.price_cache = {}
        
        # Initialize state tracking (no batch dimension)
        self.reset()
        
        # Add caching for data access
        self.price_cache = {}
        self.observation_cache = {}
        self.metrics_cache = {}
        
        # Add batch processing parameters
        self.batch_size = 32
        self.precompute_window = 20
        self.cache_update_frequency = 50  # Update cache every N steps
        
        # Add caching for historical calculations
        self.history_cache = {
            'volatility': {},  # Cache for historical volatilities
            'returns': {},     # Cache for historical returns
            'prices': {}       # Cache for historical prices
        }
    
    def reset(self):
        """Reset the environment with memory optimization"""
        # If in live trading mode and we have a saved state, use it
        if self.is_live_trading and all(v is not None for v in self.live_portfolio_state.values()):
            self.current_step = 0
            self.positions = self.live_portfolio_state['positions']
            self.remaining_capital = self.live_portfolio_state['remaining_capital']
            self.invested_capital = self.live_portfolio_state['invested_capital']
            self.prev_portfolio_value = self.live_portfolio_state['prev_portfolio_value']
        else:
            # Regular reset for training/evaluation
            self.current_step = 0
            self.positions = torch.zeros(self.num_tickers, device=self.cpu_device)
            self.remaining_capital = float(self.initial_capital)
            self.invested_capital = 0.0
            self.prev_portfolio_value = float(self.initial_capital)
        
        # Clear price cache
        self.price_cache.clear()
        
        # Get initial observation (this will move to GPU)
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current observation with memory optimization"""
        try:
            # Process on CPU first
            # Get market data for current step
            current_date = self.dates[self.current_step]
            market_data = self.data_df.loc[current_date]
            
            # Convert to tensor on CPU
            market_tensor = torch.tensor(
                market_data.values,
                dtype=torch.float32,
                device=self.cpu_device
            ).reshape(-1)
            
            # Add portfolio state on CPU
            portfolio_tensor = torch.tensor(
                [self.invested_capital, self.remaining_capital],
                dtype=torch.float32,
                device=self.cpu_device
            )
            
            # Combine features on CPU
            state = torch.cat([market_tensor, portfolio_tensor])
            
            # Move to GPU only at the end
            return state.to(self.gpu_device)
            
        except Exception as e:
            print(f"Error in _get_observation: {e}")
            # Return zero tensor on GPU
            return torch.zeros(self.observation_space.shape[0], device=self.gpu_device)
    
    def step(self, actions):
        """Take a step in the environment with memory optimization"""
        try:
            # Move actions to CPU for processing
            discrete_actions = actions[0].to(self.cpu_device) if isinstance(actions[0], torch.Tensor) else torch.tensor(actions[0], device=self.cpu_device)
            allocation_percentages = actions[1].to(self.cpu_device) if isinstance(actions[1], torch.Tensor) else torch.tensor(actions[1], device=self.cpu_device)
            
            # Get current prices and forecasts on CPU
            current_date = self.dates[self.current_step]
            current_data = self.data_df.loc[current_date]
            current_prices = torch.tensor(
                self.unscaled_close_df.loc[current_date]['Close'].values,
                device=self.cpu_device,
                dtype=torch.float32
            )
            
            # Get forecast data
            forecast_returns = torch.tensor(
                current_data['Forecast_Return'].values,
                device=self.cpu_device,
                dtype=torch.float32
            )
            forecast_momentum = torch.tensor(
                current_data['Forecast_Momentum'].values,
                device=self.cpu_device,
                dtype=torch.float32
            )
            forecast_confidence = torch.tensor(
                current_data['Forecast_Confidence'].values,
                device=self.cpu_device,
                dtype=torch.float32
            )
            
            # Process actions on CPU
            buy_masks = (discrete_actions == 2)  # Buy actions
            sell_masks = (discrete_actions == 0)  # Sell actions
            hold_masks = (discrete_actions == 1)  # Hold actions
            
            # Add randomness to allocation percentages on CPU
            random_factor = 0.8 + 0.4 * torch.rand_like(allocation_percentages)
            
            # Pre-allocation safety check
            max_initial_allocation = self.initial_capital * 0.95  # Leave 5% buffer
            
            # Ensure allocation percentages sum to 1.0
            allocation_percentages = allocation_percentages / (allocation_percentages.sum() + 1e-8)
            
            # Calculate initial buy amounts with stricter limits
            potential_buy_amounts = torch.where(
                buy_masks,
                torch.minimum(
                    self.remaining_capital * allocation_percentages * random_factor,
                    torch.tensor(max_initial_allocation / self.num_tickers, device=self.cpu_device)
                ),
                torch.zeros_like(allocation_percentages)
            )
            
            # Scale down if total exceeds remaining capital or initial capital
            total_potential_buy = potential_buy_amounts.sum()
            scale_factor = torch.minimum(
                torch.minimum(
                    self.remaining_capital / (total_potential_buy + 1e-8),
                    torch.tensor(max_initial_allocation / (total_potential_buy + 1e-8), device=self.cpu_device)
                ),
                torch.tensor(1.0, device=self.cpu_device)
            ).item()
            
            potential_buy_amounts = potential_buy_amounts * scale_factor
            
            # Execute trades on CPU
            for i in range(self.num_tickers):
                if buy_masks[i]:
                    # Calculate shares to buy
                    shares_to_buy = torch.floor(potential_buy_amounts[i] / current_prices[i])
                    cost = shares_to_buy * current_prices[i]
                    
                    if cost <= self.remaining_capital:
                        self.positions[i] += shares_to_buy
                        self.remaining_capital -= cost
                        self.invested_capital += cost
                
                elif sell_masks[i]:
                    # Sell all shares
                    if self.positions[i] > 0:
                        sale_proceeds = self.positions[i] * current_prices[i]
                        self.remaining_capital += sale_proceeds
                        self.invested_capital -= sale_proceeds
                        self.positions[i] = 0
            
            # Calculate portfolio value and return
            portfolio_value = (self.positions * current_prices).sum() + self.remaining_capital
            portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value if self.prev_portfolio_value > 0 else 0
            
            # Update state
            self.prev_portfolio_value = portfolio_value
            self.current_step += 1
            
            # Calculate reward components on CPU
            base_reward = portfolio_return
            
            # Calculate forecast alignment reward
            forecast_alignment_reward = 0.0
            for i in range(self.num_tickers):
                if buy_masks[i] and forecast_returns[i] > 0:
                    # Reward for buying when forecast predicts increase
                    forecast_alignment_reward += 0.001 * forecast_returns[i] * forecast_confidence[i]
                elif sell_masks[i] and forecast_returns[i] < 0:
                    # Reward for selling when forecast predicts decrease
                    forecast_alignment_reward += 0.001 * abs(forecast_returns[i]) * forecast_confidence[i]
                elif hold_masks[i] and abs(forecast_returns[i]) < 0.001:
                    # Small reward for holding when forecast predicts stability
                    forecast_alignment_reward += 0.0001 * forecast_confidence[i]
            
            # Add momentum-based rewards
            momentum_reward = 0.0
            for i in range(self.num_tickers):
                if buy_masks[i] and forecast_momentum[i] > 0:
                    momentum_reward += 0.0005 * forecast_momentum[i]
                elif sell_masks[i] and forecast_momentum[i] < 0:
                    momentum_reward += 0.0005 * abs(forecast_momentum[i])
            
            # Add position-based penalties
            total_positions = self.positions.sum()
            if total_positions < 1e-6:
                base_reward -= 0.001  # Small penalty for having no positions
            
            # Add transaction cost penalty
            transaction_mask = buy_masks | sell_masks
            num_transactions = transaction_mask.sum()
            transaction_penalty = 0.0001 * num_transactions
            
            # Add forecast misalignment penalty
            misalignment_penalty = 0.0
            for i in range(self.num_tickers):
                if buy_masks[i] and forecast_returns[i] < -0.01:
                    # Penalty for buying when forecast predicts significant decrease
                    misalignment_penalty += 0.001 * abs(forecast_returns[i]) * forecast_confidence[i]
                elif sell_masks[i] and forecast_returns[i] > 0.01:
                    # Penalty for selling when forecast predicts significant increase
                    misalignment_penalty += 0.001 * forecast_returns[i] * forecast_confidence[i]
            
            # Combine rewards
            reward = base_reward + forecast_alignment_reward + momentum_reward - transaction_penalty - misalignment_penalty
            
            # Check if episode is done
            done = self.current_step >= len(self.dates) - 1
            
            # Get next observation
            next_state = self._get_observation()
            
            # Prepare info dict with additional forecast metrics
            info = {
                'portfolio_value': portfolio_value,
                'portfolio_return': portfolio_return,
                'positions': self.positions.clone(),
                'transaction_cost': transaction_penalty,
                'forecast_alignment_reward': forecast_alignment_reward,
                'momentum_reward': momentum_reward,
                'misalignment_penalty': misalignment_penalty
            }
            
            # Move reward and next_state to GPU for the agent
            if isinstance(reward, torch.Tensor):
                reward = reward.to(self.gpu_device)
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.to(self.gpu_device)
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), 0.0, True, {}
    
    def get_train_observation(self, idx):
        """Get observation for training dataset at specific index"""
        original_step = self.current_step
        self.current_step = idx
        obs = self._get_observation()
        self.current_step = original_step
        return obs
    
    def get_val_observation(self, idx):
        """Get observation for validation dataset at specific index"""
        original_mode = self.mode
        original_step = self.current_step
        self.mode = 'eval'
        self.current_step = idx
        obs = self._get_observation()
        self.current_step = original_step
        self.mode = original_mode
        return obs
    
    def set_mode(self, mode):
        """Switch between training, rollout and evaluation modes"""
        if mode not in ['train', 'rollout', 'eval']:
            raise ValueError("Mode must be either 'train', 'rollout' or 'eval'")
        
        # Save live trading state before mode switch if in live trading
        if self.is_live_trading:
            self.live_portfolio_state = {
                'positions': self.positions.clone(),
                'remaining_capital': float(self.remaining_capital),
                'invested_capital': float(self.invested_capital),
                'prev_portfolio_value': float(self.prev_portfolio_value)
            }
        
        self.mode = mode
        if mode == 'train':
            self.dates = self.pure_train_dates
            self.is_live_trading = False
        elif mode == 'rollout':
            self.dates = self.rollout_dates
            self.is_live_trading = False
        else:  # eval
            self.dates = self.val_dates
            # Keep is_live_trading state as is
        
        self.data_df = self.train_df if mode != 'eval' else self.val_df
        self.unscaled_close_df = self.unscaled_close_train_df if mode != 'eval' else self.unscaled_close_val_df
        self.reset()
    
    def start_live_trading(self, initial_capital=None):
        """Start live trading mode with optional initial capital update"""
        if initial_capital is not None:
            self.initial_capital = initial_capital
            self.max_portfolio_value = self.initial_capital * self.max_leverage
        
        self.is_live_trading = True
        self.live_portfolio_state = {
            'positions': torch.zeros(self.num_tickers, device=self.cpu_device),
            'remaining_capital': float(self.initial_capital),
            'invested_capital': 0.0,
            'prev_portfolio_value': float(self.initial_capital)
        }
        self.set_mode('eval')  # Switch to eval mode for live trading
    
    def update_live_portfolio_state(self, positions=None, remaining_capital=None):
        """Update the live trading portfolio state with actual values"""
        if not self.is_live_trading:
            raise ValueError("Cannot update portfolio state: not in live trading mode")
        
        if positions is not None:
            self.live_portfolio_state['positions'] = torch.tensor(positions, device=self.cpu_device)
            self.positions = self.live_portfolio_state['positions']
        
        if remaining_capital is not None:
            self.live_portfolio_state['remaining_capital'] = float(remaining_capital)
            self.remaining_capital = self.live_portfolio_state['remaining_capital']
        
        # Update other derived values
        current_prices = self.get_current_prices()
        self.invested_capital = (self.positions * current_prices).sum()
        self.live_portfolio_state['invested_capital'] = float(self.invested_capital)
        
        portfolio_value = self.invested_capital + self.remaining_capital
        self.live_portfolio_state['prev_portfolio_value'] = float(portfolio_value)
        self.prev_portfolio_value = self.live_portfolio_state['prev_portfolio_value']

    def _calculate_action_diversity(self, actions):
        """Calculate diversity of trading actions across tickers"""
        # Count actions per type using one-hot encoding and sum
        action_counts = torch.zeros(3, device=actions.device)
        action_counts.scatter_add_(0, actions, torch.ones_like(actions, dtype=torch.float32))
        
        # Calculate entropy of action distribution
        probs = action_counts / len(actions)
        # Only consider non-zero probabilities
        mask = probs > 0
        entropy = -(probs[mask] * torch.log(probs[mask] + 1e-10)).sum()
        
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(3.0, device=actions.device))
        return entropy / max_entropy

    def _calculate_allocation_diversity(self, weights):
        """Calculate diversity of portfolio allocations"""
        weights = torch.abs(weights)
        total_weight = weights.sum()
        
        if total_weight == 0:
            return torch.tensor(0.0, device=weights.device)
        
        # Normalize weights
        weights = weights / total_weight
        sorted_weights, _ = torch.sort(weights)
        n = len(weights)
        n_tensor = torch.tensor(n, dtype=torch.float32, device=weights.device)
        
        # Calculate Gini coefficient using tensor operations
        indices = torch.arange(1, n + 1, dtype=torch.float32, device=weights.device)
        gini = (2 * (indices * sorted_weights).sum() / (n_tensor * weights.sum())) - (n_tensor + 1) / n_tensor
        return 1 - gini

    def _calculate_correlation_penalty(self, returns, weights):
        """Optimized correlation penalty calculation with caching"""
        cache_key = (self.current_step, 'correlation')
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        if not hasattr(self, 'returns_history'):
            self.returns_history = deque(maxlen=20)
        
        self.returns_history.append(returns)
        
        if len(self.returns_history) < 2:
            return torch.tensor(0.0, device=returns.device)
        
        # Batch process correlation calculation
        returns_stack = torch.stack(list(self.returns_history))
        centered = returns_stack - returns_stack.mean(dim=0)
        cov = centered.T @ centered / (returns_stack.size(0) - 1)
        std = torch.sqrt(torch.diag(cov))
        corr_matrix = cov / (std.unsqueeze(1) @ std.unsqueeze(0))
        
        mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1)
        correlations = torch.abs(corr_matrix * mask)
        
        avg_correlation = correlations.sum() / mask.sum()
        
        # Cache the result
        self.metrics_cache[cache_key] = avg_correlation
        return avg_correlation

    def _get_prices_for_date(self, date):
        """Optimized price lookup with caching"""
        if date in self.history_cache['prices']:
            return self.history_cache['prices'][date]
            
        prices = self.unscaled_close_df.xs(date, level='Date')['Close'].values
        prices_tensor = torch.tensor(prices, dtype=torch.float32, device=self.gpu_device)
        self.history_cache['prices'][date] = prices_tensor
        return prices_tensor

    def _precompute_batch_data(self, start_idx, end_idx):
        """Pre-compute and cache historical data for batch processing"""
        dates = self.dates[start_idx:end_idx]
        
        # Batch compute historical prices and returns
        prices_batch = []
        returns_batch = []
        
        # First pass: Cache all prices
        for date in dates:
            if date not in self.history_cache['prices']:
                try:
                    prices = self.unscaled_close_df.xs(date, level='Date')['Close'].values
                    self.history_cache['prices'][date] = torch.tensor(
                        prices, dtype=torch.float32, device=self.gpu_device
                    )
                except Exception as e:
                    print(f"Warning: Could not get prices for date {date}: {e}")
                    # Use previous prices or zeros if no previous prices exist
                    prev_prices = prices_batch[-1] if prices_batch else torch.zeros(
                        self.num_tickers, dtype=torch.float32, device=self.gpu_device
                    )
                    self.history_cache['prices'][date] = prev_prices
            
            prices_batch.append(self.history_cache['prices'][date])
        
        # Second pass: Safely compute returns
        for i in range(1, len(prices_batch)):
            current_date = dates[i]
            prev_date = dates[i-1]
            
            if current_date not in self.history_cache['returns']:
                try:
                    returns = (prices_batch[i] - prices_batch[i-1]) / prices_batch[i-1]
                    self.history_cache['returns'][current_date] = returns
                except Exception as e:
                    print(f"Warning: Could not compute returns for date {current_date}: {e}")
                    # Use zeros for returns if computation fails
                    self.history_cache['returns'][current_date] = torch.zeros_like(prices_batch[i])
        
        # Third pass: Safely compute volatilities
        for i in range(self.precompute_window, len(dates)):
            current_date = dates[i]
            try:
                # Collect available returns for the window
                window_returns = []
                for j in range(i-self.precompute_window, i):
                    if dates[j] in self.history_cache['returns']:
                        window_returns.append(self.history_cache['returns'][dates[j]])
                
                # Only compute volatility if we have enough returns
                if len(window_returns) >= self.precompute_window // 2:  # Require at least half the window
                    window_returns_tensor = torch.stack(window_returns)
                    volatility = window_returns_tensor.std(dim=0)
                    self.history_cache['volatility'][(current_date, self.precompute_window)] = volatility
                else:
                    # Use a default volatility if not enough data
                    self.history_cache['volatility'][(current_date, self.precompute_window)] = torch.ones(
                        self.num_tickers, device=self.gpu_device
                    ) * 0.01  # Default 1% volatility
            except Exception as e:
                print(f"Warning: Could not compute volatility for date {current_date}: {e}")
                # Use default volatility on error
                self.history_cache['volatility'][(current_date, self.precompute_window)] = torch.ones(
                    self.num_tickers, device=self.gpu_device
                ) * 0.01


