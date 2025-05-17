import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Tuple
import torch
from collections import deque

from RL_framework import RLFramework


class PortfolioEnv(gym.Env):
    def __init__(self, tickers=['NVDA', 'FTNT'], mode='train', initial_capital=10_000):
        # Initialize live trading attributes first
        self.is_live_trading = False
        self.live_portfolio_state = {
            'positions': None,
            'remaining_capital': None,
            'invested_capital': None,
            'prev_portfolio_value': None
        }

        # Initialize buy-and-hold baseline tracking
        self.track_buy_and_hold = True
        self.buy_and_hold_positions = None
        self.buy_and_hold_initial_prices = None
        self.buy_and_hold_value = initial_capital
        self.buy_and_hold_prev_value = initial_capital
        self.outperformance_history = deque(maxlen=100)  # Track outperformance vs buy-and-hold

        # Look-ahead parameters for reward calculation
        self.look_ahead_window = 5  # Number of days to look ahead for evaluating trading decisions
        self.market_timing_multiplier = {
            'correct_sell': 20.0,    # 10x increased multiplier for well-timed sells (before price drops)
            'incorrect_sell': 20.0,  # 10x increased penalty multiplier for poorly-timed sells (before price rises)
            'correct_buy': 20.0,     # 10x increased multiplier for well-timed buys (before price rises)
            'incorrect_buy': 20.0    # 10x increased penalty multiplier for poorly-timed buys (before price drops)
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

        # Get all available dates (combining training and validation data)
        train_dates = self.train_df.index.get_level_values('Date').unique()
        val_dates = self.val_df.index.get_level_values('Date').unique()

        # Print date range information for debugging
        print(f"Training data date range: {train_dates.min()} to {train_dates.max()}")
        print(f"Validation data date range: {val_dates.min()} to {val_dates.max()}")

        # Define specific date ranges for rollout (October, November, and December 2024)
        oct_2024_start = pd.Timestamp('2024-10-01')
        oct_2024_end = pd.Timestamp('2024-10-31')
        nov_2024_start = pd.Timestamp('2024-11-01')
        nov_2024_end = pd.Timestamp('2024-11-30')
        dec_2024_start = pd.Timestamp('2024-12-01')
        dec_2024_end = pd.Timestamp('2024-12-31')

        # Define specific date ranges for final evaluation (January, February, and March 2025)
        jan_2025_start = pd.Timestamp('2025-01-01')
        jan_2025_end = pd.Timestamp('2025-01-31')
        feb_2025_start = pd.Timestamp('2025-02-01')
        feb_2025_end = pd.Timestamp('2025-02-28')  # February 2025 has 28 days
        mar_2025_start = pd.Timestamp('2025-03-01')
        mar_2025_end = pd.Timestamp('2025-03-31')

        # Filter validation dates to get October, November, and December 2024 for rollout
        rollout_dates = val_dates[
            ((val_dates >= oct_2024_start) & (val_dates <= oct_2024_end)) |
            ((val_dates >= nov_2024_start) & (val_dates <= nov_2024_end)) |
            ((val_dates >= dec_2024_start) & (val_dates <= dec_2024_end))
        ]

        # Filter validation dates to get January, February, and March 2025 for final evaluation
        final_eval_dates = val_dates[
            ((val_dates >= jan_2025_start) & (val_dates <= jan_2025_end)) |
            ((val_dates >= feb_2025_start) & (val_dates <= feb_2025_end)) |
            ((val_dates >= mar_2025_start) & (val_dates <= mar_2025_end))
        ]

        # Use all dates before October 2024 for training
        all_train_dates = np.concatenate([
            train_dates,
            val_dates[(val_dates < oct_2024_start)]
        ])

        # Sort the combined training dates
        all_train_dates = np.sort(all_train_dates)

        # Set training, rollout, and final evaluation dates
        self.pure_train_dates = all_train_dates
        self.rollout_dates = rollout_dates
        self.final_eval_dates = final_eval_dates

        # Set val_dates to empty (removing separate evaluation set)
        self.val_dates = np.array([])

        print(f"Using all {len(self.pure_train_dates)} available dates before October 2024 for training")
        print(f"Rollout will use October, November, and December 2024 dates ({len(self.rollout_dates)} dates)")
        print(f"Final evaluation will use January, February, and March 2025 dates ({len(self.final_eval_dates)} dates)")

        # Initialize mode-specific dates
        self.dates = self.pure_train_dates  # Default to training mode
        self.data_df = self.train_df
        self.unscaled_close_df = self.unscaled_close_train_df

        print(f"Data splits:")
        print(f"Training dates: {len(self.pure_train_dates)}")
        print(f"Rollout dates: {len(self.rollout_dates)}")
        print(f"Final evaluation dates: {len(self.final_eval_dates)}")
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

        # Initialize caches before reset
        self.price_cache = {}
        self.observation_cache = {}
        self.metrics_cache = {}

        # Add caching for historical calculations - MUST be initialized before reset
        self.history_cache = {
            'volatility': {},  # Cache for historical volatilities
            'returns': {},     # Cache for historical returns
            'prices': {}       # Cache for historical prices
        }

        # Add batch processing parameters
        self.batch_size = 32
        self.precompute_window = 20
        self.cache_update_frequency = 50  # Update cache every N steps

        # Initialize state tracking (no batch dimension)
        self.reset()

    def reset(self):
        """Reset the environment with memory optimization"""
        # If in live trading mode and we have a saved state, use it
        if self.is_live_trading and all(v is not None for v in self.live_portfolio_state.values()):
            self.current_step = 0
            self.positions = self.live_portfolio_state['positions']
            self.remaining_capital = self.live_portfolio_state['remaining_capital']
            self.invested_capital = max(0.0, self.live_portfolio_state['invested_capital'])  # Ensure non-negative
            self.prev_portfolio_value = self.live_portfolio_state['prev_portfolio_value']
        else:
            # Regular reset for training/evaluation
            self.current_step = 0
            self.positions = torch.zeros(self.num_tickers, device=self.cpu_device)
            self.remaining_capital = float(self.initial_capital)
            self.invested_capital = 0.0
            self.prev_portfolio_value = float(self.initial_capital)

            # Debug output for reset
            '''
            if self.mode == 'rollout':
                print(f"\nEnvironment reset:")
                print(f"  Initial capital: ${self.initial_capital:.2f}")
                print(f"  Positions: {self.positions}")
                print(f"  Mode: {self.mode}")
            '''

            # Reset buy-and-hold baseline
            if self.track_buy_and_hold:
                # Initialize buy-and-hold with equal allocation across all tickers
                if self.current_step < len(self.dates):
                    current_date = self.dates[self.current_step]
                    initial_prices = self._get_prices_for_date(current_date)

                    # Calculate equal allocation
                    allocation_per_ticker = self.initial_capital / self.num_tickers

                    # Calculate shares to buy for each ticker
                    self.buy_and_hold_positions = torch.zeros(self.num_tickers, device=self.cpu_device)
                    remaining_capital = float(self.initial_capital)

                    for i in range(self.num_tickers):
                        # Calculate shares to buy (floor to ensure we don't exceed capital)
                        shares = torch.floor(allocation_per_ticker / initial_prices[i]).item()
                        cost = shares * initial_prices[i].item()

                        # Update positions and remaining capital
                        self.buy_and_hold_positions[i] = shares
                        remaining_capital -= cost

                    # Store initial state
                    self.buy_and_hold_initial_prices = initial_prices.clone()
                    self.buy_and_hold_value = self.initial_capital
                    self.buy_and_hold_prev_value = self.initial_capital

                    # Reset outperformance history
                    self.outperformance_history.clear()

        # Clear price cache
        self.price_cache.clear()

        # Sanity check for portfolio values
        total_value = self.remaining_capital + self.invested_capital
        if abs(total_value - self.prev_portfolio_value) > 1.0:
            print(f"WARNING: Portfolio value mismatch during reset!")
            print(f"  Remaining capital: ${self.remaining_capital:.2f}")
            print(f"  Invested capital: ${self.invested_capital:.2f}")
            print(f"  Total: ${total_value:.2f}")
            print(f"  Previous portfolio value: ${self.prev_portfolio_value:.2f}")
            # Fix the mismatch
            self.prev_portfolio_value = total_value

        # Get initial observation (this will move to GPU)
        return self._get_observation()

    def _get_observation(self):
        """Get the current observation with memory optimization"""
        try:
            # Process on CPU first
            # Get market data for current step
            current_date = self.dates[self.current_step]

            # Check if we need to use validation data for this date (for dates from validation set during training)
            use_val_data = False
            if self.mode == 'train':
                try:
                    # Try to get data from training set first
                    _ = self.train_df.loc[current_date]
                except KeyError:
                    # If date not in training set, it must be from validation set (before Jan 2025)
                    use_val_data = True

            # Use appropriate data source based on mode and date
            if self.mode == 'rollout' or use_val_data:
                # Use validation data for rollout mode or validation dates during training
                market_data = self.val_df.loc[current_date]
            else:
                # Use training data for normal training dates
                market_data = self.train_df.loc[current_date]

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

            # Get current date
            current_date = self.dates[self.current_step]

            # Check if we need to use validation data for this date (for dates from validation set during training)
            use_val_data = False
            if self.mode == 'train':
                try:
                    # Try to get data from training set first
                    _ = self.train_df.loc[current_date]
                except KeyError:
                    # If date not in training set, it must be from validation set (before Jan 2025)
                    use_val_data = True

            # Use appropriate data source based on mode and date
            if self.mode == 'rollout' or use_val_data:
                # Use validation data for rollout mode or validation dates during training
                current_data = self.val_df.loc[current_date]
                current_prices = torch.tensor(
                    self.unscaled_close_val_df.loc[current_date]['Close'].values,
                    device=self.cpu_device,
                    dtype=torch.float32
                )
            else:
                # Use training data for normal training dates
                current_data = self.train_df.loc[current_date]
                current_prices = torch.tensor(
                    self.unscaled_close_train_df.loc[current_date]['Close'].values,
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

            # Calculate portfolio rebalancing needs
            total_portfolio_value = self.remaining_capital + (self.positions * current_prices).sum().item()

            # Calculate current allocation percentages
            current_position_values = self.positions * current_prices
            current_allocation = torch.zeros_like(allocation_percentages)
            for i in range(self.num_tickers):
                if total_portfolio_value > 0:
                    current_allocation[i] = current_position_values[i] / total_portfolio_value

            # Calculate rebalancing targets
            target_values = allocation_percentages * total_portfolio_value
            value_difference = target_values - current_position_values

            # Determine which tickers are underallocated and overallocated
            underallocated_mask = value_difference > 0
            overallocated_mask = value_difference < 0

            # Calculate rebalancing actions
            # 1. For underallocated assets with buy action, allocate more capital
            # 2. For overallocated assets with sell action, sell to reach target allocation

            # Calculate rebalancing buy amounts with improved logic
            rebalance_buy_amounts = torch.zeros_like(allocation_percentages)

            # First, handle tickers with no positions - highest priority
            zero_position_mask = self.positions < 1e-6
            for i in range(self.num_tickers):
                if zero_position_mask[i]:
                    # Always allocate for tickers with no positions, regardless of action
                    # This ensures portfolio diversification
                    min_buy_amount = current_prices[i] * 1.1  # Add 10% buffer for at least one share

                    # If it's a buy action, allocate more aggressively
                    if buy_masks[i]:
                        # Allocate more capital for buy actions on zero-position tickers
                        target_amount = max(
                            min_buy_amount,
                            self.remaining_capital * allocation_percentages[i] * 1.5  # 50% boost
                        )
                        rebalance_buy_amounts[i] = target_amount
                    else:
                        # Even for non-buy actions, ensure minimum allocation for diversification
                        # but with lower priority (only if we have significant capital)
                        if self.remaining_capital > self.initial_capital * 0.2:  # If we have >20% of initial capital
                            rebalance_buy_amounts[i] = min_buy_amount

            # Next, handle underallocated tickers
            for i in range(self.num_tickers):
                # Skip tickers with no positions (already handled)
                if zero_position_mask[i]:
                    continue

                # Handle underallocated tickers
                if underallocated_mask[i]:
                    # Calculate allocation gap
                    allocation_gap = (target_values[i] - current_position_values[i]) / total_portfolio_value

                    # For buy actions, boost allocation based on gap size
                    if buy_masks[i]:
                        # More aggressive boost for larger gaps
                        boost_factor = 1.0 + min(allocation_gap * 8.0, 1.0)  # Cap boost at 100%
                        rebalance_buy_amounts[i] = value_difference[i] * boost_factor
                    # For hold actions on severely underallocated tickers, still allocate some capital
                    elif hold_masks[i] and allocation_gap > 0.2:  # >20% underallocation
                        rebalance_buy_amounts[i] = value_difference[i] * 0.5  # Allocate 50% of the gap

            # For sell actions, we'll handle them separately in the execution loop
            # to ensure we're selling the right amount to reach target allocation

            # Add maximum allocation constraint to prevent extreme overallocation
            # Calculate maximum allocation percentage per ticker (e.g., 70%)
            max_allocation_pct = 0.7

            # Check if any ticker exceeds maximum allocation
            for i in range(self.num_tickers):
                current_allocation_pct = current_position_values[i] / total_portfolio_value if total_portfolio_value > 0 else 0

                # If a ticker is overallocated beyond the maximum and we're trying to buy more
                if current_allocation_pct > max_allocation_pct and buy_masks[i]:
                    # Prevent further buying by zeroing out the buy amount
                    rebalance_buy_amounts[i] = 0

                    if self.mode == 'rollout':
                        print(f"  Preventing further buying of {self.tickers[i]} due to maximum allocation constraint ({current_allocation_pct:.2%} > {max_allocation_pct:.2%})")

            # Calculate initial buy amounts with rebalancing consideration
            potential_buy_amounts = torch.where(
                buy_masks,
                torch.minimum(
                    torch.maximum(
                        # Standard allocation approach
                        self.remaining_capital * allocation_percentages * random_factor,
                        # Rebalancing approach (if applicable)
                        torch.where(rebalance_buy_amounts > 0, rebalance_buy_amounts, torch.zeros_like(rebalance_buy_amounts))
                    ),
                    # Cap at max initial allocation per ticker
                    torch.tensor(max_initial_allocation / self.num_tickers, device=self.cpu_device)
                ),
                torch.zeros_like(allocation_percentages)
            )

            # Debug rebalancing info in rollout mode
            '''
            if self.mode == 'rollout':
                print(f"  Portfolio rebalancing:")
                print(f"    Total portfolio value: ${total_portfolio_value:.2f}")
                print(f"    Current allocation: {current_allocation}")
                print(f"    Target allocation: {allocation_percentages}")
                print(f"    Value difference: {value_difference}")
                print(f"    Underallocated tickers: {underallocated_mask}")
                print(f"    Rebalance buy amounts: {rebalance_buy_amounts}")
            '''

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

            # Debug information
            '''
            if self.mode == 'rollout':
                print(f"\nStep {self.current_step} - Trading decisions:")
                print(f"  Remaining capital: ${self.remaining_capital:.2f}")
                print(f"  Invested capital: ${self.invested_capital:.2f}")
                print(f"  Current positions: {self.positions}")
                print(f"  Buy masks: {buy_masks}")
                print(f"  Sell masks: {sell_masks}")
                print(f"  Hold masks: {hold_masks}")
                print(f"  Allocation percentages: {allocation_percentages}")
                print(f"  Potential buy amounts: {potential_buy_amounts}")
            '''

            # Execute trades on CPU
            for i in range(self.num_tickers):
                if buy_masks[i]:
                    # Calculate shares to buy
                    shares_to_buy = torch.floor(potential_buy_amounts[i] / current_prices[i])

                    # Ensure we buy at least 1 share if we have enough capital and this is a mandated buy
                    if shares_to_buy < 1 and self.remaining_capital >= current_prices[i]:
                        # This is likely a mandated buy for a ticker with no position
                        # Force buying at least 1 share if we can afford it
                        shares_to_buy = torch.tensor(1.0, device=self.cpu_device)
                        if self.mode == 'rollout':
                            print(f"  Forcing minimum purchase of 1 share for {self.tickers[i]}")

                    cost = shares_to_buy * current_prices[i]

                    if cost <= self.remaining_capital and cost > 0:
                        self.positions[i] += shares_to_buy
                        self.remaining_capital -= cost
                        self.invested_capital += cost

                        # Debug information
                        '''
                        if self.mode == 'rollout':
                            print(f"  Buying {shares_to_buy:.0f} shares of {self.tickers[i]} at ${current_prices[i]:.2f} = ${cost:.2f}")
                        '''
                    else:
                        # Debug information
                        '''
                        if self.mode == 'rollout':
                            if cost <= 0:
                                print(f"  Attempted to buy {self.tickers[i]} but calculated shares to buy was 0")
                            else:
                                print(f"  Attempted to buy {self.tickers[i]} but insufficient capital (cost: ${cost:.2f}, remaining: ${self.remaining_capital:.2f})")
                        '''

                elif sell_masks[i]:
                    # Check if we have shares to sell
                    if self.positions[i] > 0:
                        # Determine how much to sell based on rebalancing needs
                        current_value = self.positions[i] * current_prices[i]
                        target_value = allocation_percentages[i] * total_portfolio_value

                        # Check if we need to preserve some minimum diversification
                        tickers_with_positions = (self.positions > 0).sum().item()
                        last_position = tickers_with_positions == 1 and self.positions[i] > 0

                        # If this is our last position and we have significant capital, don't sell everything
                        if last_position and self.remaining_capital < self.initial_capital * 0.5:
                            # We'll preserve at least 1 share for diversification

                            # If we're overallocated, sell only enough to reach target allocation
                            # Otherwise, sell to minimum preservation level
                            if current_value > target_value and value_difference[i] < 0:
                                # Calculate shares to sell for rebalancing
                                shares_to_sell = torch.floor(abs(value_difference[i]) / current_prices[i])

                                # Ensure we don't sell more than we have minus preservation amount
                                max_sellable = torch.maximum(
                                    self.positions[i] - torch.tensor(1.0, device=self.cpu_device),
                                    torch.tensor(0.0, device=self.cpu_device)
                                )
                                shares_to_sell = torch.minimum(shares_to_sell, max_sellable)

                                if self.mode == 'rollout' and max_sellable < self.positions[i]:
                                    print(f"  Preserving minimum position (1 share) in {self.tickers[i]} for diversification")
                            else:
                                # Sell down to minimum preservation level
                                shares_to_sell = torch.maximum(
                                    self.positions[i] - torch.tensor(1.0, device=self.cpu_device),
                                    torch.tensor(0.0, device=self.cpu_device)
                                )

                                if self.mode == 'rollout':
                                    print(f"  Preserving minimum position (1 share) in {self.tickers[i]} as it's our last position")
                        else:
                            # Normal selling logic (not the last position)
                            if current_value > target_value and value_difference[i] < 0:
                                # Calculate shares to sell for rebalancing
                                shares_to_sell = torch.floor(abs(value_difference[i]) / current_prices[i])

                                # Ensure we don't sell more than we have
                                shares_to_sell = torch.minimum(shares_to_sell, self.positions[i])
                            else:
                                # Sell all shares (traditional approach)
                                shares_to_sell = self.positions[i]

                        # Ensure we sell at least 1 share if we decided to sell and have shares
                        if shares_to_sell < 1 and self.positions[i] >= 1:
                            shares_to_sell = torch.tensor(1.0, device=self.cpu_device)

                        # Calculate sale proceeds
                        sale_proceeds = shares_to_sell * current_prices[i]

                        # Update positions and capital
                        self.positions[i] -= shares_to_sell
                        self.remaining_capital += sale_proceeds
                        # Ensure invested_capital doesn't go negative
                        self.invested_capital = max(0.0, self.invested_capital - sale_proceeds)

                        # Debug information
                        '''
                        if self.mode == 'rollout':
                            if shares_to_sell < self.positions[i] + shares_to_sell:  # Partial sell
                                print(f"  Rebalancing: Selling {shares_to_sell:.0f} shares of {self.tickers[i]} at ${current_prices[i]:.2f} = ${sale_proceeds:.2f}")
                                print(f"    Current value: ${current_value:.2f}, Target value: ${target_value:.2f}")
                                print(f"    Remaining shares: {self.positions[i]:.0f}")
                            else:  # Full sell
                                print(f"  Selling all {shares_to_sell:.0f} shares of {self.tickers[i]} at ${current_prices[i]:.2f} = ${sale_proceeds:.2f}")
                        '''
                    else:
                        # Debug information
                        '''
                        if self.mode == 'rollout':
                            print(f"  Attempted to sell {self.tickers[i]} but no shares owned")
                        '''

            # Calculate portfolio value and return
            portfolio_value = (self.positions * current_prices).sum() + self.remaining_capital
            portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value if self.prev_portfolio_value > 0 else 0

            # Calculate buy-and-hold performance if tracking is enabled
            market_outperformance_reward = 0.0
            if self.track_buy_and_hold and self.buy_and_hold_positions is not None:
                # Calculate current buy-and-hold value
                buy_and_hold_value = (self.buy_and_hold_positions * current_prices).sum()

                # Calculate buy-and-hold return
                buy_and_hold_return = (buy_and_hold_value - self.buy_and_hold_prev_value) / self.buy_and_hold_prev_value if self.buy_and_hold_prev_value > 0 else 0

                # Calculate outperformance (how much agent beat the market)
                outperformance = portfolio_return - buy_and_hold_return
                self.outperformance_history.append(outperformance)

                # Define outperformance multiplier (used later in reward calculation)
                outperformance_multiplier = 15.0  # 10x increase from 1.5 to drastically increase the weight of market outperformance

                # Drastically increased incentive for beating the market (10x boost)
                if outperformance > 0:
                    # Significantly increased exponential reward for outperformance (10x greater reward for outperformance)
                    # Increased scaling factor from 0.15 to 1.5 (10x increase)
                    market_outperformance_reward = 1.5 * (1.0 + torch.exp(torch.tensor(outperformance * 15)) - 1.0)

                    # Extra reward for consistent outperformance (10x increase from 0.05 to 0.5)
                    if len(self.outperformance_history) >= 3:
                        consecutive_outperformance = sum(1 for perf in list(self.outperformance_history)[-3:] if perf > 0)
                        if consecutive_outperformance >= 2:
                            # Drastically increased reward for consistency (higher for 3 consecutive than 2)
                            market_outperformance_reward += 0.5 * consecutive_outperformance
                else:
                    # Severe penalty for underperforming the market (10x increase in penalty)
                    # Use a sigmoid function to cap the penalty for severe underperformance
                    # Increased penalty factor from -0.05 to -0.5 (10x increase)
                    market_outperformance_reward = -0.5 * (1.0 / (1.0 + torch.exp(torch.tensor(-outperformance * 8))))

                # Update buy-and-hold previous value for next step
                self.buy_and_hold_prev_value = buy_and_hold_value

                # Debug info in rollout mode
                if self.mode == 'rollout':
                    print(f"  Market comparison:")
                    print(f"    Portfolio return: {portfolio_return:.4f}")
                    print(f"    Buy & Hold return: {buy_and_hold_return:.4f}")
                    print(f"    Outperformance: {outperformance:.4f}")
                    print(f"    Outperformance reward: {market_outperformance_reward:.4f} (with multiplier: {market_outperformance_reward * outperformance_multiplier:.4f})")

            # Store the current step before incrementing it
            current_step_before_increment = self.current_step

            # Update state
            self.prev_portfolio_value = portfolio_value
            self.current_step += 1

            # Calculate reward components on CPU
            base_reward = portfolio_return

            # Initialize market timing reward and future price changes
            market_timing_reward = 0.0
            future_price_changes = torch.zeros(self.num_tickers, device=self.cpu_device)

            # Only use look-ahead mechanism during training, not during evaluation or rollout
            if self.mode == 'train':
                # Get future price movements for market timing evaluation
                # Use the step before increment to ensure we're looking at the current date
                future_price_changes = self._get_future_price_movements(current_step_before_increment, window=self.look_ahead_window)

                # Calculate market timing rewards based on future price movements
                for i in range(self.num_tickers):
                    future_change = future_price_changes[i].item()

                    if sell_masks[i] and future_change < -0.01:  # Price will drop, good sell
                        # 10x increased reward for well-timed sell before price drop (higher reward for bigger drops)
                        timing_reward = 0.02 * abs(future_change) * self.market_timing_multiplier['correct_sell']
                        market_timing_reward += timing_reward

                    elif sell_masks[i] and future_change > 0.01:  # Price will rise, bad sell
                        # 10x increased penalty for poorly-timed sell before price rise (higher penalty for bigger rises)
                        timing_penalty = 0.02 * future_change * self.market_timing_multiplier['incorrect_sell']
                        market_timing_reward -= timing_penalty

                    elif buy_masks[i] and future_change > 0.01:  # Price will rise, good buy
                        # 10x increased reward for well-timed buy before price rise (higher reward for bigger rises)
                        timing_reward = 0.02 * future_change * self.market_timing_multiplier['correct_buy']
                        market_timing_reward += timing_reward

                    elif buy_masks[i] and future_change < -0.01:  # Price will drop, bad buy
                        # 10x increased penalty for poorly-timed buy before price drop (higher penalty for bigger drops)
                        timing_penalty = 0.02 * abs(future_change) * self.market_timing_multiplier['incorrect_buy']
                        market_timing_reward -= timing_penalty

            # In rollout mode, print information about what would have been the market timing rewards
            # This is for debugging only and doesn't affect the actual rewards
            if self.mode == 'rollout':
                debug_future_changes = self._get_future_price_movements(current_step_before_increment, window=self.look_ahead_window)
                print(f"  Market timing information (for debugging only, not used in rewards):")

                for i in range(self.num_tickers):
                    future_change = debug_future_changes[i].item()

                    if sell_masks[i] and future_change < -0.01:  # Price will drop, good sell
                        timing_reward = 0.02 * abs(future_change) * self.market_timing_multiplier['correct_sell']
                        print(f"    Well-timed SELL for {self.tickers[i]}: would add +{timing_reward:.6f} (future change: {future_change:.2%})")

                    elif sell_masks[i] and future_change > 0.01:  # Price will rise, bad sell
                        timing_penalty = 0.02 * future_change * self.market_timing_multiplier['incorrect_sell']
                        print(f"    Poorly-timed SELL for {self.tickers[i]}: would subtract -{timing_penalty:.6f} (future change: {future_change:.2%})")

                    elif buy_masks[i] and future_change > 0.01:  # Price will rise, good buy
                        timing_reward = 0.02 * future_change * self.market_timing_multiplier['correct_buy']
                        print(f"    Well-timed BUY for {self.tickers[i]}: would add +{timing_reward:.6f} (future change: {future_change:.2%})")

                    elif buy_masks[i] and future_change < -0.01:  # Price will drop, bad buy
                        timing_penalty = 0.02 * abs(future_change) * self.market_timing_multiplier['incorrect_buy']
                        print(f"    Poorly-timed BUY for {self.tickers[i]}: would subtract -{timing_penalty:.6f} (future change: {future_change:.2%})")

            # Calculate forecast alignment reward (keep existing logic but with reduced weight)
            forecast_alignment_reward = 0.0
            for i in range(self.num_tickers):
                if buy_masks[i] and forecast_returns[i] > 0:
                    # Reward for buying when forecast predicts increase
                    forecast_alignment_reward += 0.0005 * forecast_returns[i] * forecast_confidence[i]  # Reduced from 0.001
                elif sell_masks[i] and forecast_returns[i] < 0:
                    # Reward for selling when forecast predicts decrease
                    forecast_alignment_reward += 0.0005 * abs(forecast_returns[i]) * forecast_confidence[i]  # Reduced from 0.001
                elif hold_masks[i] and abs(forecast_returns[i]) < 0.001:
                    # Small reward for holding when forecast predicts stability
                    forecast_alignment_reward += 0.0001 * forecast_confidence[i]

            # Add momentum-based rewards (keep existing logic but with reduced weight)
            momentum_reward = 0.0
            for i in range(self.num_tickers):
                if buy_masks[i] and forecast_momentum[i] > 0:
                    momentum_reward += 0.0003 * forecast_momentum[i]  # Reduced from 0.0005
                elif sell_masks[i] and forecast_momentum[i] < 0:
                    momentum_reward += 0.0003 * abs(forecast_momentum[i])  # Reduced from 0.0005

            # Add position-based penalties and rewards
            total_positions = self.positions.sum()
            if total_positions < 1e-6:
                # Stronger penalty for having no positions - always apply this penalty regardless of outperformance
                # as having no positions is fundamentally against the goal of the agent
                base_reward -= 0.005  # Increased from 0.001

                # Add extra penalty if we're just repeatedly selling with no positions
                sell_count = sell_masks.sum().item()
                if sell_count > 0:
                    base_reward -= 0.002 * sell_count  # Penalty for trying to sell when no positions
            else:
                # Small reward for having positions (encourages buying)
                # This reward will be conditional on market outperformance
                position_reward = 0.001 * (total_positions / self.num_tickers)

                # Only add this reward if we're outperforming the market, otherwise heavily reduce it
                if self.track_buy_and_hold and self.buy_and_hold_positions is not None:
                    if outperformance > 0:
                        base_reward += position_reward
                    else:
                        # 95% reduction when underperforming
                        base_reward += position_reward * 0.05

                        if self.mode == 'rollout':
                            print(f"  Reduced position reward: {position_reward * 0.05:.6f} (original: {position_reward:.6f})")
                else:
                    # If not tracking buy-and-hold, add the reward as normal
                    base_reward += position_reward

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

                # Add penalties for invalid actions
                if sell_masks[i] and self.positions[i] <= 0:
                    # Stronger penalty for attempting to sell when no shares owned
                    misalignment_penalty += 0.05  # Increased from 0.01 to 0.05

                    # Track invalid sell attempts for this ticker
                    ticker_key = f"invalid_sell_{self.tickers[i]}"
                    if not hasattr(self, 'invalid_action_counts'):
                        self.invalid_action_counts = {}

                    self.invalid_action_counts[ticker_key] = self.invalid_action_counts.get(ticker_key, 0) + 1

                    # Escalating penalty for repeated invalid sells on the same ticker
                    repeat_count = self.invalid_action_counts.get(ticker_key, 0)
                    if repeat_count > 1:
                        misalignment_penalty += 0.02 * min(repeat_count, 5)  # Additional penalty, capped at 5x

                        if self.mode == 'rollout':
                            print(f"  Escalating penalty for repeated invalid sell on {self.tickers[i]} (attempt #{repeat_count})")

                # Add penalty for buying overallocated tickers
                current_allocation_pct = current_position_values[i] / total_portfolio_value if total_portfolio_value > 0 else 0
                if buy_masks[i] and current_allocation_pct > 0.7:  # Same threshold as max_allocation_pct
                    # Penalty for trying to buy already overallocated tickers
                    overallocation_penalty = 0.01 * (current_allocation_pct - 0.7) * 10  # Scales with degree of overallocation
                    misalignment_penalty += overallocation_penalty

                    if self.mode == 'rollout':
                        print(f"  Penalty for buying overallocated ticker {self.tickers[i]}: {overallocation_penalty:.4f} (allocation: {current_allocation_pct:.2%})")

            # Add reward for successful trades
            successful_buys = 0
            diversification_bonus = 0.0
            rebalancing_bonus = 0.0

            # Calculate portfolio balance metrics
            # 1. How well the portfolio matches the target allocation
            # 2. How diversified the portfolio is

            # Calculate allocation error (lower is better)
            if total_positions > 1e-6:
                # Calculate updated portfolio value and allocation after trades
                updated_position_values = self.positions * current_prices
                updated_portfolio_value = updated_position_values.sum() + self.remaining_capital

                if updated_portfolio_value > 0:
                    updated_allocation = torch.zeros_like(allocation_percentages)
                    for i in range(self.num_tickers):
                        updated_allocation[i] = updated_position_values[i] / updated_portfolio_value

                    # Calculate allocation error (mean absolute deviation from target)
                    allocation_error = torch.abs(updated_allocation - allocation_percentages).mean().item()

                    # Reward for good allocation (lower error)
                    if allocation_error < 0.1:  # Less than 10% average deviation
                        rebalancing_bonus = 0.003 * (1.0 - allocation_error * 10)  # Scale from 0.003 to 0

                        # Debug info
                        if self.mode == 'rollout':
                            print(f"  Rebalancing bonus: {rebalancing_bonus:.4f} (allocation error: {allocation_error:.4f})")

            for i in range(self.num_tickers):
                if buy_masks[i]:
                    shares_to_buy = torch.floor(potential_buy_amounts[i] / current_prices[i])
                    cost = shares_to_buy * current_prices[i]
                    if cost > 0 and cost <= self.remaining_capital:
                        successful_buys += 1

                        # Extra reward for buying a ticker with no position (encourages diversification)
                        if self.positions[i] < 1e-6:
                            # Calculate how many tickers currently have positions
                            tickers_with_positions = (self.positions > 0).sum().item()

                            # Higher bonus for first few positions (diminishing returns)
                            if tickers_with_positions == 0:
                                # First position gets highest bonus
                                diversification_bonus += 0.02
                            elif tickers_with_positions < self.num_tickers - 1:
                                # Subsequent positions get good but diminishing bonus
                                diversification_bonus += 0.01
                            else:
                                # Last position completes the portfolio
                                diversification_bonus += 0.015  # Triple bonus for completing the portfolio

                            # Debug info
                            if self.mode == 'rollout':
                                print(f"  Enhanced diversification bonus for buying {self.tickers[i]} with no previous position: {diversification_bonus:.4f}")
                                print(f"  Portfolio now has positions in {tickers_with_positions + 1}/{self.num_tickers} tickers")

            # Determine if we achieved market outperformance
            achieved_outperformance = False
            if self.track_buy_and_hold and self.buy_and_hold_positions is not None:
                achieved_outperformance = outperformance > 0

            # Reward for successful buys and portfolio management - ONLY if we achieved market outperformance
            if successful_buys > 0 or rebalancing_bonus > 0:
                if achieved_outperformance:
                    # Full bonuses when outperforming the market
                    base_reward += 0.002 * successful_buys + diversification_bonus + rebalancing_bonus

                    # Debug info
                    if self.mode == 'rollout':
                        if diversification_bonus > 0:
                            print(f"  Total diversification bonus: {diversification_bonus:.4f}")
                        if rebalancing_bonus > 0:
                            print(f"  Total rebalancing bonus: {rebalancing_bonus:.4f}")
                else:
                    # Heavily reduced bonuses when underperforming the market (95% reduction)
                    reduced_bonus = (0.002 * successful_buys + diversification_bonus + rebalancing_bonus) * 0.05
                    base_reward += reduced_bonus

                    # Debug info
                    if self.mode == 'rollout':
                        print(f"  Market underperformance detected - bonuses reduced by 95%")
                        if diversification_bonus > 0:
                            print(f"  Reduced diversification bonus: {diversification_bonus * 0.05:.4f} (original: {diversification_bonus:.4f})")
                        if rebalancing_bonus > 0:
                            print(f"  Reduced rebalancing bonus: {rebalancing_bonus * 0.05:.4f} (original: {rebalancing_bonus:.4f})")
                        if successful_buys > 0:
                            print(f"  Reduced successful buys bonus: {0.002 * successful_buys * 0.05:.4f} (original: {0.002 * successful_buys:.4f})")

            # Combine rewards focusing on market outperformance, base reward, and market timing
            # Increase outperformance multiplier even further when underperforming to create stronger incentive
            outperformance_factor = outperformance_multiplier
            if not achieved_outperformance:
                outperformance_factor *= 1.5  # 50% higher penalty for underperformance

            reward = base_reward + (market_outperformance_reward * outperformance_factor) - transaction_penalty

            # Add market timing rewards during training only, with conditional application
            if self.mode == 'train':
                # If we're outperforming the market, apply full market timing reward
                # If underperforming, still apply but at a reduced rate to maintain some learning signal
                if self.track_buy_and_hold and self.buy_and_hold_positions is not None:
                    if outperformance > 0:
                        # Full market timing reward when outperforming
                        reward += market_timing_reward
                    else:
                        # Reduced market timing reward when underperforming (50% reduction)
                        # We keep this higher than other bonuses because timing is still important
                        reward += market_timing_reward * 0.5

                        if self.mode == 'rollout':
                            print(f"  Reduced market timing reward: {market_timing_reward * 0.5:.6f} (original: {market_timing_reward:.6f})")
                else:
                    # If not tracking buy-and-hold, apply market timing reward as normal
                    reward += market_timing_reward

            # Debug reward components in rollout mode
            if self.mode == 'rollout':
                print(f"  Reward components (prioritizing market outperformance above all else):")
                print(f"    Base reward (portfolio return): {base_reward:.6f}")
                print(f"    Market outperformance: {'ACHIEVED' if achieved_outperformance else 'NOT ACHIEVED'}")
                print(f"    Market outperformance reward: {market_outperformance_reward:.6f} (with multiplier: {market_outperformance_reward * outperformance_factor:.6f})")
                print(f"    Transaction penalty: {transaction_penalty:.6f}")
                print(f"    Market timing reward: {market_timing_reward:.6f} (used only during training, reduced by 50% when underperforming)")
                print(f"    Total reward: {reward:.6f}")

                # Add explanation of the outperformance-based reward system
                if achieved_outperformance:
                    print(f"    [Full bonuses applied due to market outperformance]")
                else:
                    print(f"    [Most bonuses reduced by 95% due to market underperformance]")
                    print(f"    [Market timing rewards reduced by 50% due to market underperformance]")
                    print(f"    [Outperformance penalty increased by 50% to {outperformance_factor:.1f}x]")

                # Print other components for reference but note they're not used
                print(f"  Unused components (for reference only):")
                print(f"    Forecast alignment reward: {forecast_alignment_reward:.6f} (not used)")
                print(f"    Momentum reward: {momentum_reward:.6f} (not used)")
                print(f"    Misalignment penalty: {misalignment_penalty:.6f} (not used)")

            # Check if episode is done
            done = self.current_step >= len(self.dates) - 1

            # Get next observation
            next_state = self._get_observation()

            # Prepare info dict with the metrics we care about (market outperformance, portfolio return, transaction costs, and market timing)
            info = {
                'portfolio_value': portfolio_value,
                'portfolio_return': portfolio_return,
                'positions': self.positions.clone(),
                'market_outperformance_reward': market_outperformance_reward * outperformance_multiplier,  # Include the multiplied value
                'transaction_cost': transaction_penalty,  # Keep transaction costs
                'market_timing_reward': market_timing_reward,  # Include market timing reward
                'future_price_changes': future_price_changes.clone()
            }

            # Add buy-and-hold metrics if tracking is enabled
            if self.track_buy_and_hold and self.buy_and_hold_positions is not None:
                info.update({
                    'buy_and_hold_value': buy_and_hold_value,
                    'buy_and_hold_return': buy_and_hold_return,
                    'market_outperformance': outperformance,
                    'consecutive_outperformance': consecutive_outperformance if 'consecutive_outperformance' in locals() else 0
                })

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

    def get_rollout_observation(self, idx):
        """Get observation for rollout dataset at specific index"""
        original_mode = self.mode
        original_step = self.current_step
        self.mode = 'rollout'
        self.current_step = idx
        obs = self._get_observation()
        self.current_step = original_step
        self.mode = original_mode
        return obs

    def get_val_observation(self, idx):
        """Get observation for validation/final evaluation dataset at specific index"""
        original_mode = self.mode
        original_step = self.current_step
        self.mode = 'final_eval'  # Use final_eval mode for Jan-Mar 2025 dates
        self.current_step = idx
        obs = self._get_observation()
        self.current_step = original_step
        self.mode = original_mode
        return obs

    def set_mode(self, mode):
        """Switch between training, rollout, and final evaluation modes"""
        if mode not in ['train', 'rollout', 'final_eval']:
            raise ValueError("Mode must be either 'train', 'rollout', or 'final_eval'")

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

            # For training, use a combination of train_df and val_df for dates before October 2024
            # We'll use train_df as the base and handle any validation data in the step method
            self.data_df = self.train_df
            self.unscaled_close_df = self.unscaled_close_train_df

            print(f"Training mode activated with {len(self.pure_train_dates)} dates before October 2024")
        elif mode == 'rollout':
            self.dates = self.rollout_dates
            self.is_live_trading = False

            # For rollout mode, we need to use the validation dataframe since
            # the rollout dates (Oct, Nov, Dec 2024) come from the validation set
            self.data_df = self.val_df
            self.unscaled_close_df = self.unscaled_close_val_df

            print(f"Rollout mode activated with {len(self.rollout_dates)} dates from October-December 2024")
        elif mode == 'final_eval':
            self.dates = self.final_eval_dates
            self.is_live_trading = False

            # For final evaluation mode, we need to use the validation dataframe since
            # the final evaluation dates (Jan, Feb, Mar 2025) come from the validation set
            self.data_df = self.val_df
            self.unscaled_close_df = self.unscaled_close_val_df

            print(f"Final evaluation mode activated with {len(self.final_eval_dates)} dates from January-March 2025")

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

    def _calculate_correlation_penalty(self, returns, _):
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

    def get_current_prices(self):
        """Get current prices for all tickers at the current step"""
        if self.current_step < len(self.dates):
            current_date = self.dates[self.current_step]
            return self._get_prices_for_date(current_date).to(self.cpu_device)
        else:
            # Fallback to last available date
            last_date = self.dates[-1]
            return self._get_prices_for_date(last_date).to(self.cpu_device)

    def _get_prices_for_date(self, date):
        """Optimized price lookup with caching"""
        if date in self.history_cache['prices']:
            return self.history_cache['prices'][date]

        try:
            prices = self.unscaled_close_df.xs(date, level='Date')['Close'].values
            prices_tensor = torch.tensor(prices, dtype=torch.float32, device=self.gpu_device)
            self.history_cache['prices'][date] = prices_tensor
            return prices_tensor
        except KeyError:
            # If date lookup fails, find the closest available date in the index
            try:
                # Convert to integer index if it's a date object
                if isinstance(date, (np.datetime64, pd.Timestamp)):
                    # Find the closest date in the index
                    date_idx = -1
                    for i, d in enumerate(self.dates):
                        if d == date or str(d) == str(date):
                            date_idx = i
                            break

                    # If found in dates array but not in DataFrame index, use the previous available date
                    if date_idx >= 0:
                        # Try previous dates until we find one that exists in the DataFrame
                        for i in range(date_idx, -1, -1):
                            try_date = self.dates[i]
                            try:
                                prices = self.unscaled_close_df.xs(try_date, level='Date')['Close'].values
                                prices_tensor = torch.tensor(prices, dtype=torch.float32, device=self.gpu_device)
                                self.history_cache['prices'][date] = prices_tensor  # Cache with original date
                                return prices_tensor
                            except KeyError:
                                continue

                # If all else fails, use the first date in the DataFrame
                first_date = self.unscaled_close_df.index.get_level_values('Date')[0]
                prices = self.unscaled_close_df.xs(first_date, level='Date')['Close'].values
                prices_tensor = torch.tensor(prices, dtype=torch.float32, device=self.gpu_device)
                self.history_cache['prices'][date] = prices_tensor  # Cache with original date
                return prices_tensor
            except Exception as e:
                # Last resort: return zeros
                prices_tensor = torch.zeros(self.num_tickers, dtype=torch.float32, device=self.gpu_device)
                self.history_cache['prices'][date] = prices_tensor
                return prices_tensor

    def _get_future_price_movements(self, current_date_idx, window=5):
        """
        Get future price movements for evaluating trading decisions.
        Returns a tensor of price changes (in percentage) for each ticker over the next 'window' days.
        Positive values indicate price increases, negative values indicate price decreases.

        Args:
            current_date_idx: Either an integer index into self.dates or a datetime object
            window: Number of days to look ahead
        """
        try:
            # Handle case where current_date_idx is a datetime object
            if isinstance(current_date_idx, (np.datetime64, pd.Timestamp)):

                # Convert datetime to index
                date_str = str(current_date_idx)
                # Find the index of this date in self.dates
                for i, date in enumerate(self.dates):
                    if str(date) == date_str:
                        print(f"DEBUG: Found matching date at index {i}")
                        current_date_idx = i
                        break
                else:
                    # Date not found in self.dates
                    print(f"Warning: Date {date_str} not found in available dates")
                    return torch.zeros(self.num_tickers, device=self.cpu_device)

            # Ensure current_date_idx is an integer
            if not isinstance(current_date_idx, (int, np.integer)):
                print(f"Warning: Invalid current_date_idx type: {type(current_date_idx)}")
                # Try to convert to int as a last resort
                try:
                    current_date_idx = int(current_date_idx)
                except (ValueError, TypeError):
                    print(f"DEBUG: Failed to convert to int")
                    return torch.zeros(self.num_tickers, device=self.cpu_device)

            # Ensure we don't go beyond available dates
            max_idx = min(current_date_idx + window, len(self.dates) - 1)

            if current_date_idx >= len(self.dates) - 1:
                # No future data available, return zeros
                return torch.zeros(self.num_tickers, device=self.cpu_device)

            # Get current prices
            current_date = self.dates[current_date_idx]
            current_prices = self._get_prices_for_date(current_date).to(self.cpu_device)

            # Get future prices
            future_date = self.dates[max_idx]
            future_prices = self._get_prices_for_date(future_date).to(self.cpu_device)

            # Calculate percentage changes
            price_changes = (future_prices - current_prices) / current_prices

            return price_changes

        except Exception as e:
            print(f"Error in _get_future_price_movements: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(self.num_tickers, device=self.cpu_device)

    def _precompute_batch_data(self, start_idx, end_idx):
        """Pre-compute and cache historical data for batch processing"""
        dates = self.dates[start_idx:end_idx]

        # Batch compute historical prices and returns
        prices_batch = []

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
            # prev_date = dates[i-1]  # Not used but kept for reference

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


