import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import gc
import torch.optim as optim

# component imports
from buffer import PrioritizedReplayBuffer
from meta_learner import MetaStrategyLearner, StrategyBank, MarketContextEncoder, MarketRegimeDetector, StrategyMemory
from fin_dataset import FinancialDataset, TemporalDataLoader

from actor import Actor
from critic import Critic

class PortfolioAgent:
    def __init__(self, env, buffer_size=10000, train_batch_size=64, eval_batch_size=32, num_workers=4):
        # Initialize environment and basic parameters
        self.env = env
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        # Data loading parameters
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()
        
        # Define style names
        self.style_names = ['aggressive', 'moderate', 'conservative']
        
        # Network parameters - calculate sizes based on environment
        self.state_size = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.total_features
        self.action_size = env.num_tickers * 3  # 3 actions (buy, sell, hold) per ticker
        self.hidden_size = 256
        self.num_layers = 2
        
        # Set batch sizes from parameters
        self.batch_size = train_batch_size  # Main batch size for training
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        print(f"Initializing PortfolioAgent with:")
        print(f"State size: {self.state_size}")
        print(f"Action size: {self.action_size}")
        print(f"Number of tickers: {env.num_tickers}")
        print(f"Features per ticker: {env.num_features_per_ticker}")
        print(f"Training batch size: {self.train_batch_size}")
        print(f"Evaluation batch size: {self.eval_batch_size}")
        
        # Initialize networks with optimized architecture
        self.actor = Actor(
            env=env,
            hidden_size=self.hidden_size,
            embedding_dim=64
        ).to(self.gpu_device)
        
        self.critic = Critic(
            env=env,
            hidden_size=self.hidden_size,
            embedding_dim=64
        ).to(self.gpu_device)
        
        self.target_critic = Critic(
            env=env,
            hidden_size=self.hidden_size,
            embedding_dim=64
        ).to(self.gpu_device)
        
        # Copy weights to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers with learning rate scheduling
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Learning parameters
        self.gamma = 0.99
        self.tau = 0.001
        self.update_every = 1
        self.t_step = 0
        
        # Exploration parameters with adaptive bounds
        self.exploration_noise = 0.1
        self.exploration_temp = 1.0
        self.risk_preference = 0.5
        self.exploration_bounds = (0.05, 0.5)  # Tighter bounds
        self.exploration_temp_bounds = (0.5, 2.5)  # Wider range
        self.risk_bounds = (0.1, 1.5)  # Wider range for more aggressive possibilities
        self.lr_bounds = {
            'actor': (1e-5, 2e-3),  # Wider range
            'critic': (1e-5, 2e-3)
        }
        
        # Initialize market context
        self.market_context = {
            'current_regime': 'neutral_medium_vol',  # Default regime
            'regime_history': deque(maxlen=100),
            'returns_history': deque(maxlen=100),
            'prices_history': deque(maxlen=100),
            'volatility_history': deque(maxlen=100),
            'correlation_history': deque(maxlen=100)
        }
        
        # Initialize style performance tracking
        self.style_performance = {}
        for style in ['aggressive', 'moderate', 'conservative']:
            self.style_performance[style] = {
                'total': 0,
                'successes': 0,
                'score_sum': 0.0,
                'regime_performance': {},
                'training_metrics': {
                    'returns': deque(maxlen=100),
                    'sharpe': deque(maxlen=100),
                    'drawdown': deque(maxlen=50),
                    'learning_progress': deque(maxlen=100),
                    'adaptation_count': 0
                }
            }
        
        # Initialize meta-learning components
        self.meta_memory = deque(maxlen=10000)
        self.meta_batch_size = 32
        self.meta_update_freq = 10
        self.min_meta_samples = 100
        self.current_style = 'moderate'  # Start with moderate style
        self.style_eval_threshold = 0.005
        self.min_style_prob = 0.1
        
        # Initialize meta-learner and market encoder with CORRECT dimensions
        input_dim = 17  # 12 (vol) + 5 (strategy metrics)
        self.meta_learner = MetaStrategyLearner(
            input_dim=input_dim,
            hidden_dim=256,
            num_styles=3
        ).to(self.gpu_device)
        
        self.market_encoder = MarketContextEncoder(
            input_dim=8,  # ONLY market features
            hidden_dim=256
        ).to(self.gpu_device)
        
        # Initialize strategy bank
        self.strategy_bank = StrategyBank(num_strategies=3)
        
        # Initialize strategy memory
        self.strategy_memory = StrategyMemory(capacity=50)
        
        # Initialize meta optimizer
        self.meta_optimizer = torch.optim.Adam(
            list(self.meta_learner.parameters()) + 
            list(self.market_encoder.parameters()),
            lr=1e-4
        )
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Initialize adaptive parameters
        self.adaptive_params = {
            'base_exploration_noise': 0.1,
            'base_exploration_temp': 1.0,
            'base_risk_preference': 0.5,
            'vol_scaling_factor': 1.0,
            'trend_scaling_factor': 1.0,
            'correlation_scaling_factor': 1.0
        }
        
        # Initialize adaptation bounds
        self.adaptation_bounds = {
            'vol_scaling': (0.5, 2.0),
            'trend_scaling': (0.5, 2.0),
            'correlation_scaling': (0.5, 2.0)
        }
        
        # Initialize parameter history
        self.parameter_history = []
        
        # Initialize experience replay buffer with prioritization
        self.buffer_size_bounds = (5000, 50000)
        self.initial_buffer_size = buffer_size
        self.current_buffer_size = buffer_size
        self.buffer_adjustment_rate = 1.2
        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=0.6,
            beta=0.4
        )
        
        # Training state
        self.mode = 'train'
        self.training = True
        self.eval_history = deque(maxlen=20)
        self.returns_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=100)
        self.allocation_history = deque(maxlen=5)
        
        # Cache settings
        self.cache_update_frequency = 10
        self.batch_history_size = 100
        self.history_cache = {}

        # Initialize meta-learning buffers properly
        self.current_episode_meta_experiences = []
        self.episode_in_progress = False
        self.min_episode_experiences = 5  # Minimum experiences needed to consider an episode valid

        # Add data loader parameters
        self.sequence_length = 5
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.shuffle_within_sequences = True
        
        # Initialize data loaders
        self._initialize_data_loaders()

    def _initialize_data_loaders(self):
        """Initialize data loaders for different modes with multi-worker support"""
        # Create datasets
        self.train_dataset = FinancialDataset(
            env=self.env,
            mode='train',
            sequence_length=self.sequence_length,
            batch_size=self.train_batch_size,
            shuffle_within_sequences=self.shuffle_within_sequences
        )
        
        self.eval_dataset = FinancialDataset(
            env=self.env,
            mode='eval',
            sequence_length=self.sequence_length,
            batch_size=self.eval_batch_size,
            shuffle_within_sequences=False
        )
        
        self.rollout_dataset = FinancialDataset(
            env=self.env,
            mode='rollout',
            sequence_length=self.sequence_length,
            batch_size=self.eval_batch_size,
            shuffle_within_sequences=False
        )
        
        # Create data loaders with multi-worker support
        self.train_loader = TemporalDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        self.eval_loader = TemporalDataLoader(
            dataset=self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        self.rollout_loader = TemporalDataLoader(
            dataset=self.rollout_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        print(f"\nInitialized data loaders with:")
        print(f"Number of workers: {self.num_workers}")
        print(f"Pin memory: {self.pin_memory}")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Eval dataset size: {len(self.eval_dataset)}")
        print(f"Rollout dataset size: {len(self.rollout_dataset)}")

    def _clear_caches(self):
        """Clear various caches to prevent memory issues"""
        # Clear history caches
        self.history_cache = {
            'volatility': {},
            'returns': {},
            'prices': {},
            'metrics': {}
        }
        
        # Clear Python's garbage collector
        import gc
        gc.collect()
        
        # If CUDA is available, perform additional GPU memory cleanup
        if torch.cuda.is_available():
            # Force garbage collection on CUDA memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Optional: Reset peak memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
        
        # Clear any cached tensors in the networks
        for module in [self.actor, self.critic, self.target_critic]:
            for param in module.parameters():
                if hasattr(param, 'grad'):
                    param.grad = None
        
        # Clear optimizer states if needed
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        
        # Clear allocation history
        if hasattr(self, 'allocation_history'):
            self.allocation_history.clear()

    def _update_history_cache(self, current_step):
        """Pre-compute and cache historical data for batch processing"""
        if current_step % self.cache_update_frequency != 0:
            return
        
        start_idx = max(0, current_step - self.batch_history_size)
        dates = self.env.dates[start_idx:current_step + 1]
        
        # Batch compute historical prices and returns
        prices_batch = []
        returns_batch = []
        
        for date in dates:
            prices = self.env._get_prices_for_date(date)
            prices_batch.append(prices)
            
            if len(prices_batch) > 1:
                returns = (prices_batch[-1] - prices_batch[-2]) / prices_batch[-2]
                returns_batch.append(returns)
        
        # Cache results
        self.history_cache['prices'].update({
            date: prices for date, prices in zip(dates, prices_batch)
        })
        
        if returns_batch:
            self.history_cache['returns'].update({
                date: returns for date, returns in zip(dates[1:], returns_batch)
            })
            
            # Pre-compute volatilities for different windows
            for window in [5, 10, 20]:
                if len(returns_batch) >= window:
                    returns_tensor = torch.stack(returns_batch[-window:])
                    volatility = returns_tensor.std(dim=0)
                    self.history_cache['volatility'][(dates[-1], window)] = volatility
    
    def _batch_process_experiences(self, batch):
        """Process multiple experiences efficiently"""
        try:
            # Unpack batch
            states, actions, rewards, next_states, dones = batch
            
            # Ensure all inputs are on GPU
            states = states.to(self.gpu_device)
            next_states = next_states.to(self.gpu_device)
            if isinstance(actions, tuple):
                actions = (actions[0].to(self.gpu_device), actions[1].to(self.gpu_device))
            else:
                actions = actions.to(self.gpu_device)
            rewards = rewards.to(self.gpu_device)
            if isinstance(dones, torch.Tensor):
                dones = dones.to(self.gpu_device)
            
            # Get expected feature dimension from environment
            expected_features = self.env.total_features
            
            # Ensure consistent dimensions and feature size
            if states.dim() == 1:
                states = states.unsqueeze(0)
            if next_states.dim() == 1:
                next_states = next_states.unsqueeze(0)
                
            # Ensure correct feature dimension
            if states.size(-1) != expected_features:
                if states.size(-1) > expected_features:
                    states = states[..., :expected_features]
                else:
                    pad_size = expected_features - states.size(-1)
                    states = F.pad(states, (0, pad_size))
                    
            if next_states.size(-1) != expected_features:
                if next_states.size(-1) > expected_features:
                    next_states = next_states[..., :expected_features]
                else:
                    pad_size = expected_features - next_states.size(-1)
                    next_states = F.pad(next_states, (0, pad_size))
            
            # Pre-compute all critic values in one pass
            with torch.no_grad():
                # Ensure both tensors have the same shape before concatenating
                if states.size() != next_states.size():
                    # Reshape tensors to match
                    target_shape = (-1, expected_features)
                    states = states.view(target_shape)
                    next_states = next_states.view(target_shape)
                
                # Concatenate and compute in chunks if batch is large
                combined_states = torch.cat([states, next_states], dim=0)
                chunk_size = 128  # Adjust based on your GPU memory
                chunks = combined_states.split(chunk_size)
                
                all_critic_values = []
                for chunk in chunks:
                    chunk_values = self.critic(chunk.to(self.gpu_device))
                    all_critic_values.append(chunk_values)
                
                all_critic_values = torch.cat(all_critic_values, dim=0)
                current_values, next_values = torch.split(all_critic_values, [states.size(0), next_states.size(0)])
                
                # Free memory
                del all_critic_values, combined_states
                torch.cuda.empty_cache()
            
            # Pre-compute all actor outputs in one pass
            with torch.no_grad():
                # Process in chunks again
                actor_outputs = []
                for chunk in chunks:
                    discrete_probs, allocation_probs = self.actor(chunk.to(self.gpu_device))
                    actor_outputs.append((discrete_probs, allocation_probs))
                
                # Combine chunks
                discrete_probs = torch.cat([out[0] for out in actor_outputs], dim=0)
                allocation_probs = torch.cat([out[1] for out in actor_outputs], dim=0)
                
                # Split for current and next states
                current_probs = (
                    discrete_probs[:states.size(0)],
                    allocation_probs[:states.size(0)]
                )  # Close the parenthesis
                next_probs = (
                    discrete_probs[states.size(0):],
                    allocation_probs[states.size(0):]
                )
                
                # Free memory
                del discrete_probs, allocation_probs, actor_outputs
                torch.cuda.empty_cache()
            
            # Increment step counter
            self.t_step += 1
            
            return {
                'current_values': current_values,
                'next_values': next_values,
                'current_probs': current_probs,
                'next_probs': next_probs
            }
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            self._clear_caches()  # Emergency cleanup
            return None

    def learn(self):
        """Enhanced learning with batch processing and meta-learning integration"""
        if len(self.memory) < self.batch_size:
            return
        
        # Clear caches periodically during learning
        if self.t_step % self.cache_update_frequency == 0:
            self._clear_caches()

        # Sample batch with importance sampling weights
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, device=self.gpu_device)
        
        # Ensure all tensors are on GPU
        if not isinstance(actions, tuple):
            actions = actions.to(self.gpu_device)
        else:
            actions = (actions[0].to(self.gpu_device), actions[1].to(self.gpu_device))
        
        rewards = rewards.to(self.gpu_device)
        weights = weights.to(self.gpu_device)
        
        # Batch process all experiences
        batch_results = self._batch_process_experiences((states, actions, rewards, next_states, dones))
        
        if batch_results is None:
            print("Warning: Batch processing failed, skipping learning step")
            return
        
        # Calculate episode return from rewards for tracking
        episode_return = torch.mean(rewards).item()
        
        # Track the episode return for use in adapt_parameters
        if not hasattr(self, 'current_episode_returns'):
            self.current_episode_returns = []
        self.current_episode_returns.append(episode_return)
        
        # Increment step counter
        self.t_step += 1
        
        # Return batch results for further processing if needed
        return batch_results

    def adapt_parameters(self, is_episode_start=False):
        """Adapt agent parameters based on market context and meta-learning"""
        # If this is episode start and we had a previous episode, end it
        if is_episode_start and self.episode_in_progress:
            self.end_episode()
        
        # Create market features
        market_features = self._create_market_features()
        
        # Get current regime encoding and probabilities
        with torch.no_grad():
            regime_encoding, regime_probs = self.market_encoder(market_features)
            
            # Get strategy performance metrics
            strategy_metrics = self._get_strategy_metrics()  # Shape: [1, 9]
            
            # Get style probabilities from meta-learner
            style_weights = self.meta_learner(market_features, strategy_metrics)
            
            # Take first batch element if batched
            if style_weights.dim() == 2:
                style_weights = style_weights[0]
            
            # Select style based on probabilities
            selected_style = torch.argmax(style_weights).item()
            
            # Ensure selected_style is valid
            selected_style = min(max(0, selected_style), 2)  # Ensure in range [0, 2]
            self.current_style = self.style_names[selected_style]
        
        # Get strategy parameters with regime awareness
        strategy_params = self.strategy_bank.strategies[selected_style](regime_encoding)
        
        # Apply strategy parameters with training-specific adjustments
        if self.mode == 'train':
            # During training, allow for parameter exploration
            self._apply_strategy_params(strategy_params, allow_exploration=True)
            
            # Only update meta-learning at episode boundaries
            if is_episode_start:
                # Calculate average episode return if we have returns
                if hasattr(self, 'current_episode_returns') and self.current_episode_returns:
                    avg_episode_return = np.mean(self.current_episode_returns)
                    
                    # Get previous best score for current regime
                    current_regime = self.market_context['current_regime']
                    regime_stats = self.style_performance[self.current_style]['regime_performance'].get(current_regime, {})
                    prev_best = max(
                        [s['score'] for s in regime_stats.get('adaptation_history', [])]
                        if regime_stats.get('adaptation_history')
                        else [float('-inf')]
                    )
                    
                    # Update style performance
                    self._update_style_performance(self.current_style, avg_episode_return, prev_best)
                    
                    # Perform meta-learning update if we have enough samples
                    if len(self.meta_memory) >= self.min_meta_samples:
                        self._meta_learning_step()
                    
                    # Reset episode returns
                    self.current_episode_returns = []
        else:
            # During evaluation, use deterministic parameters
            self._apply_strategy_params(strategy_params, allow_exploration=False)
        
        # Update adaptation history
        if is_episode_start:
            self.parameter_history.append({
                'episode': len(self.eval_history) if hasattr(self, 'eval_history') else 0,
                'style': self.current_style,
                'market_regime': self.market_context['current_regime'],
                'style_weights': style_weights.detach().cpu().numpy(),
                'regime_probs': regime_probs.cpu().numpy(),
                'exploration_noise': float(self.exploration_noise),
                'exploration_temp': float(self.exploration_temp),
                'risk_preference': float(self.risk_preference),
                'learning_rate': float(self.actor_optimizer.param_groups[0]['lr']),
                'training_mode': self.mode == 'train'
            })
    
    def _soft_update(self, local_model, target_model, tau=0.001):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    

    def _get_grad_norm(self, model):
        """Calculate gradient norm for the model"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _process_rewards(self, rewards):
        """Apply asymmetric scaling to rewards based on portfolio state and risk"""
        # Get current portfolio state
        total_value = self.env.invested_capital + self.env.remaining_capital
        initial_value = self.env.initial_capital
        
        # Calculate portfolio performance metrics
        portfolio_return = (total_value - initial_value) / initial_value
        
        # Update returns and volatility history
        self.returns_history.append(portfolio_return)
        if len(self.returns_history) > 1:
            # Calculate rolling volatility using standard deviation of returns
            returns_array = np.array(list(self.returns_history))
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility
            self.volatility_history.append(volatility)
        
        # Dynamic scaling factors based on portfolio state with proper constraints
        if portfolio_return >= 0:
            # For positive portfolio performance, scale up gains with constraints
            gain_scale = self.gain_multiplier * (1.0 + torch.clamp(torch.tensor(portfolio_return), 0.0, 1.0))
            loss_scale = self.loss_multiplier
        else:
            # For negative portfolio performance, reduce loss scaling to encourage recovery
            gain_scale = self.gain_multiplier
            loss_scale = self.loss_multiplier * (1.0 - torch.clamp(torch.tensor(abs(portfolio_return)), 0.0, 0.5))
        
        # Apply asymmetric scaling with risk adjustment and constraints
        processed_rewards = torch.where(
            rewards > 0,
            rewards * torch.clamp(gain_scale * (1.0 + self.risk_preference * 0.1), 0.1, 5.0),
            rewards * torch.clamp(loss_scale * (1.0 - self.risk_preference * 0.05), 0.1, 2.0)
        )
        
        return processed_rewards
    
    def update_gamma(self, returns, eval_score=None):
        """Dynamically adjust gamma based on returns and evaluation performance"""
        if len(self.returns_history) > 0:
            # Calculate return statistics
            avg_return = np.mean(self.returns_history)
            return_volatility = np.std(self.returns_history)
            
            # Store new statistics
            self.returns_history.append(returns)
            self.volatility_history.append(return_volatility)
            
            # Adjust gamma based on return characteristics
            if return_volatility > np.mean(self.volatility_history):
                # Higher volatility -> lower gamma (more focus on immediate rewards)
                self.gamma = max(self.gamma_start, self.gamma * 0.995)
            else:
                # Lower volatility -> higher gamma (more focus on long-term rewards)
                self.gamma = min(self.gamma_end, self.gamma * 1.005)
            
            # Additional adjustment based on eval score if provided
            if eval_score is not None:
                if eval_score < 0:
                    # Poor performance -> reduce gamma to focus on immediate improvements
                    self.gamma = max(self.gamma_start, self.gamma * 0.99)
                else:
                    # Good performance -> increase gamma for long-term optimization
                    self.gamma = min(self.gamma_end, self.gamma * 1.01)
    

    def _calculate_regime_metrics(self):
        """Calculate comprehensive regime-specific performance metrics."""
        current_regime = self.market_context['current_regime']
        
        # Get recent history for the current regime
        regime_returns = []
        regime_values = []
        
        # Use last 20 entries for metrics calculation
        for i in range(min(20, len(self.market_context['returns_history']))):
            if self.market_context['regime_history'][-i-1] == current_regime:
                regime_returns.append(self.market_context['returns_history'][-i-1])
                regime_values.append(self.market_context['prices_history'][-i-1])
        
        if not regime_returns:
            return None
        
        regime_returns = np.array(regime_returns)
        regime_values = np.array(regime_values)
        
        # Calculate regime-specific metrics
        metrics = {
            'mean_return': float(np.mean(regime_returns)),
            'volatility': float(np.std(regime_returns) * np.sqrt(252)),
            'sharpe': float(np.sqrt(252) * (np.mean(regime_returns) / (np.std(regime_returns) + 1e-6))),
            'max_drawdown': 0.0
        }
        
        # Calculate max drawdown if we have values
        if len(regime_values) > 1:
            peak = np.maximum.accumulate(regime_values)
            drawdown = (peak - regime_values) / peak
            metrics['max_drawdown'] = float(np.max(drawdown))
        
        return metrics

    def _force_exploration_with_regime(self):
        """Force exploration phase with regime-specific adjustments."""
        current_regime = self.market_context['current_regime']
        regime_multiplier = self.get_regime_multiplier()
        
        print(f"\nForcing exploration phase in {current_regime} regime")
        print(f"Regime multiplier: {regime_multiplier:.2f}")
        
        # Adjust exploration parameters based on regime
        if 'bear' in current_regime:
            # More conservative exploration in bear markets
            noise_scale = 1.2
            temp_scale = 1.1
            lr_scale = 1.3
        else:
            noise_scale = 1.5
            temp_scale = 1.3
            lr_scale = 1.5
        
        self.exploration_noise = min(
            self.exploration_bounds[1],
            self.exploration_noise * noise_scale * regime_multiplier
        )
        self.exploration_temp = min(
            self.exploration_temp_bounds[1],
            self.exploration_temp * temp_scale * regime_multiplier
        )
        
        # Adjust risk preference based on regime volatility and trend
        vol_factor = self.adaptive_params['vol_scaling_factor']
        trend_factor = self.adaptive_params['trend_scaling_factor']
        
        if 'bear' in current_regime:
            if vol_factor > 1.2:  # High volatility bear market
                self.risk_preference = max(self.risk_bounds[0], self.risk_preference * 0.6)
            else:
                self.risk_preference = max(self.risk_bounds[0], self.risk_preference * 0.8)
        else:
            if trend_factor > 1.2:  # Strong trend
                self.risk_preference = min(self.risk_bounds[1], self.risk_preference * 1.3)
            else:
                self.risk_preference = np.random.uniform(self.risk_bounds[0], self.risk_bounds[1])
        
        # Adjust learning rates based on regime
        new_lr = min(
            self.lr_bounds['actor'][1],
            self.actor_optimizer.param_groups[0]['lr'] * lr_scale * regime_multiplier
        )
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr
        
        print(f"Forced exploration parameters:")
        print(f"  Noise: {self.exploration_noise:.3f}")
        print(f"  Temperature: {self.exploration_temp:.3f}")
        print(f"  Risk: {self.risk_preference:.3f}")
        print(f"  Learning Rate: {new_lr:.6f}")

    def _adjust_parameters_aggressively(self, regime_multiplier):
        """Aggressive parameter adjustment with randomization"""
        print("\nApplying aggressive parameter adjustment")
        
        base_params = self.style_base_params['aggressive']
        
        # Add random perturbation to regime multiplier
        noise = np.random.normal(0, 0.2)
        adjusted_multiplier = regime_multiplier * (1.0 + noise)
        
        # Start from aggressive base values and scale up
        noise_scale = np.random.uniform(1.5, 2.0)
        self.exploration_noise = min(
            self.exploration_bounds[1],
            base_params['exploration_noise'] * noise_scale * adjusted_multiplier
        )
        
        temp_scale = np.random.uniform(1.4, 1.8)
        self.exploration_temp = min(
            self.exploration_temp_bounds[1],
            base_params['exploration_temp'] * temp_scale * adjusted_multiplier
        )
        
        # Risk preference with stronger regime awareness
        if 'bear' in self.market_context['current_regime']:
            risk_scale = np.random.uniform(0.6, 0.8)  # More cautious in bear markets
        else:
            risk_scale = np.random.uniform(1.2, 1.5)  # Very aggressive in other markets
        self.risk_preference = np.clip(
            base_params['risk_preference'] * risk_scale,
            self.risk_bounds[0],
            self.risk_bounds[1]
        )
        
        # Higher learning rate for faster adaptation
        lr_scale = np.random.uniform(1.5, 2.0)
        new_lr = min(
            self.lr_bounds['actor'][1],
            base_params['learning_rate'] * lr_scale * adjusted_multiplier
        )
        
        # Apply learning rate changes
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr * 1.2  # Slightly higher for critic
        
        print(f"Aggressive Adjustment Results:")
        print(f"  Noise: {self.exploration_noise:.3f} (scale: {noise_scale:.2f})")
        print(f"  Temperature: {self.exploration_temp:.3f} (scale: {temp_scale:.2f})")
        print(f"  Risk: {self.risk_preference:.3f} (scale: {risk_scale:.2f})")
        print(f"  Learning Rate: {new_lr:.6f} (scale: {lr_scale:.2f})")

    def _adjust_parameters_conservatively(self, regime_multiplier):
        """Conservative parameter adjustment with randomization"""
        print("\nApplying conservative parameter adjustment")
        
        base_params = self.style_base_params['conservative']
        
        # Smaller random perturbation
        noise = np.random.normal(0, 0.1)
        adjusted_multiplier = regime_multiplier * (1.0 + noise)
        
        # Start from conservative base values and scale down
        noise_scale = np.random.uniform(0.3, 0.5)
        self.exploration_noise = max(
            self.exploration_bounds[0],
            base_params['exploration_noise'] * noise_scale * adjusted_multiplier
        )
        
        temp_scale = np.random.uniform(0.4, 0.6)
        self.exploration_temp = max(
            self.exploration_temp_bounds[0],
            base_params['exploration_temp'] * temp_scale * adjusted_multiplier
        )
        
        # Much more conservative risk preference
        if 'bear' in self.market_context['current_regime']:
            risk_scale = np.random.uniform(0.2, 0.4)  # Extremely conservative in bear markets
        else:
            risk_scale = np.random.uniform(0.5, 0.7)  # Still conservative in other markets
        self.risk_preference = np.clip(
            base_params['risk_preference'] * risk_scale,
            self.risk_bounds[0],
            self.risk_bounds[1]
        )
        
        # Lower learning rate for stability
        lr_scale = np.random.uniform(0.3, 0.5)
        new_lr = max(
            self.lr_bounds['actor'][0],
            base_params['learning_rate'] * lr_scale * adjusted_multiplier
        )
        
        # Apply learning rate changes
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr * 0.8  # Lower for critic
        
        print(f"Conservative Adjustment Results:")
        print(f"  Noise: {self.exploration_noise:.3f} (scale: {noise_scale:.2f})")
        print(f"  Temperature: {self.exploration_temp:.3f} (scale: {temp_scale:.2f})")
        print(f"  Risk: {self.risk_preference:.3f} (scale: {risk_scale:.2f})")
        print(f"  Learning Rate: {new_lr:.6f} (scale: {lr_scale:.2f})")

    def _adjust_parameters_moderately(self, regime_multiplier):
        """Moderate parameter adjustment with randomization"""
        print("\nApplying moderate parameter adjustment")
        
        base_params = self.style_base_params['moderate']
        
        # Moderate random perturbation
        noise = np.random.normal(0, 0.15)
        adjusted_multiplier = regime_multiplier * (1.0 + noise)
        
        # Start from moderate base values and adjust around them
        noise_scale = np.random.uniform(0.8, 1.2)
        self.exploration_noise = np.clip(
            base_params['exploration_noise'] * noise_scale * adjusted_multiplier,
            self.exploration_bounds[0],
            self.exploration_bounds[1]
        )
        
        temp_scale = np.random.uniform(0.9, 1.1)
        self.exploration_temp = np.clip(
            base_params['exploration_temp'] * temp_scale * adjusted_multiplier,
            self.exploration_temp_bounds[0],
            self.exploration_temp_bounds[1]
        )
        
        # Balanced risk preference
        if 'bear' in self.market_context['current_regime']:
            risk_scale = np.random.uniform(0.5, 0.7)  # Moderately conservative in bear markets
        else:
            risk_scale = np.random.uniform(0.8, 1.2)  # Balanced in other markets
        self.risk_preference = np.clip(
            base_params['risk_preference'] * risk_scale,
            self.risk_bounds[0],
            self.risk_bounds[1]
        )
        
        # Balanced learning rate
        lr_scale = np.random.uniform(0.8, 1.2)
        new_lr = np.clip(
            base_params['learning_rate'] * lr_scale * adjusted_multiplier,
            self.lr_bounds['actor'][0],
            self.lr_bounds['actor'][1]
        )
        
        # Apply learning rate changes
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr  # Same for critic
        
        print(f"Moderate Adjustment Results:")
        print(f"  Noise: {self.exploration_noise:.3f} (scale: {noise_scale:.2f})")
        print(f"  Temperature: {self.exploration_temp:.3f} (scale: {temp_scale:.2f})")
        print(f"  Risk: {self.risk_preference:.3f} (scale: {risk_scale:.2f})")
        print(f"  Learning Rate: {new_lr:.6f} (scale: {lr_scale:.2f})")

    def store_strategy_for_regime(self, score, regime_metrics=None):
        """Store strategy with enhanced regime-specific performance tracking"""
        current_regime = self.market_context['current_regime']
        
        # Initialize regime-specific best scores if not exists
        if not hasattr(self, 'regime_best_scores'):
            self.regime_best_scores = {}
        
        # Get regime-specific scores and best score
        regime_scores = []
        if current_regime in self.strategy_memory.strategies:
            for strategy in self.strategy_memory.strategies[current_regime]:
                # Move score to CPU if it's a tensor
                if torch.is_tensor(strategy['score']):
                    regime_scores.append(strategy['score'].cpu().numpy())
                else:
                    regime_scores.append(strategy['score'])
        
        # Update regime's best score
        current_best = self.regime_best_scores.get(current_regime, float('-inf'))
        if score > current_best:
            self.regime_best_scores[current_regime] = score
        
        # Handle initial case when no previous scores exist
        if not regime_scores:  # This is the first strategy for this regime
            print(f"\nInitializing first strategy for {current_regime}")
            relative_improvement = 0.0
            regime_avg = score  # Use current score as initial benchmark
            self.regime_best_scores[current_regime] = score
        else:
            # Convert scores to numpy array after ensuring they're on CPU
            regime_scores_array = np.array(regime_scores)
            regime_avg = np.mean(regime_scores_array)
            best_regime_score = self.regime_best_scores[current_regime]
            
            # Calculate improvements
            avg_improvement = ((score - regime_avg) / (abs(regime_avg) + 1e-6))
            best_improvement = ((score - best_regime_score) / (abs(best_regime_score) + 1e-6))
            
            # Use the more optimistic improvement metric to encourage exploration
            relative_improvement = max(avg_improvement, best_improvement)
        
        regime_duration = len([r for r in self.market_context['regime_history'] if r == current_regime])
        
        # More dynamic threshold based on regime type and duration
        base_threshold = 0.05  # 5% improvement threshold
        if 'bear' in current_regime:
            base_threshold *= 0.8  # Lower threshold for bear markets
        if regime_duration > 20:
            base_threshold *= 0.9  # Lower threshold for persistent regimes
        
        # Determine if we should store the strategy
        should_store = (
            not regime_scores or  # Always store first strategy
            score > self.regime_best_scores[current_regime] or  # Store if it's the best score
            relative_improvement > base_threshold  # Store if significant improvement
        )
        
        if should_store:
            current_params = {
                'exploration_noise': self.exploration_noise,
                'exploration_temp': self.exploration_temp,
                'risk_preference': self.risk_preference,
                'learning_rate': self.actor_optimizer.param_groups[0]['lr'],
                'vol_scaling': self.adaptive_params['vol_scaling_factor'],
                'trend_scaling': self.adaptive_params['trend_scaling_factor'],
                'correlation_scaling': self.adaptive_params['correlation_scaling_factor'],
                'regime_duration': regime_duration,
                'relative_improvement': float(relative_improvement)
            }
            
            # Add regime-specific metrics if available
            if regime_metrics:
                current_params.update({
                    'regime_metrics': regime_metrics
                })
            
            self.strategy_memory.add_strategy(
                current_regime,
                current_params,
                score
            )
            
            print(f"\nStored new strategy for {current_regime}:")
            print(f"Score: {score:.4f}")
            if regime_scores:
                print(f"Previous Regime Avg: {regime_avg:.4f}")
                print(f"Previous Best Score: {self.regime_best_scores[current_regime]:.4f}")
                print(f"Relative Improvement: {relative_improvement:.2%}")
            else:
                print("First strategy for this regime")
            
            print(f"Regime Duration: {regime_duration}")
            if regime_metrics:
                print("Regime Metrics:")
                for key, value in regime_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
        else:
            print(f"\nStrategy for {current_regime} did not meet storage criteria:")
            print(f"Score: {score:.4f}")
            print(f"Previous Best Score: {self.regime_best_scores[current_regime]:.4f}")
            print(f"Regime Average: {regime_avg:.4f}")
            print(f"Relative Improvement: {relative_improvement:.2%}")
            print(f"Required Improvement: {base_threshold:.2%}")
            print(f"Regime Duration: {regime_duration}")

    def get_regime_multiplier(self):
        """Get adjustment multiplier based on current market regime."""
        regime = self.market_context['current_regime']
        
        # Enhanced multipliers for different regime components
        volatility_mult = {
            'low_vol': 0.7,    # Reduced from 0.8
            'medium_vol': 1.0,
            'high_vol': 1.3    # Increased from 1.2
        }
        
        trend_mult = {
            'strong_bear': 1.4,  # Increased from 1.3
            'bear': 1.2,       # Increased from 1.1
            'neutral': 1.0,
            'bull': 1.2,
            'strong_bull': 1.4
        }
        
        try:
            # Parse regime components with better error handling
            if '_' not in regime:
                print(f"Warning: Invalid regime format '{regime}', using default components")
                trend, vol = 'neutral', 'medium_vol'
            else:
                # Split and handle multi-component regimes
                components = regime.split('_')
                
                # Handle 'strong_bull' or 'strong_bear' as a single trend component
                if components[0] == 'strong':
                    trend = f"strong_{components[1]}"
                    vol = '_'.join(components[2:])  # Join remaining components for volatility
                else:
                    trend = components[0]
                    vol = '_'.join(components[1:])  # Join remaining components for volatility
                
                # Validate components
                if trend not in trend_mult:
                    print(f"Warning: Unknown trend component '{trend}', using 'neutral'")
                    trend = 'neutral'
                if vol not in volatility_mult:
                    print(f"Warning: Unknown volatility component '{vol}', using 'medium_vol'")
                    vol = 'medium_vol'
            
            # Get multipliers with detailed logging
            vol_multiplier = volatility_mult[vol]
            trend_multiplier = trend_mult[trend]
            
            # Add correlation adjustment
            if len(self.market_context['correlation_history']) > 0:
                recent_correlation = np.mean(list(self.market_context['correlation_history'])[-5:])
                correlation_mult = 1.0 + abs(recent_correlation) * 0.3  # Increased impact
            else:
                correlation_mult = 1.0
            
            # Combine multipliers with correlation impact
            combined_multiplier = vol_multiplier * trend_multiplier * correlation_mult
            
            print(f"\nRegime Multiplier Details:")
            print(f"Regime: {regime}")
            print(f"Trend Component: {trend} (multiplier: {trend_multiplier:.2f})")
            print(f"Volatility Component: {vol} (multiplier: {vol_multiplier:.2f})")
            print(f"Correlation Component: (multiplier: {correlation_mult:.2f})")
            print(f"Combined Multiplier: {combined_multiplier:.2f}")
            
            return combined_multiplier
            
        except Exception as e:
            print(f"\nWarning: Error processing regime '{regime}': {str(e)}")
            print("Using neutral multiplier (1.0)")
            print(f"Current Market Context:")
            print(f"- Current Regime: {self.market_context['current_regime']}")
            print(f"- Regime History: {list(self.market_context['regime_history'])[-5:]}")
            return 1.0

    def adapt_strategy_to_market(self):
        """Adapt current strategy based on market conditions."""
        # Get best strategy for current regime
        best_strategy = self.strategy_memory.get_best_strategy(
            self.market_context['current_regime'],
            {
                'exploration_noise': self.exploration_noise,
                'exploration_temp': self.exploration_temp,
                'risk_preference': self.risk_preference,
                'learning_rate': self.actor_optimizer.param_groups[0]['lr']
            }
        )
        
        # Apply strategy with current market adaptation factors
        self.exploration_noise = float(best_strategy['exploration_noise'] * self.adaptive_params['vol_scaling_factor'])
        self.exploration_temp = float(best_strategy['exploration_temp'] * self.adaptive_params['trend_scaling_factor'])
        self.risk_preference = float(best_strategy['risk_preference'] * self.adaptive_params['correlation_scaling_factor'])
        
        # Update learning rate
        new_lr = float(best_strategy['learning_rate'] * self.adaptive_params['vol_scaling_factor'])
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        
        print(f"\nAdapted strategy to {self.market_context['current_regime']} regime")
        print(f"Exploration: {self.exploration_noise:.3f}, Temp: {self.exploration_temp:.3f}")
        print(f"Risk: {self.risk_preference:.3f}, LR: {new_lr:.6f}")

    def _force_exploration(self):
        """Force high exploration to break out of local optima"""
        print("\nForcing high exploration phase...")
        
        # Set very high exploration parameters
        self.exploration_noise = self.exploration_bounds[1]
        self.exploration_temp = self.exploration_temp_bounds[1]
        
        # Randomize risk preference
        self.risk_preference = np.random.uniform(self.risk_bounds[0], self.risk_bounds[1])
        
        # Increase learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.lr_bounds['actor'][1]
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.lr_bounds['critic'][1]
        
        # Increase buffer size for more diverse experiences
        new_buffer_size = min(
            int(self.current_buffer_size * 1.5),
            self.buffer_size_bounds[1]
        )
        if new_buffer_size != self.current_buffer_size:
            self._adjust_buffer_size(new_buffer_size)
        
        print(f"Forced exploration parameters - "
              f"Noise: {self.exploration_noise:.3f}, "
              f"Temp: {self.exploration_temp:.3f}, "
              f"Risk: {self.risk_preference:.3f}, "
              f"Buffer: {self.current_buffer_size}")

    def _restore_default_parameters(self):
        """Restore parameters to their default values."""
        print("\nRestoring default parameters...")
        
        # Set balanced default values
        self.exploration_noise = 0.2
        self.exploration_temp = 1.0
        self._adjust_buffer_size(10000)  # Default buffer size
        self.risk_preference = 0.5
        
        # Reset learning rates to default values
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = 1e-4
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = 1e-4
        
        print(f"Restored parameters - Exploration: {self.exploration_noise:.3f}, "
              f"Exploration Temp: {self.exploration_temp:.3f}, "
              f"Buffer Size: {self.current_buffer_size}, "
              f"Risk: {self.risk_preference:.3f}, "
              f"Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
    
    def act(self, state, eval_mode=False):
        """Get action from the actor network."""
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Move to GPU if needed
        if state.device != self.gpu_device:
            state = state.to(self.gpu_device)
        
        # Set networks to eval mode during action selection
        self.actor.eval()
        
        with torch.no_grad():
            # Get action probabilities with current exploration settings
            discrete_probs, allocation_probs = self.actor(
                state,
                exploration_temp=1.0 if eval_mode else self.exploration_temp,
                risk_preference=1.0 if eval_mode else self.risk_preference
            )
            
            # Sample discrete actions
            discrete_actions = []
            for j in range(self.env.num_tickers):
                ticker_probs = discrete_probs[j]
                
                if eval_mode:
                    # During evaluation, take the most probable action
                    action = torch.argmax(ticker_probs).item()
                else:
                    # During training, sample from the probability distribution
                    action = torch.multinomial(ticker_probs, 1).item()
                
                discrete_actions.append(action)
            
            # Convert to tensors
            discrete_actions = torch.tensor(discrete_actions, device=self.gpu_device)
            
            # Add exploration noise to allocation probabilities during training
            if not eval_mode:
                noise = torch.randn_like(allocation_probs) * self.exploration_noise
                allocation_probs = torch.clamp(allocation_probs + noise, 0, 1)
                allocation_probs = allocation_probs / (allocation_probs.sum() + 1e-6)
        
        # Set networks back to training mode if not in eval mode
        if not eval_mode:
            self.actor.train()
        
        return (discrete_actions, allocation_probs)

    def update_market_context(self, state, reward, done):
        """Update market context with new information and handle meta-experience storage"""
        if not hasattr(self, 'env') or not hasattr(self.env, 'unscaled_close_df'):
            return
            
        # Get current prices and calculate returns
        current_prices = self.env._get_prices_for_date(self.env.dates[self.env.current_step])
        self.market_context['prices_history'].append(current_prices.cpu().numpy())
        
        if len(self.market_context['prices_history']) > 1:
            returns = (current_prices - torch.tensor(self.market_context['prices_history'][-2], device=self.gpu_device)) / torch.tensor(self.market_context['prices_history'][-2], device=self.gpu_device)
            self.market_context['returns_history'].append(returns.cpu().numpy())
            
            # Calculate volatility
            if len(self.market_context['returns_history']) > 5:
                recent_returns = np.array(list(self.market_context['returns_history'])[-5:])
                vol = np.std(recent_returns, axis=0) * np.sqrt(252)
                self.market_context['volatility_history'].append(vol)
            
            # Calculate correlations
            if len(self.market_context['returns_history']) > 20:
                returns_matrix = np.array(list(self.market_context['returns_history'])[-20:])
                correlation_matrix = np.corrcoef(returns_matrix.T)
                avg_correlation = np.mean(correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)])
                self.market_context['correlation_history'].append(avg_correlation)
        
        # Detect market regime
        if len(self.market_context['returns_history']) > max(self.regime_detector.window_sizes):
            current_regime = self.regime_detector.detect_regime(
                list(self.market_context['returns_history']),
                list(self.market_context['prices_history'])
            )
            self.market_context['current_regime'] = current_regime
            self.market_context['regime_history'].append(current_regime)

        # Create market features and get current style prediction
        market_features = self._create_market_features()
        with torch.no_grad():
            strategy_metrics = self._get_strategy_metrics()
            style_weights = self.meta_learner(market_features, strategy_metrics)
            style_weights = F.softmax(style_weights, dim=-1)
            selected_style = torch.argmax(style_weights[0]).item()
            
            # Store meta-experience if significant
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item()
            else:
                reward_val = reward
                
            self._store_meta_experience(market_features, selected_style, reward_val)
            
            # Update style if changed
            if selected_style != self.style_names.index(self.current_style):
                self.current_style = self.style_names[selected_style]
                self.adapt_parameters()  # Adapt parameters for new style

    def _create_market_features(self):
        """Create market features with correct dimensions"""
        # Calculate features - ensure these return SINGLE values
        vol = self._calc_rolling_volatility(window=21).item()  # Convert to Python float
        vol_short = self._calc_rolling_volatility(window=10).item()
        trend_short = self._calc_trend_strength(halflife=10).item()
        trend_long = self._calc_trend_strength(halflife=21).item()
        
        # Calculate correlation matrix and flatten
        corr_matrix = self._calc_asset_correlations(window=63)
        flattened_corr = corr_matrix.flatten()  # Ensure this is 4 elements
        
        # Create feature dictionary with PROPER shapes
        features = {
            # 2 volatility measures as scalars
            'volatility': torch.tensor([vol, vol_short], device=self.gpu_device).view(1, 2),
            # 2 trend measures as scalars
            'trend': torch.tensor([trend_short, trend_long], device=self.gpu_device).view(1, 2),
            # Flattened 2x2 correlation matrix (4 elements)
            'correlation_matrix': torch.tensor(flattened_corr, device=self.gpu_device).view(1, 4)
        }
        
        return features

    def _calc_rolling_volatility(self, window):
        """Calculate SINGLE aggregate volatility measure across all assets"""
        # Get current date
        current_date = self.env.dates[self.env.current_step]
        
        # Initialize returns list
        returns_list = []
        
        # Collect returns for the window
        for i in range(max(0, self.env.current_step - window), self.env.current_step):
            date = self.env.dates[i]
            if date in self.env.history_cache['returns']:
                returns_list.append(self.env.history_cache['returns'][date])
        
        # If we have returns data
        if returns_list:
            # Stack returns and calculate volatility
            returns = torch.stack(returns_list)  # Shape: [window, num_assets]
            volatility = torch.std(returns, dim=0) * np.sqrt(252)  # Annualize
            return torch.mean(volatility)  # Return scalar mean volatility
        else:
            # Return default volatility if no data
            return torch.tensor(0.01, device=self.gpu_device)  # Default 1% volatility

    def _calc_trend_strength(self, halflife):
        """Calculate SINGLE aggregate trend measure"""
        # Get current date
        current_date = self.env.dates[self.env.current_step]
        
        # Initialize price lists
        recent_prices = []
        old_prices = []
        
        # Collect recent prices
        for i in range(max(0, self.env.current_step - halflife), self.env.current_step):
            date = self.env.dates[i]
            if date in self.env.history_cache['prices']:
                recent_prices.append(self.env.history_cache['prices'][date])
        
        # Collect older prices
        for i in range(max(0, self.env.current_step - 2*halflife), self.env.current_step - halflife):
            date = self.env.dates[i]
            if date in self.env.history_cache['prices']:
                old_prices.append(self.env.history_cache['prices'][date])
        
        # If we have enough data
        if recent_prices and old_prices:
            # Calculate average prices for each period
            recent_avg = torch.stack(recent_prices).mean(dim=0)
            old_avg = torch.stack(old_prices).mean(dim=0)
            
            # Calculate trend strength
            trend = (recent_avg - old_avg) / (old_avg + 1e-8)
            return torch.mean(trend)  # Return scalar mean trend
        else:
            # Return default trend if no data
            return torch.tensor(0.0, device=self.gpu_device)


    def _calc_asset_correlations(self, window):
        """Calculate correlation matrix between assets"""
        if len(self.market_context['returns_history']) < window:
            # Return identity matrix if not enough history
            return np.eye(self.env.num_tickers, dtype=np.float32)
            
        # Get recent returns and ensure proper shape
        returns = np.array(list(self.market_context['returns_history'])[-window:], dtype=np.float32)
        
        # Ensure returns have shape [window, num_tickers]
        if returns.shape[1] != self.env.num_tickers:
            returns = returns.reshape(window, self.env.num_tickers)
            
        # Calculate correlation matrix
        corr = np.corrcoef(returns.T)
        
        # Ensure we get a 2x2 matrix
        if corr.shape != (self.env.num_tickers, self.env.num_tickers):
            print(f"Warning: Unexpected correlation matrix shape: {corr.shape}")
            return np.eye(self.env.num_tickers, dtype=np.float32)
            
        return corr.astype(np.float32)

    def _calc_liquidity_measures(self):
        # Placeholder for liquidity calculation
        return np.ones(self.env.num_tickers, dtype=np.float32)

    def _get_macro_sentiment_scores(self):
        # Placeholder for macro sentiment
        return np.zeros(self.env.num_tickers, dtype=np.float32)

    def _get_strategy_metrics(self):
        """Get performance metrics for each strategy"""
        metrics = []
        for style in ['aggressive', 'moderate', 'conservative']:
            if style in self.style_performance:
                stats = self.style_performance[style]
                success_rate = stats['successes'] / max(1, stats['total'])
                avg_score = stats['score_sum'] / max(1, stats['total'])
                
                # Get regime-specific performance
                current_regime = self.market_context['current_regime']
                regime_stats = stats.get('regime_performance', {}).get(current_regime, {})
                regime_success = regime_stats.get('successes', 0) / max(1, regime_stats.get('total', 1))
                
                metrics.append([success_rate, avg_score, regime_success])
            else:
                metrics.append([0.0, 0.0, 0.0])
        
        # Shape: [1, 9] (3 styles × 3 metrics)
        return torch.tensor(metrics, dtype=torch.float32, device=self.gpu_device).view(1, -1)

    def _apply_strategy_params(self, params, allow_exploration=False):
        """Apply strategy parameters from network output"""
        # Convert tensor to CPU and get item for each parameter
        params = params.detach().cpu()
        
        # Scale parameters to their respective ranges
        self.exploration_noise = float(params[0, 0].item() * (self.exploration_bounds[1] - self.exploration_bounds[0]) + self.exploration_bounds[0])
        self.exploration_temp = float(params[0, 1].item() * (self.exploration_temp_bounds[1] - self.exploration_temp_bounds[0]) + self.exploration_temp_bounds[0])
        self.risk_preference = float(params[0, 2].item() * (self.risk_bounds[1] - self.risk_bounds[0]) + self.risk_bounds[0])
        
        # Update learning rate
        new_lr = float(params[0, 3].item() * (self.lr_bounds['actor'][1] - self.lr_bounds['actor'][0]) + self.lr_bounds['actor'][0])
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # Add exploration noise if allowed
        if allow_exploration:
            self.exploration_noise = float(params[0, 0].item() * (self.exploration_bounds[1] - self.exploration_bounds[0]) + self.exploration_bounds[0])
            self.exploration_temp = float(params[0, 1].item() * (self.exploration_temp_bounds[1] - self.exploration_temp_bounds[0]) + self.exploration_temp_bounds[0])
            self.risk_preference = float(params[0, 2].item() * (self.risk_bounds[1] - self.risk_bounds[0]) + self.risk_bounds[0])
            
            # Update learning rate
            new_lr = float(params[0, 3].item() * (self.lr_bounds['actor'][1] - self.lr_bounds['actor'][0]) + self.lr_bounds['actor'][0])
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = new_lr

    def _store_meta_experience(self, market_features, selected_style, reward):
        """Store experience for meta-learning with enhanced filtering"""
        # Convert reward to tensor if it's not already
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.gpu_device)
        
        # Mark that we have an episode in progress
        self.episode_in_progress = True
        
        # Get current market state metrics
        current_vol = market_features['volatility'].mean().item()
        
        # Only store if this is a significant point
        should_store = False
        
        # Check if this is the first experience
        if not self.current_episode_meta_experiences:
            should_store = True  # Always store first experience
        else:
            # Get previous experience for comparison
            prev_exp = self.current_episode_meta_experiences[-1]
            prev_reward = prev_exp[2]
            
            # Calculate reward change
            reward_change = abs(reward - prev_reward)/(abs(prev_reward) + 1e-6)
            
            # Store if:
            # 1. High volatility (>15%)
            # 2. Significant reward change (>10%)
            # 3. Regime transition
            # 4. Every N steps (to maintain some regular sampling)
            should_store = (
                current_vol > 0.15 or
                reward_change > 0.1 or
                self._is_regime_transition() or
                len(self.current_episode_meta_experiences) % 20 == 0  # Store every 20th step
            )
        
        if should_store:
            experience = (
                {k: v.clone().detach() for k, v in market_features.items()},
                selected_style,
                reward.clone().detach()
            )
            self.current_episode_meta_experiences.append(experience)

    def _is_regime_transition(self):
        """Check if current step represents a regime transition"""
        if len(self.market_context['regime_history']) < 2:
            return False
        
        current_regime = self.market_context['current_regime']
        prev_regime = list(self.market_context['regime_history'])[-2]
        
        return current_regime != prev_regime

    def _calculate_experience_importance(self, experience, episode_metrics):
        """Calculate importance weight for an experience"""
        market_features, style, reward = experience
        
        # Base importance on multiple factors
        importance = 1.0
        
        # 1. Reward relative to episode mean
        reward_normalized = (reward - episode_metrics['mean_reward']) / (episode_metrics['std_reward'] + 1e-6)
        importance *= (1.0 + abs(reward_normalized))
        
        # 2. Volatility impact
        current_vol = market_features['volatility'].mean().item()
        if current_vol > episode_metrics['mean_vol']:
            importance *= 1.2
        
        # 3. Regime-based importance
        if 'bear' in self.market_context['current_regime']:
            importance *= 1.3  # Higher importance in bear markets
        
        # 4. Strategy success impact
        if style in self.style_performance:
            style_stats = self.style_performance[style]
            if style_stats['total'] > 0:
                success_rate = style_stats['successes'] / style_stats['total']
                if success_rate > 0.6:  # Successful strategy
                    importance *= 1.2
        
        return importance

    def end_episode(self):
        """Handle end of episode for meta-learning with optimized storage"""
        if not self.episode_in_progress:
            return

        # Only process episode if we have enough experiences
        if len(self.current_episode_meta_experiences) >= self.min_episode_experiences:
            # Calculate episode metrics
            episode_rewards = torch.tensor([exp[2].item() for exp in self.current_episode_meta_experiences])
            episode_metrics = {
                'mean_reward': episode_rewards.mean().item(),
                'std_reward': episode_rewards.std().item(),
                'mean_vol': np.mean([exp[0]['volatility'].mean().item() for exp in self.current_episode_meta_experiences]),
            }
            
            if hasattr(self, 'current_episode_returns'):
                returns_tensor = torch.tensor(self.current_episode_returns)
                episode_metrics['sharpe'] = torch.sqrt(torch.tensor(252.)) * (
                    returns_tensor.mean() / (returns_tensor.std() + 1e-6)
                )
            else:
                episode_metrics['sharpe'] = torch.tensor(0.0)
            
            # Filter and enhance experiences
            enhanced_experiences = []
            for exp in self.current_episode_meta_experiences:
                market_features, style, reward = exp
                
                # Calculate importance weight
                importance = self._calculate_experience_importance(exp, episode_metrics)
                
                # Create enhanced experience
                enhanced_exp = (
                    market_features,
                    style,
                    reward,
                    {
                        'importance': importance,
                        'episode_mean_reward': episode_metrics['mean_reward'],
                        'episode_std_reward': episode_metrics['std_reward'],
                        'episode_sharpe': float(episode_metrics['sharpe']),
                        'regime': self.market_context['current_regime']
                    }
                )
                enhanced_experiences.append(enhanced_exp)
            
            # Sort by importance and select top experiences
            enhanced_experiences.sort(key=lambda x: x[3]['importance'], reverse=True)
            
            # Keep only the most important experiences (about 50-100 per episode)
            max_experiences = min(100, len(enhanced_experiences))
            selected_experiences = enhanced_experiences[:max_experiences]
            
            # Add to meta memory
            self.meta_memory.extend(selected_experiences)
            

            
            # Perform meta-learning update if we have enough total experiences
            if len(self.meta_memory) >= self.min_meta_samples:
                self._meta_learning_step()
        
        # Reset episode state
        self.current_episode_meta_experiences = []
        self.current_episode_returns = []
        self.episode_in_progress = False

    def _meta_learning_step(self):
        """Perform meta-learning update with enhanced episode-collected experiences"""
        if len(self.meta_memory) < self.meta_batch_size:
            print(f"\nSkipping meta-learning step - insufficient samples ({len(self.meta_memory)} < {self.meta_batch_size})")
            return
            
        # Calculate adaptive entropy coefficient based on training progress
        entropy_coef = max(0.01, min(0.1, 1.0 / (1.0 + len(self.meta_memory) / 1000)))
            
        # Sample batch from meta memory
        batch = random.sample(list(self.meta_memory), self.meta_batch_size)
        batch_features, batch_styles, batch_rewards, batch_metrics = zip(*batch)
        
        # Convert to tensors
        batch_styles = torch.tensor(batch_styles, device=self.gpu_device)
        batch_rewards = torch.tensor([r.item() if isinstance(r, torch.Tensor) else r for r in batch_rewards], device=self.gpu_device)
        
        # Process batch features
        batch_size = len(batch_features)
        batch_combined_features = {
            'volatility': torch.zeros((batch_size, 2), device=self.gpu_device),
            'trend': torch.zeros((batch_size, 2), device=self.gpu_device),
            'correlation_matrix': torch.zeros((batch_size, 4), device=self.gpu_device)
        }
        
        # Stack features properly
        for i, features in enumerate(batch_features):
            for key in ['volatility', 'trend', 'correlation_matrix']:
                tensor = features[key]
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, device=self.gpu_device)
                # Ensure 2D shape [1, N]
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                batch_combined_features[key][i] = tensor.reshape(1, -1)
        
        # Get strategy metrics and expand to batch size
        strategy_metrics = self._get_strategy_metrics()  # Shape: [1, 9]
        strategy_metrics = strategy_metrics.expand(batch_size, -1)  # Shape: [batch_size, 9]
        
        # Get predictions
        batch_style_weights = self.meta_learner(batch_combined_features, strategy_metrics)
        
        # Calculate enhanced loss with performance metrics
        batch_policy_loss = F.cross_entropy(batch_style_weights, batch_styles, reduction='none')
        
        # Calculate advantages using episode metrics
        episode_rewards = torch.tensor([m['episode_mean_reward'] for m in batch_metrics], device=self.gpu_device)
        episode_sharpes = torch.tensor([m['episode_sharpe'] for m in batch_metrics], device=self.gpu_device)
        
        # Combine immediate rewards with episode performance
        combined_advantage = (
            0.5 * (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-6) +
            0.3 * (episode_rewards - episode_rewards.mean()) / (episode_rewards.std() + 1e-6) +
            0.2 * episode_sharpes
        )
        
        # Weight policy loss by advantage
        batch_policy_loss = (batch_policy_loss * combined_advantage).mean()
        
        # Add diversity regularization
        batch_probs = F.softmax(batch_style_weights, dim=1)
        batch_entropy = -(batch_probs * torch.log(batch_probs + 1e-10)).sum(dim=1).mean()
        
        # Combine losses with entropy coefficient
        batch_total_loss = batch_policy_loss - entropy_coef * batch_entropy
        
        # Optimization step
        self.meta_optimizer.zero_grad()
        batch_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.market_encoder.parameters(), 1.0)
        self.meta_optimizer.step()
        


    def _update_style_performance(self, style, score, prev_best_score):
        """Update performance metrics for the current trading style with enhanced tracking"""
        if style not in self.style_performance:
            self.style_performance[style] = {
                'total': 0,
                'successes': 0,
                'score_sum': 0.0,
                'regime_performance': {},
                'training_metrics': {
                    'returns': deque(maxlen=100),
                    'sharpe': deque(maxlen=100),
                    'drawdown': deque(maxlen=50),
                    'learning_progress': deque(maxlen=100),
                    'policy_loss': deque(maxlen=100),
                    'entropy': deque(maxlen=100),
                    'confidence': deque(maxlen=100),
                    'style_entropy': deque(maxlen=100),
                    'advantage_mean': deque(maxlen=100),
                    'advantage_std': deque(maxlen=100),
                    'adaptation_count': 0
                }
            }
        
        # Initialize metrics if they don't exist
        metrics = self.style_performance[style]['training_metrics']
        for key in ['policy_loss', 'entropy', 'confidence', 'style_entropy', 
                   'advantage_mean', 'advantage_std']:
            if key not in metrics:
                metrics[key] = deque(maxlen=100)
        
        # Update total attempts and running statistics
        self.style_performance[style]['total'] += 1
        self.style_performance[style]['score_sum'] += score
        
        # Track regime-specific performance
        current_regime = self.market_context['current_regime']
        if current_regime not in self.style_performance[style]['regime_performance']:
            self.style_performance[style]['regime_performance'][current_regime] = {
                'total': 0,
                'successes': 0,
                'score_sum': 0.0,
                'returns': [],
                'volatility': [],
                'sharpe': [],
                'drawdown': [],
                'learning_curve': [],
                'adaptation_history': []
            }
        
        regime_stats = self.style_performance[style]['regime_performance'][current_regime]
        regime_stats['total'] += 1
        regime_stats['score_sum'] += score
        
        # Calculate relative improvement before any other calculations
        relative_improvement = (score - prev_best_score) / (abs(prev_best_score) + 1e-6)
        
        # Calculate and store training metrics
        training_metrics = self.style_performance[style]['training_metrics']
        if len(self.market_context['returns_history']) > 0:
            recent_returns = np.array(list(self.market_context['returns_history'])[-20:])
            training_metrics['returns'].append(np.mean(recent_returns))
            
            # Calculate Sharpe ratio
            if len(recent_returns) > 1:
                sharpe = np.sqrt(252) * (np.mean(recent_returns) / (np.std(recent_returns) + 1e-6))
                training_metrics['sharpe'].append(sharpe)
                regime_stats['sharpe'].append(sharpe)
            
            # Calculate drawdown
            if len(self.market_context['prices_history']) > 1:
                prices = np.array(list(self.market_context['prices_history']))
                peak = np.maximum.accumulate(prices)
                drawdown = (peak - prices) / peak
                max_drawdown = np.max(drawdown)
                training_metrics['drawdown'].append(max_drawdown)
                regime_stats['drawdown'].append(max_drawdown)
            
            # Track learning progress
            training_metrics['learning_progress'].append(relative_improvement)
            regime_stats['learning_curve'].append(relative_improvement)
        
        # Consider it a success with more nuanced criteria
        improvement_threshold = self.style_eval_threshold
        if 'bear' in current_regime:
            improvement_threshold *= 0.5  # Lower threshold for bear markets
        
        is_success = False
        if score > prev_best_score * (1 + improvement_threshold):
            is_success = True
        elif score > 0 and 'bear' in current_regime:
            is_success = True  # Count positive returns in bear markets as success
        elif len(regime_stats['sharpe']) > 0:
            recent_sharpe = np.mean(regime_stats['sharpe'][-5:])
            if recent_sharpe > 1.0:  # Consider good risk-adjusted returns as success
                is_success = True
        
        if is_success:
            self.style_performance[style]['successes'] += 1
            regime_stats['successes'] += 1
            training_metrics['adaptation_count'] += 1
            
            # Store adaptation event
            regime_stats['adaptation_history'].append({
                'step': self.t_step,
                'score': score,
                'improvement': relative_improvement,
                'market_regime': current_regime,
                'parameters': {
                    'exploration_noise': float(self.exploration_noise),
                    'exploration_temp': float(self.exploration_temp),
                    'risk_preference': float(self.risk_preference),
                    'learning_rate': float(self.actor_optimizer.param_groups[0]['lr'])
                }
            })
        
        # Update style probabilities with enhanced logic
        total_trials = sum(stats['total'] for stats in self.style_performance.values())
        if total_trials >= 5:  # Wait for enough data
            style_metrics = {}
            for s, stats in self.style_performance.items():
                # Calculate base success rate
                success_rate = stats['successes'] / max(1, stats['total'])
                avg_score = stats['score_sum'] / max(1, stats['total'])
                
                # Get training progress metrics
                training_metrics = stats['training_metrics']
                recent_progress = np.mean(list(training_metrics['learning_progress'])[-10:]) if training_metrics['learning_progress'] else 0
                recent_sharpe = np.mean(list(training_metrics['sharpe'])[-10:]) if training_metrics['sharpe'] else 0
                
                # Calculate regime-specific performance
                if current_regime in stats['regime_performance']:
                    regime_stats = stats['regime_performance'][current_regime]
                    regime_success = regime_stats['successes'] / max(1, regime_stats['total'])
                    regime_sharpe = np.mean(regime_stats['sharpe']) if regime_stats['sharpe'] else 0
                    
                    # Combine metrics with regime awareness
                    if 'bear' in current_regime:
                        # In bear markets, weight regime performance more heavily
                        style_metrics[s] = (
                            0.3 * success_rate +
                            0.2 * (avg_score / (max(1e-6, prev_best_score))) +
                            0.2 * regime_success +
                            0.1 * max(0, regime_sharpe) +
                            0.1 * recent_progress +
                            0.1 * max(0, recent_sharpe)
                        )
                    else:
                        # In other markets, balance between overall and regime performance
                        style_metrics[s] = (
                            0.25 * success_rate +
                            0.2 * (avg_score / (max(1e-6, prev_best_score))) +
                            0.15 * regime_success +
                            0.15 * max(0, regime_sharpe) +
                            0.15 * recent_progress +
                            0.1 * max(0, recent_sharpe)
                        )
                else:
                    # Fall back to basic metrics if no regime data
                    style_metrics[s] = (
                        0.4 * success_rate +
                        0.3 * (avg_score / (max(1e-6, prev_best_score))) +
                        0.2 * recent_progress +
                        0.1 * max(0, recent_sharpe)
                    )
            
            # Convert to probabilities with minimum probability constraint
            total_metric = sum(style_metrics.values())
            if total_metric > 0:
                raw_probs = {s: m/total_metric for s, m in style_metrics.items()}
                
                # Apply minimum probability constraint
                min_prob = self.min_style_prob
                excess = 0
                for s, p in raw_probs.items():
                    if p < min_prob:
                        excess += min_prob - p
                        raw_probs[s] = min_prob
                
                # Redistribute excess proportionally
                if excess > 0:
                    above_min = {s: p-min_prob for s, p in raw_probs.items() if p > min_prob}
                    if above_min:
                        total_above = sum(above_min.values())
                        for s in above_min:
                            raw_probs[s] -= (excess * above_min[s] / total_above)
                
                self.style_probs = raw_probs
                
                # Print detailed performance metrics
                print("\nStyle Performance Update:")
                for style, prob in self.style_probs.items():
                    stats = self.style_performance[style]
                    success_rate = stats['successes'] / max(1, stats['total'])
                    avg_score = stats['score_sum'] / max(1, stats['total'])
                    
                    print(f"\n{style.capitalize()}:")
                    print(f"  Probability: {prob:.2%}")
                    print(f"  Overall Success Rate: {success_rate:.2%}")
                    print(f"  Average Score: {avg_score:.4f}")
                    
                    if current_regime in stats.get('regime_performance', {}):
                        r_stats = stats['regime_performance'][current_regime]
                        r_success = r_stats['successes'] / max(1, r_stats['total'])
                        r_sharpe = np.mean(r_stats['sharpe']) if r_stats['sharpe'] else 0
                        print(f"  {current_regime} Performance:")
                        print(f"    Success Rate: {r_success:.2%}")
                        print(f"    Sharpe Ratio: {r_sharpe:.2f}")

    def _process_rewards(self, rewards):
        """Apply asymmetric scaling to rewards with enhanced regime awareness"""
        # Get current portfolio state
        total_value = self.env.invested_capital + self.env.remaining_capital
        initial_value = self.env.initial_capital
        
        # Calculate portfolio performance metrics
        portfolio_return = (total_value - initial_value) / initial_value
        
        # Update returns and volatility history
        self.returns_history.append(portfolio_return)
        if len(self.returns_history) > 1:
            returns_array = np.array(list(self.returns_history))
            volatility = np.std(returns_array) * np.sqrt(252)
            self.volatility_history.append(volatility)
        
        # Get regime information
        current_regime = self.market_context['current_regime']
        regime_duration = len([r for r in self.market_context['regime_history'] if r == current_regime])
        
        # Enhanced reward scaling based on regime
        if 'bear' in current_regime:
            # Progressive penalty scaling in bear markets
            loss_scale = self.loss_multiplier * (1.5 + regime_duration/20)  # Increases with duration
            gain_scale = self.gain_multiplier * (1.2 + regime_duration/30)  # Reward good decisions more
        else:
            loss_scale = self.loss_multiplier
            gain_scale = self.gain_multiplier
        
        # Apply asymmetric scaling with enhanced risk adjustment
        processed_rewards = torch.where(
            rewards > 0,
            rewards * torch.clamp(gain_scale * (1.0 + self.risk_preference * 0.2), 0.1, 5.0),
            rewards * torch.clamp(loss_scale * (1.0 - self.risk_preference * 0.1), 0.1, 3.0)
        )
        
        return processed_rewards

    def _adjust_buffer_size(self, new_size: int):
        """Safely adjust the replay buffer size."""
        try:
            # Ensure new size is within bounds
            new_size = max(self.buffer_size_bounds[0], min(new_size, self.buffer_size_bounds[1]))
            
            if new_size == self.current_buffer_size:
                return
            
            print(f"\nAdjusting buffer size from {self.current_buffer_size} to {new_size}")
            
            # Create new buffer with desired size
            new_buffer = PrioritizedReplayBuffer(
                capacity=new_size,
                alpha=self.memory.alpha,
                beta=self.memory.beta
            )
            
            # Transfer existing experiences to new buffer
            experiences = list(self.memory.buffer)
            priorities = list(self.memory.priorities)
            
            # Transfer in chunks to manage memory
            chunk_size = 1000
            for i in range(0, len(experiences), chunk_size):
                chunk_experiences = experiences[i:i + chunk_size]
                chunk_priorities = priorities[i:i + chunk_size]
                
                for exp, priority in zip(chunk_experiences, chunk_priorities):
                    if isinstance(exp, tuple):
                        state, action, reward, next_state, done = exp
                    else:  # CompressedExperience
                        state, action, reward, next_state, done = exp.decompress()
                    
                    new_buffer.add(state, action, reward, next_state, done)
                    new_buffer.priorities[-1] = priority
            
            # Update buffer attributes
            new_buffer.max_priority = self.memory.max_priority
            new_buffer.experience_hash_set = self.memory.experience_hash_set.copy()
            new_buffer.td_error_history = self.memory.td_error_history.copy()
            new_buffer.priority_variance = self.memory.priority_variance
            
            # Replace old buffer
            self.memory = new_buffer
            self.current_buffer_size = new_size
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error adjusting buffer size: {str(e)}")
            print("Continuing with current buffer size")
            # If adjustment fails, keep current buffer
            return