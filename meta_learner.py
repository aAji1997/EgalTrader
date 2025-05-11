import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
import numpy as np
import hashlib
import time


class MarketRegimeDetector:
    """Detects market regimes using volatility and trend analysis"""
    def __init__(self, window_sizes=[5, 10, 20, 50]):
        self.window_sizes = sorted(window_sizes)  # Sort to ensure consistent behavior
        self.regime_history = deque(maxlen=100)
        self.volatility_thresholds = {
            'low': 0.1,
            'medium': 0.2,
            'high': 0.3
        }
        self.trend_thresholds = {
            'strong_down': -0.02,
            'down': -0.01,
            'neutral': 0.01,
            'up': 0.02
        }
        # Default regime for insufficient history
        self.default_regime = 'neutral_medium_vol'
        
        # Valid regime components for validation
        self.valid_trends = {'strong_bear', 'bear', 'neutral', 'bull', 'strong_bull'}
        self.valid_vols = {'low_vol', 'medium_vol', 'high_vol'}
    
    def detect_regime(self, returns_history, prices_history):
        """Detect current market regime using vectorized operations and precomputed values."""
        if not returns_history or not prices_history:
            return self.default_regime

        # Convert to numpy arrays once
        returns_array = np.asarray(returns_history)
        prices_array = np.asarray(prices_history)
        
        max_window = max(self.window_sizes)
        min_required = max(max_window, 1)  # Ensure at least 1 period
        
        if len(returns_array) < min_required:
            # Handle insufficient data with vectorized operations
            available_window = min(len(returns_array), max(self.window_sizes))
            if available_window == 0:
                return self.default_regime
                
            # Calculate with available data using the largest possible window
            recent_returns = returns_array[-available_window:]
            recent_prices = prices_array[-available_window:]
            
            # Vectorized volatility calculation
            vol = np.mean(np.std(recent_returns, axis=0)) * np.sqrt(252)
            price_ratio = recent_prices[-1] / recent_prices[0]
            trend = np.mean(price_ratio - 1) / (available_window/252)
            
            return self._classify_regime(vol, trend)
        
        # Precompute slices for max window to avoid repeated slicing
        returns_slice = returns_array[-max_window:]
        prices_slice = prices_array[-max_window:]
        
        # Precompute constants
        sqrt_252 = np.sqrt(252)
        annualization_factor = 252
        
        # Vectorized calculations for all window sizes
        volatilities = []
        trends = []
        
        for window in self.window_sizes:
            # Get window slices from precomputed max window slice
            window_returns = returns_slice[-window:]
            window_prices = prices_slice[-window:]
            
            # Vectorized volatility calculation
            asset_vols = np.std(window_returns, axis=0) * sqrt_252
            volatilities.append(np.mean(asset_vols))
            
            # Vectorized trend calculation
            price_changes = (window_prices[-1] / window_prices[0]) - 1
            annualized_trend = np.mean(price_changes) / (window/annualization_factor)
            trends.append(annualized_trend)
        
        # Use precomputed thresholds
        avg_vol = np.mean(volatilities)
        avg_trend = np.mean(trends)
        
        return self._classify_regime(avg_vol, avg_trend)

    def _classify_regime(self, vol, trend):
        """Classify regime based on precomputed thresholds."""
        # Volatility classification
        if vol < self.volatility_thresholds['low']:
            vol_regime = 'low_vol'
        elif vol < self.volatility_thresholds['medium']:
            vol_regime = 'medium_vol'
        else:
            vol_regime = 'high_vol'
        
        # Trend classification
        if trend < self.trend_thresholds['strong_down']:
            trend_regime = 'strong_bear'
        elif trend < self.trend_thresholds['down']:
            trend_regime = 'bear'
        elif trend < self.trend_thresholds['neutral']:
            trend_regime = 'neutral'
        elif trend < self.trend_thresholds['up']:
            trend_regime = 'bull'
        else:
            trend_regime = 'strong_bull'
        
        # Validate components
        trend_regime = trend_regime if trend_regime in self.valid_trends else 'neutral'
        vol_regime = vol_regime if vol_regime in self.valid_vols else 'medium_vol'
        
        regime = f"{trend_regime}_{vol_regime}"
        self.regime_history.append(regime)
        
        return regime

class StrategyMemory:
    """Maintains memory of successful strategies for different market regimes"""
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.strategies = {}  # Regime -> List of successful strategies
        self.strategy_scores = {}  # Strategy -> Historical scores
        self.similarity_threshold = 0.8
        
        # Strategy parameter bounds
        self.param_bounds = {
            'exploration_noise': (0.01, 0.5),
            'exploration_temp': (0.5, 2.0),
            'risk_preference': (0.1, 0.9),
            'learning_rate': (1e-5, 1e-3)
        }
    
    def add_strategy(self, regime, strategy_params, score):
        """Add a successful strategy to memory"""
        if regime not in self.strategies:
            self.strategies[regime] = []
        
        # Create strategy fingerprint
        strategy_key = self._create_strategy_key(strategy_params)
        
        # Check if similar strategy exists
        similar_found = False
        for existing_strategy in self.strategies[regime]:
            if self._calculate_similarity(strategy_params, existing_strategy['params']) > self.similarity_threshold:
                # Update existing strategy
                existing_strategy['score'] = 0.9 * existing_strategy['score'] + 0.1 * score
                similar_found = True
                break
        
        if not similar_found:
            # Add new strategy
            strategy_entry = {
                'params': strategy_params,
                'score': score,
                'key': strategy_key,
                'timestamp': time.time()
            }
            
            self.strategies[regime].append(strategy_entry)
            self.strategy_scores[strategy_key] = deque(maxlen=10)
            self.strategy_scores[strategy_key].append(score)
            
            # Maintain capacity per regime
            if len(self.strategies[regime]) > self.capacity:
                # Remove worst performing strategy
                self.strategies[regime].sort(key=lambda x: x['score'])
                removed_strategy = self.strategies[regime].pop(0)
                del self.strategy_scores[removed_strategy['key']]
    
    def get_best_strategy(self, regime, current_params=None):
        """Get best strategy for current regime with interpolation"""
        if regime not in self.strategies or not self.strategies[regime]:
            return current_params
        
        # Sort strategies by score
        sorted_strategies = sorted(self.strategies[regime], key=lambda x: x['score'], reverse=True)
        
        if current_params is None:
            # Return best strategy
            return sorted_strategies[0]['params']
        
        # Find similar strategies for interpolation
        similar_strategies = []
        for strategy in sorted_strategies:
            similarity = self._calculate_similarity(current_params, strategy['params'])
            if similarity > self.similarity_threshold:
                similar_strategies.append((strategy, similarity))
        
        if not similar_strategies:
            # No similar strategies found, return best
            return sorted_strategies[0]['params']
        
        # Interpolate between similar strategies
        total_weight = sum(sim for _, sim in similar_strategies)
        interpolated_params = {}
        
        for param_name in current_params.keys():
            weighted_sum = sum(
                strategy['params'][param_name] * sim 
                for strategy, sim in similar_strategies
            )
            interpolated_params[param_name] = weighted_sum / total_weight
            
            # Ensure within bounds
            if param_name in self.param_bounds:
                min_val, max_val = self.param_bounds[param_name]
                interpolated_params[param_name] = np.clip(
                    interpolated_params[param_name], 
                    min_val, 
                    max_val
                )
        
        return interpolated_params
    
    def _create_strategy_key(self, params):
        """Create unique key for strategy"""
        param_str = '_'.join(f"{k}:{v:.4f}" for k, v in sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _calculate_similarity(self, params1, params2):
        """Calculate similarity between two parameter sets"""
        similarities = []
        for param_name in params1.keys():
            if param_name in self.param_bounds:
                min_val, max_val = self.param_bounds[param_name]
                range_size = max_val - min_val
                diff = abs(params1[param_name] - params2[param_name])
                param_similarity = 1 - (diff / range_size)
                similarities.append(param_similarity)
        
        return np.mean(similarities) if similarities else 0.0

class MetaStrategyLearner(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, num_styles=3):
        super().__init__()
        self.input_dim = input_dim  # Should be 17 (8 market features + 9 strategy metrics)
        self.hidden_dim = hidden_dim
        self.num_styles = num_styles
        
        # Feature processing layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Style head
        self.style_head = nn.Linear(hidden_dim, num_styles)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.fc3, self.style_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, features, strategy_metrics):
        """Forward pass with consistent tensor shapes"""
        # Get batch size from first feature
        batch_size = features['volatility'].size(0)
        
        # Combine market features - shape [batch_size, 8]
        market_features = torch.cat([
            features['volatility'].view(batch_size, 2),
            features['trend'].view(batch_size, 2),
            features['correlation_matrix'].view(batch_size, 4)
        ], dim=1)
        
        # Ensure strategy metrics have correct shape [batch_size, 9]
        if strategy_metrics.dim() == 1:
            strategy_metrics = strategy_metrics.unsqueeze(0)
        if strategy_metrics.size(0) != batch_size:
            strategy_metrics = strategy_metrics.expand(batch_size, -1)
        
        # Combine all features - shape [batch_size, 17]
        x = torch.cat([market_features, strategy_metrics], dim=1)
        
        # Forward pass through network
        h1 = self.dropout(self.norm1(F.relu(self.fc1(x))))
        h2 = self.dropout(self.norm2(F.relu(self.fc2(h1)))) + h1
        h3 = self.dropout(self.norm3(F.relu(self.fc3(h2)))) + h2
        
        # Style head
        style_logits = self.style_head(h3)
        style_weights = F.softmax(style_logits, dim=-1)
        
        return style_weights

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))

class StyleTransformerWithGating(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim)
        )
        
        # Enhanced gating mechanism with context
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Feature-wise gating
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        transformed = self.transform(x)
        context_gates = self.context_gate(x)
        feature_gates = self.feature_gate(x)
        
        # Combine gates with learned weighting
        combined_gate = 0.7 * context_gates + 0.3 * feature_gates
        gated_output = transformed * combined_gate
        
        return self.output_norm(gated_output)

class MarketContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim  # Should be 8 (2 vol + 2 trend + 4 corr)
        self.hidden_dim = hidden_dim
        
        # Feature processing layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Regime encoding layers
        self.regime_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Regime probability head
        self.regime_head = nn.Linear(hidden_dim, 9)  # 9 possible regimes
    
    def forward(self, features):
        """Process market features to get regime encoding and probabilities"""
        # Combine features into a single tensor - shape [batch_size, 8]
        batch_size = features['volatility'].size(0)
        x = torch.cat([
            features['volatility'].view(batch_size, 2),
            features['trend'].view(batch_size, 2),
            features['correlation_matrix'].view(batch_size, 4)
        ], dim=1)
        
        # Process features
        h = self.feature_net(x)
        
        # Get regime encoding
        regime_encoding = self.regime_net(h)
        
        # Get regime probabilities
        regime_logits = self.regime_head(regime_encoding)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        return regime_encoding, regime_probs

class StrategyBank:
    def __init__(self, num_strategies):
        # Initialize strategy heads with correct input dimension and move to GPU
        self.strategies = nn.ModuleList([
            StrategyHead(input_dim=256) for _ in range(num_strategies)
        ]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.performance_metrics = defaultdict(lambda: {
            'returns': deque(maxlen=100),
            'sharpe': deque(maxlen=100),
            'drawdown': deque(maxlen=50),
            'regime_success': defaultdict(float)
        })
        
    def update_performance(self, strategy_idx, regime, metrics):
        self.performance_metrics[strategy_idx]['returns'].append(metrics['return'])
        self.performance_metrics[strategy_idx]['sharpe'].append(metrics['sharpe'])
        self.performance_metrics[strategy_idx]['regime_success'][regime] += metrics['success']

class StrategyHead(nn.Module):
    def __init__(self, input_dim=256, output_dim=4):  # 4 parameters: noise, temp, risk, lr
        super().__init__()
        print(f"\nInitializing StrategyParameterHead:")  # Renamed for clarity
        print(f"Input dim: {input_dim}")
        print(f"Parameter output dim: {output_dim}")  # Clarified this is for parameters
        
        # Transform network for parameter prediction (not style selection)
        self.feature_projectors = nn.ModuleDict({
            'volatility': nn.Sequential(
                nn.Linear(2, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, input_dim)
            ),
            'trend': nn.Sequential(
                nn.Linear(2, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, input_dim)
            ),
            'correlation_matrix': nn.Sequential(
                nn.Linear(4, input_dim // 4),  # 2x2 matrix flattened to 4
                nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, input_dim)
            ),
            'liquidity_metrics': nn.Sequential(
                nn.Linear(2, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, input_dim)
            ),
            'macro_sentiment': nn.Sequential(
                nn.Linear(2, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, input_dim)
            )
        })
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(input_dim * 5, input_dim),  # Combine all projected features
            nn.LayerNorm(input_dim),
            nn.GELU()
        )
        
        # Parameter prediction network (not style selection)
        self.parameter_net = nn.Sequential(  # Renamed for clarity
            nn.Linear(input_dim, input_dim//2),
            nn.LayerNorm(input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2, output_dim),
            nn.Sigmoid()  # Bound outputs to [0,1]
        )
        
    def forward(self, x):
        """
        Process market features and output strategy parameters (not styles)
        Args:
            x: Dictionary of market features or tensor of encoded features
        Returns:
            Tensor of shape [batch_size, 4] containing parameter values [noise, temp, risk, lr]
        """
        
        if isinstance(x, dict):
            # Project each feature type separately
            projected_features = []
            for key, projector in self.feature_projectors.items():
                if isinstance(x[key], torch.Tensor):
                    
                    if x[key].device != next(self.parameters()).device:
                        x[key] = x[key].to(next(self.parameters()).device)
                    
                    # Handle correlation matrix specially
                    if key == 'correlation_matrix':
                        # Ensure matrix is 2D and flatten
                        if x[key].dim() > 2:
                            feature = x[key].view(x[key].size(0), -1)  # [batch, 2, 2] -> [batch, 4]
                        else:
                            feature = x[key].view(-1, 4)  # [2, 2] -> [1, 4]
                    else:
                        # For other features, ensure 2D with batch dimension
                        feature = x[key].view(1, -1) if x[key].dim() == 1 else x[key]
                    
                    
                    projected = projector(feature)
                    
                    projected_features.append(projected)
                else:
                    
                    feature = torch.tensor(x[key], dtype=torch.float32, device=next(self.parameters()).device)
                    
                    # Handle correlation matrix specially
                    if key == 'correlation_matrix':
                        # Ensure matrix is 2D and flatten
                        if feature.dim() > 2:
                            feature = feature.view(feature.size(0), -1)  # [batch, 2, 2] -> [batch, 4]
                        else:
                            feature = feature.view(-1, 4)  # [2, 2] -> [1, 4]
                    else:
                        # For other features, ensure 2D with batch dimension
                        feature = feature.view(1, -1) if feature.dim() == 1 else feature
                    
                    
                    projected = projector(feature)
                    
                    projected_features.append(projected)
            
            # Concatenate all projected features
            x = torch.cat(projected_features, dim=-1)
            
            x = self.feature_combiner(x)
            
        else:
            # If input is already a tensor (e.g., regime encoding), use it directly
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
            if x.dim() == 1:
                x = x.unsqueeze(0)

        # Process through main network
        out = self.parameter_net(x)
        
        return out

class StyleHead(nn.Module):
    def __init__(self, input_dim, num_styles=3):
        super().__init__()
        self.styles = ['aggressive', 'moderate', 'conservative']
        
        # Linear layer for style logits
        self.style_layer = nn.Linear(input_dim, num_styles)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.style_layer.weight, mode='fan_in', nonlinearity='relu')
            
    def forward(self, x, return_selection=False):
        """Process input through style head and get style selection"""
        # Get logits and probabilities
        logits = self.style_layer(x)  # [hidden_dim] -> [num_styles]
        probabilities = F.softmax(logits, dim=-1)  # [num_styles]
        
        if return_selection:
            # Get the selected style index
            selected_style = torch.argmax(probabilities).item()
            # Get the style name
            selected_style_name = self.styles[selected_style]
            return probabilities, selected_style, selected_style_name
            
        return probabilities