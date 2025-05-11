import torch.nn.functional as F
import torch
import torch.nn as nn
import gymnasium as gym

from embedding import TickerEmbedding, FeatureExtractor

class Actor(nn.Module):
    def __init__(self, env: gym.Env, hidden_size=512, embedding_dim=64):
        super(Actor, self).__init__()
        self.env = env
        self.num_tickers = env.num_tickers
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Calculate features excluding portfolio features
        self.features_per_ticker = env.num_features_per_ticker
        self.market_features = self.features_per_ticker * env.num_tickers
        
        print(f"Actor initialized with:")
        print(f"Market features: {self.market_features}")
        print(f"Number of tickers: {self.num_tickers}")
        print(f"Features per ticker: {self.features_per_ticker}")
        print(f"Total features (including portfolio): {env.total_features}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Hidden size: {hidden_size}")
        
        # Add exploration parameters
        self.min_action_prob = 0.1
        
        # Ticker embeddings
        self.ticker_embeddings = TickerEmbedding(self.num_tickers, embedding_dim)
        
        # Separate feature extractors for each ticker
        self.ticker_extractors = nn.ModuleList([
            FeatureExtractor(
                num_features=self.features_per_ticker,
                hidden_size=hidden_size,
                embedding_dim=embedding_dim
            )
            for _ in range(self.num_tickers)
        ])
        
        # Portfolio feature extractor
        self.portfolio_extractor = nn.Sequential(
            nn.Linear(2 + embedding_dim, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-ticker attention for global market context
        self.market_attention = nn.MultiheadAttention(hidden_size, 4, dropout=0.1)
        self.market_context_ln = nn.LayerNorm(hidden_size)
        
        # Ticker-specific attention mechanism with market context
        self.ticker_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Double size to include market context
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Action heads with enhanced ticker-specific processing
        action_head_input_size = hidden_size * 2 + hidden_size // 4  # ticker features + market context + portfolio
        
        self.discrete_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_head_input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3)  # 3 actions per ticker
            )
            for _ in range(self.num_tickers)
        ])
        
        self.allocation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_head_input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)  # 1 allocation value per ticker
            )
            for _ in range(self.num_tickers)
        ])
        
        # Add ticker-specific bias parameters
        self.ticker_biases = nn.Parameter(torch.zeros(self.num_tickers))
        
        # Add market context integration layer
        self.market_context_integration = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
    
    def forward(self, state, exploration_temp=1.0, risk_preference=1.0):
        """
        Forward pass with proper handling of ticker-specific features and embeddings
        Args:
            state: The current state tensor [batch_size, total_features] or [total_features]
            exploration_temp: Exploration temperature parameter passed from the agent
            risk_preference: Risk preference parameter passed from the agent
        Returns:
            discrete_probs: Action probabilities [batch_size, num_tickers, 3]
            allocation_probs: Allocation values [batch_size, num_tickers]
        """
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Handle 3D input (batch_size, sequence_length, features)
        if state.dim() == 3:
            batch_size, seq_len, features = state.shape
            state = state.reshape(batch_size * seq_len, features)
        else:
            batch_size = state.size(0)
        
        # Split state into market and portfolio features
        market_features = state[:, :self.market_features]
        portfolio_features = state[:, -2:]
        
        # Get ticker embeddings
        ticker_indices = torch.arange(self.num_tickers, device=state.device)
        ticker_embeddings = self.ticker_embeddings(ticker_indices)  # [num_tickers, embedding_dim]
        
        # Process each ticker's features
        ticker_features = []
        for i in range(self.num_tickers):
            # Extract features for current ticker
            start_idx = i * self.features_per_ticker
            end_idx = start_idx + self.features_per_ticker
            ticker_state = market_features[:, start_idx:end_idx]
            
            # Get ticker-specific embedding
            ticker_embed = ticker_embeddings[i].unsqueeze(0).expand(batch_size, -1)
            
            # Process through ticker-specific extractor
            ticker_output = self.ticker_extractors[i](ticker_state, ticker_embed)
            ticker_features.append(ticker_output)
        
        # Stack ticker features
        ticker_features = torch.stack(ticker_features, dim=1)  # [batch_size, num_tickers, hidden_size]
        
        # Process portfolio features with embeddings
        portfolio_embed = ticker_embeddings.mean(dim=0, keepdim=True).expand(batch_size, -1)  # Average embedding
        portfolio_input = torch.cat([portfolio_features, portfolio_embed], dim=1)
        portfolio_features = self.portfolio_extractor(portfolio_input)  # [batch_size, hidden_size//4]
        
        # Apply market attention
        market_context, _ = self.market_attention(
            ticker_features.transpose(0, 1),  # [num_tickers, batch_size, hidden_size]
            ticker_features.transpose(0, 1),
            ticker_features.transpose(0, 1)
        )
        market_context = market_context.transpose(0, 1)  # [batch_size, num_tickers, hidden_size]
        
        # Integrate market context
        market_context = self.market_context_ln(market_context)
        
        # Process each ticker with market context and portfolio features
        discrete_probs = []
        allocation_probs = []
        
        for i in range(self.num_tickers):
            # Combine ticker features with market context
            ticker_context = torch.cat([
                ticker_features[:, i],
                market_context[:, i],
                portfolio_features
            ], dim=1)
            
            # Get discrete action probabilities
            discrete_logits = self.discrete_heads[i](ticker_context)
            discrete_probs_i = F.softmax(discrete_logits / exploration_temp, dim=1)
            discrete_probs.append(discrete_probs_i)
            
            # Get allocation probabilities
            allocation_logits = self.allocation_heads[i](ticker_context)
            allocation_probs_i = torch.sigmoid(allocation_logits * risk_preference)
            allocation_probs.append(allocation_probs_i)
        
        # Stack probabilities
        discrete_probs = torch.stack(discrete_probs, dim=1)  # [batch_size, num_tickers, 3]
        allocation_probs = torch.cat(allocation_probs, dim=1)  # [batch_size, num_tickers]
        
        # Apply minimum action probability
        discrete_probs = discrete_probs * (1 - self.min_action_prob) + self.min_action_prob / 3
        
        # Normalize allocation probabilities
        allocation_probs = allocation_probs / (allocation_probs.sum(dim=1, keepdim=True) + 1e-6)
        
        # Remove batch dimension if input was 1D
        if state.size(0) == 1:
            discrete_probs = discrete_probs.squeeze(0)
            allocation_probs = allocation_probs.squeeze(0)
        
        return discrete_probs, allocation_probs