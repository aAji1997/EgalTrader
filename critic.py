import torch.nn.functional as F
import torch
import torch.nn as nn
import gymnasium as gym

from embedding import TickerEmbedding, FeatureExtractor

class Critic(nn.Module):
    def __init__(self, env: gym.Env, hidden_size=512, embedding_dim=64):
        super(Critic, self).__init__()
        self.num_features = env.observation_space.shape[0]
        self.num_tickers = env.num_tickers
        self.embedding_dim = embedding_dim
        
        # Ticker embeddings
        self.ticker_embeddings = TickerEmbedding(env.num_tickers, embedding_dim)
        
        # Feature extractors with embeddings
        self.feature_extractor = FeatureExtractor(self.num_features + embedding_dim, hidden_size, embedding_dim)
        
        # Cross-ticker attention for value estimation
        self.value_attention = nn.MultiheadAttention(hidden_size, 4, dropout=0.1)
        self.value_ln = nn.LayerNorm(hidden_size)
        
        # Enhanced value estimation network
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Double size to include attention context
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Ticker-specific value biases
        self.ticker_value_biases = nn.Parameter(torch.zeros(env.num_tickers))
    
    def forward(self, state):
        """
        Forward pass with proper dimension handling
        Args:
            state: Input state tensor [batch_size, total_features]
        Returns:
            value: Value estimate [batch_size, 1]
        """
        batch_size = state.size(0)
        
        # Generate ticker embeddings [num_tickers, embedding_dim]
        ticker_indices = torch.arange(self.num_tickers, device=state.device)
        ticker_embeddings = self.ticker_embeddings(ticker_indices)
        
        # Average embedding for global state [1, embedding_dim]
        global_embedding = ticker_embeddings.mean(dim=0, keepdim=True)
        
        # Combine state with global embedding
        state_with_embedding = torch.cat([
            state,
            global_embedding.expand(batch_size, -1)  # [batch_size, embedding_dim]
        ], dim=-1)
        
        # Extract features with original ticker embeddings [batch_size, hidden_size]
        features = self.feature_extractor(state_with_embedding, ticker_embeddings)
        
        # Add sequence dimension for attention [1, batch_size, hidden_size]
        features_for_attention = features.unsqueeze(0)
        
        # Apply cross-ticker attention for value estimation
        attended_features, _ = self.value_attention(
            features_for_attention,
            features_for_attention,
            features_for_attention
        )  # Output: [1, batch_size, hidden_size]
        
        # Remove sequence dimension [batch_size, hidden_size]
        attended_features = attended_features.squeeze(0)
        attended_features = self.value_ln(attended_features)
        
        # Combine original and attended features [batch_size, hidden_size * 2]
        combined_features = torch.cat([features, attended_features], dim=-1)
        
        # Generate value estimate with ticker-specific biases [batch_size, 1]
        value = self.value_head(combined_features)
        value = value + self.ticker_value_biases.sum().unsqueeze(0)
        
        return value