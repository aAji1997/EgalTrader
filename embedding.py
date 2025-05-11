import torch
import torch.nn as nn
import torch.nn.functional as F


class TickerEmbedding(nn.Module):
    def __init__(self, num_tickers, embedding_dim=64):
        super(TickerEmbedding, self).__init__()

        self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.ticker_embedding.weight)
    
    def forward(self, ticker_indices):
        return self.ticker_embedding(ticker_indices)

class FeatureExtractor(nn.Module):
    def __init__(self, num_features, hidden_size=512, embedding_dim=64):
        super(FeatureExtractor, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        
        # Layer normalization and dropout
        self.input_ln = nn.LayerNorm(num_features)
        self.feature_dropout = nn.Dropout(0.1)
        
        print(f"FeatureExtractor initialized with {num_features} input features")
        
        # Separate forecast features from other features
        self.forecast_features = ['Forecast_Close', 'Forecast_Return', 'Forecast_Momentum', 'Forecast_Confidence']
        self.num_forecast_features = len(self.forecast_features)
        self.num_other_features = num_features - self.num_forecast_features
        
        # Dedicated forecast processing branch
        self.forecast_net = nn.Sequential(
            nn.Linear(self.num_forecast_features + embedding_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        # Main branch for other features
        self.main_net = nn.Sequential(
            nn.Linear(self.num_other_features + embedding_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        # Attention mechanism for forecast features
        self.forecast_attention = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Cross-feature interaction layers
        self.cross_feature_fc = nn.Linear(hidden_size, hidden_size)
        self.cross_feature_ln = nn.LayerNorm(hidden_size)
        
        # Multi-head self-attention for cross-ticker interactions
        self.num_heads = 4
        self.head_dim = hidden_size // (2 * self.num_heads)
        self.mha = nn.MultiheadAttention(hidden_size, self.num_heads, dropout=0.1)
        
        # Final processing layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        # Gating mechanism for feature selection
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, x, ticker_embeddings=None):
        """Forward pass with separate forecast processing"""
        # Ensure input is properly shaped and typed
        x = x.to(torch.float32)
        
        # Handle empty tensor
        if x.size(0) == 0:
            raise ValueError(f"Empty input tensor with shape: {x.shape}")
        
        # Handle 1D input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Use LayerNorm and dropout on input
        x = self.input_ln(x)
        x = self.feature_dropout(x)
        
        # Split features into forecast and other features
        forecast_idx = slice(-self.num_forecast_features, None)
        other_idx = slice(None, -self.num_forecast_features)
        
        forecast_features = x[:, forecast_idx]
        other_features = x[:, other_idx]
        
        # Process forecast features with embeddings
        if ticker_embeddings is not None:
            if ticker_embeddings.dim() == 3:
                ticker_embeddings = ticker_embeddings.mean(dim=1)
            elif ticker_embeddings.dim() == 2 and ticker_embeddings.size(0) != x.size(0):
                ticker_embeddings = ticker_embeddings.mean(dim=0, keepdim=True).expand(x.size(0), -1)
            
            forecast_input = torch.cat([forecast_features, ticker_embeddings], dim=-1)
            other_input = torch.cat([other_features, ticker_embeddings], dim=-1)
        else:
            forecast_input = forecast_features
            other_input = other_features
        
        # Process through separate branches
        forecast_out = self.forecast_net(forecast_input)
        other_out = self.main_net(other_input)
        
        # Calculate attention weights for forecast features
        forecast_attention = self.forecast_attention(forecast_out)
        
        # Apply attention to forecast features
        forecast_out = forecast_out * forecast_attention
        
        # Combine features
        combined = torch.cat([other_out, forecast_out], dim=-1)
        
        # Cross-feature interactions
        cross_features = self.cross_feature_ln(self.cross_feature_fc(combined))
        
        # Multi-head attention for cross-ticker interactions
        if combined.dim() == 2:
            combined = combined.unsqueeze(0)
        
        attended_features, _ = self.mha(
            cross_features.unsqueeze(0),
            cross_features.unsqueeze(0),
            cross_features.unsqueeze(0)
        )
        
        # Remove sequence dimension if added
        if attended_features.dim() == 3:
            attended_features = attended_features.squeeze(0)
        
        # Final processing with gating
        gates = self.feature_gate(attended_features)
        x = F.elu(self.ln1(self.fc1(attended_features * gates)))
        
        return x