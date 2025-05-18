# EgalTrader v1: Adaptive Portfolio Management Agent

A sophisticated deep reinforcement learning-based trading agent that combines market regime detection, multi-style trading strategies, and advanced forecasting for optimal portfolio management.

## 🌟 Key Features

### Advanced Market Analysis
- Real-time market regime detection using volatility and trend analysis
- Multi-dimensional market feature extraction including price, volume, and correlation metrics
- Integration of technical indicators and market sentiment

### Intelligent Trading Strategies
- Three distinct trading styles: aggressive, moderate, and conservative
- Dynamic style adaptation based on market conditions
- Risk-aware position sizing and portfolio allocation
- Transaction cost optimization and drawdown protection

### Deep Learning Architecture
- Actor-Critic network with ticker embeddings
- Multi-head attention for cross-asset interactions
- Dedicated forecast processing branch for predictive signals
- Adaptive exploration and risk preferences

### Forecasting Capabilities
- TiDE (Time-series Dense Encoder) model for price prediction
- Multi-horizon forecasting with confidence metrics
- Integration of market seasonality and calendar effects
- Automated hyperparameter optimization using Optuna

### Meta-Learning & Adaptation
- Strategy memory bank for regime-specific optimization
- Experience replay with prioritized sampling
- Adaptive parameter tuning based on market conditions
- Performance tracking across different market regimes

## 🛠️ Technical Architecture

### Environment
- Custom OpenAI Gym environment for portfolio management
- Support for multiple assets and trading frequencies
- Real-time data processing and state management
- Flexible reward function incorporating multiple objectives

### Agent Components
- Feature Extractor: Processes market data and forecasts
- Actor Network: Generates discrete actions and allocation weights
- Critic Network: Evaluates state-action values
- Meta-Learner: Adapts trading strategies to market conditions

### Risk Management
- Position size limits and leverage constraints
- Dynamic stop-loss and take-profit mechanisms
- Portfolio diversification metrics
- Volatility-adjusted position sizing

## 📊 Performance Metrics

The agent tracks various performance metrics including:
- Portfolio returns and Sharpe ratio
- Maximum drawdown and recovery time
- Strategy adaptation success rate
- Regime-specific performance metrics

## 🚀 Getting Started

### Prerequisites

1. Install .NET SDK:
```bash
# .NET 8.0 or above is required for yfinance data acquisition
# Download and install from: https://dotnet.microsoft.com/download/dotnet/8.0
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

The agent requires two training phases:

1. Train the forecasting model:
```bash
# Train the TiDE forecasting model
python forecasting.py
```
This will:
- Train the TiDE model using Optuna for hyperparameter optimization
- Save the best model to `trained_models/TiDE_optuna_best_trained`
- Generate and save historical forecasts for training data

2. Train the trading agent:
```bash
# Train and evaluate the trading agent
python RL_training.py
```
This will:
- Load the trained forecasting model
- Initialize the portfolio environment and agent
- Train the agent using the forecasting signals
- Evaluate performance on validation data

The agent can then be used for live trading by importing the trained models:
```python
from portfolio_agent import PortfolioAgent
from RL_environment import PortfolioEnv

# Initialize for live trading
env = PortfolioEnv(tickers=['NVDA', 'FTNT'], mode='eval')
env.start_live_trading(initial_capital=10000)
agent = PortfolioAgent(env)
agent.load_models()  # Load trained models
```

## 📈 Live Trading Support

The agent supports live trading with:
- Real-time market data integration
- Portfolio state management
- Risk monitoring and position tracking

## 🔧 Configuration

Key parameters can be configured including:
- Trading universe and constraints
- Risk preferences and position limits
- Training hyperparameters
- Market regime thresholds

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are made by:
- Anand Aji
- Siddharth Arora
- Dhruv Joshi
- Wai Wing Tang
