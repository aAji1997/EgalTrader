import torch
from RL_training import RLTrainer
from RL_environment import PortfolioEnv
import os
from datetime import datetime
import json
import numpy as np

class LiveTrader:
    def __init__(self, tickers, initial_capital, model_dir="memory"):
        """
        Initialize the live trading system.
        
        Args:
            tickers (list): List of ticker symbols to trade
            initial_capital (float): Initial capital to start trading with
            model_dir (str): Directory containing the trained model files
        """
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.model_dir = model_dir
        
        # Initialize environment and trainer
        self.env = PortfolioEnv(tickers=tickers, initial_capital=initial_capital)
        self.trainer = RLTrainer(save_dir=model_dir)
        
        # Trading state
        self.is_trading = False
        self.current_positions = {ticker: 0 for ticker in tickers}
        self.remaining_capital = initial_capital
        
        # Trading history
        self.trade_history = []
        
        # Ensure the figures directory exists for saving trade logs
        self.figures_dir = "figures"
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
    
    def start_trading(self):
        """Start live trading with the current portfolio state."""
        try:
            # Verify model exists
            best_model_path = os.path.join(self.model_dir, 'best_model.pth')
            if not os.path.exists(best_model_path):
                raise FileNotFoundError("No trained model found. Please train the model first.")
            
            # Start live trading mode in environment
            self.env.start_live_trading(initial_capital=self.initial_capital)
            
            # Update environment with current positions if any
            positions = [self.current_positions[ticker] for ticker in self.tickers]
            self.env.update_live_portfolio_state(
                positions=positions,
                remaining_capital=self.remaining_capital
            )
            
            self.is_trading = True
            print(f"\nLive trading started with:")
            print(f"Initial capital: ${self.initial_capital:,.2f}")
            print(f"Tickers: {', '.join(self.tickers)}")
            
        except Exception as e:
            print(f"Error starting live trading: {str(e)}")
            self.is_trading = False
    
    def get_next_trades(self, execute=True):
        """
        Get the optimal trades for the next trading day and optionally execute them.
        
        Args:
            execute (bool): Whether to automatically execute the trades
        
        Returns:
            dict: Dictionary of trades for each ticker
            None: If there's an error or trading hasn't started
        """
        if not self.is_trading:
            print("Error: Live trading has not been started. Call start_trading() first.")
            return None
        
        try:
            # Get trades from trainer
            trades = self.trainer.get_optimal_live_next_day_trades()
            
            if trades:
                # Log the trades
                self._log_trades(trades)
                
                # Execute trades if requested
                if execute:
                    self.execute_trades(trades)
                
                return trades
            
            return None
            
        except Exception as e:
            print(f"Error getting next trades: {str(e)}")
            return None
    
    def execute_trades(self, trades):
        """
        Execute the suggested trades by calculating new positions and remaining capital.
        
        Args:
            trades (dict): Dictionary of trades for each ticker from get_optimal_live_next_day_trades
        """
        try:
            # Get current prices from environment
            current_prices = self.env.get_current_prices()
            
            # Calculate total portfolio value
            portfolio_value = self.remaining_capital
            for ticker, shares in self.current_positions.items():
                idx = self.env.ticker_to_idx[ticker]
                portfolio_value += shares * current_prices[idx]
            
            new_positions = {}
            total_cost = 0
            
            # Process each trade
            for ticker, trade_info in trades.items():
                idx = self.env.ticker_to_idx[ticker]
                price = current_prices[idx]
                
                if trade_info['action'] == 'buy':
                    # Calculate shares to buy based on allocation
                    target_value = portfolio_value * trade_info.get('allocation', 0)
                    shares_to_buy = int(target_value / price)  # Round down to whole shares
                    cost = shares_to_buy * price
                    new_positions[ticker] = self.current_positions[ticker] + shares_to_buy
                    total_cost += cost
                    
                elif trade_info['action'] == 'sell':
                    # Calculate shares to sell based on allocation
                    current_shares = self.current_positions[ticker]
                    target_value = portfolio_value * trade_info.get('allocation', 0)
                    target_shares = int(target_value / price)  # Round down to whole shares
                    shares_to_sell = current_shares - target_shares
                    proceeds = shares_to_sell * price
                    new_positions[ticker] = target_shares
                    total_cost -= proceeds
                    
                else:  # hold
                    new_positions[ticker] = self.current_positions[ticker]
            
            # Update portfolio state
            new_remaining_capital = self.remaining_capital - total_cost
            self.update_portfolio(positions=new_positions, remaining_capital=new_remaining_capital)
            
            print("\nTrades executed:")
            for ticker, trade_info in trades.items():
                old_shares = self.current_positions[ticker]
                new_shares = new_positions[ticker]
                change = new_shares - old_shares
                if change != 0:
                    action = "bought" if change > 0 else "sold"
                    print(f"{ticker}: {action} {abs(change):,} shares")
            
        except Exception as e:
            print(f"Error executing trades: {str(e)}")
    
    def update_portfolio(self, positions=None, remaining_capital=None):
        """
        Update the current portfolio state after trades are executed.
        
        Args:
            positions (dict): Dictionary of current positions {ticker: shares}
            remaining_capital (float): Current remaining capital
        """
        if not self.is_trading:
            print("Error: Live trading has not been started. Call start_trading() first.")
            return
        
        try:
            # Update internal state
            if positions is not None:
                self.current_positions.update(positions)
            if remaining_capital is not None:
                self.remaining_capital = remaining_capital
            
            # Update environment state
            position_list = [self.current_positions[ticker] for ticker in self.tickers]
            self.env.update_live_portfolio_state(
                positions=position_list,
                remaining_capital=self.remaining_capital
            )
            
            # Calculate and print portfolio summary
            self._print_portfolio_summary()
            
        except Exception as e:
            print(f"Error updating portfolio: {str(e)}")
    
    def _log_trades(self, trades):
        """Log trades to trade history and save to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trade_log = {
            'timestamp': timestamp,
            'trades': trades,
            'portfolio_state': {
                'positions': self.current_positions.copy(),
                'remaining_capital': float(self.remaining_capital)
            }
        }
        
        # Add to history
        self.trade_history.append(trade_log)
        
        # Save to file
        log_file = os.path.join(self.figures_dir, f'trade_log_{timestamp}.json')
        with open(log_file, 'w') as f:
            json.dump(trade_log, f, indent=4)
    
    def _print_portfolio_summary(self):
        """Print current portfolio state summary."""
        print("\nCurrent Portfolio Summary:")
        print("-" * 50)
        print(f"Remaining Capital: ${self.remaining_capital:,.2f}")
        print("\nPositions:")
        for ticker, shares in self.current_positions.items():
            print(f"{ticker}: {shares:,} shares")
        
        # Get current portfolio value from environment
        portfolio_value = self.env.prev_portfolio_value
        print(f"\nTotal Portfolio Value: ${portfolio_value:,.2f}")
    
    def stop_trading(self):
        """Stop live trading and save final state."""
        if not self.is_trading:
            return
        
        try:
            # Save final trading summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary = {
                'final_state': {
                    'positions': self.current_positions,
                    'remaining_capital': float(self.remaining_capital),
                    'portfolio_value': float(self.env.prev_portfolio_value)
                },
                'trade_history': self.trade_history
            }
            
            summary_file = os.path.join(self.figures_dir, f'trading_summary_{timestamp}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print("\nLive trading stopped")
            print(f"Final trading summary saved to: {summary_file}")
            
            self.is_trading = False
            
        except Exception as e:
            print(f"Error stopping live trading: {str(e)}") 


if __name__ == "__main__":
    # Initialize
    trader = LiveTrader(
        tickers=['NVDA', 'FTNT'],
        initial_capital=10_000
    )

    # Start trading
    trader.start_trading()

    # Or get trades without executing
    trades = trader.get_next_trades(execute=False)
    print("Trades to execute:")
    print(trades)
    # Later execute them if desired
    trader.execute_trades(trades)

    # Stop trading when done
    trader.stop_trading()