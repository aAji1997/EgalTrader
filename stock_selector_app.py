import streamlit as st
import json
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from trading_recommender import TradingRecommender
from forecasting import StockAnalyzer
import sys
import subprocess
import threading
from progress_tracker import progress_tracker
import datetime
from evaluate_pretrained_model import evaluate_pretrained_model

# Print debug information
print("Starting Stock Selector App...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Streamlit version: {st.__version__}")

# Set page configuration
st.set_page_config(
    page_title="Stock Portfolio Selector",
    page_icon="📈",
    layout="wide"
)

os.makedirs("trained_models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("progress", exist_ok=True)

# Reset progress tracker at app startup
progress_tracker.reset()

# Function to get stock recommendations
def get_recommendations(sector):
    recommender = TradingRecommender()
    recommendations_json = recommender.get_recommendations(sector, return_json=True)
    return recommendations_json

# Function to update metadata files with selected tickers
def update_metadata_files(tickers):
    """
    Update metadata files with the selected tickers and calculate their returns.

    Args:
        tickers: List of ticker symbols
    """
    try:
        # Get paths to metadata files
        memory_dir = "memory"
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        # Find all metadata files
        metadata_files = [f for f in os.listdir(memory_dir) if f.endswith("_metadata.json")]

        # If no metadata files exist, create a default one
        if not metadata_files:
            metadata_files = ["best_model_metadata.json"]

        # Calculate ticker-specific returns
        analyzer = StockAnalyzer(tickers=tickers, period='max')
        ticker_returns = {}

        # Get the most recent data for each ticker
        for ticker in tickers:
            try:
                # Get historical data for the ticker
                ticker_data = analyzer.data[analyzer.data['ticker'] == ticker]

                if not ticker_data.empty:
                    # Calculate return based on first and last available prices
                    first_price = ticker_data.iloc[0]['Close']
                    last_price = ticker_data.iloc[-1]['Close']
                    pct_return = ((last_price - first_price) / first_price) * 100
                    ticker_returns[ticker] = round(pct_return, 2)
                else:
                    ticker_returns[ticker] = 0.0
            except Exception as e:
                print(f"Error calculating return for {ticker}: {str(e)}")
                ticker_returns[ticker] = 0.0

        # Update each metadata file
        for metadata_file in metadata_files:
            file_path = os.path.join(memory_dir, metadata_file)
            try:
                # Check if file exists
                if os.path.exists(file_path):
                    # Read existing metadata
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Create new metadata with default values
                    metadata = {
                        "tickers": [],
                        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "initial_capital": 10000,
                        "model_version": "1.0"
                    }

                # Update tickers and ticker_returns
                metadata['tickers'] = tickers
                metadata['ticker_returns'] = ticker_returns

                # Write updated metadata
                with open(file_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                print(f"Updated metadata file: {metadata_file}")
            except Exception as e:
                print(f"Error updating metadata file {metadata_file}: {str(e)}")

    except Exception as e:
        print(f"Error updating metadata files: {str(e)}")

# Function to train the StockAnalyzer
def train_stock_analyzer(tickers):
    st.session_state.training_status = "Training forecasting model..."

    # Initialize overall progress
    import time
    progress_tracker.update_overall_progress(
        status="running",
        message="Training forecasting model...",
        progress=0.0,
        current_phase=1,
        start_time=time.time()
    )

    # Update metadata files with selected tickers
    update_metadata_files(tickers)

    analyzer = StockAnalyzer(tickers=tickers, period='max')
    analyzer.train_stock_model_darts()

    st.session_state.training_status = "Forecasting model trained successfully!"
    return analyzer

# Function to train the RL agent
def train_rl_agent(tickers, initial_capital=10000):
    st.session_state.training_status = "Training RL agent..."

    # Update overall progress
    progress_tracker.update_overall_progress(
        message="Training RL agent...",
        current_phase=2,
        progress=0.5  # Forecasting is complete, now at 50%
    )

    # Import here to avoid circular imports
    from RL_training import RLTrainer
    trainer = RLTrainer(
        eval_frequency=10,
        save_dir="memory",
        training_batch_size=64,
        eval_batch_size=32,
        rollout_episodes=10,
        initial_capital=initial_capital,
        tickers=tickers  # Pass the selected tickers to the RLTrainer
    )
    scores, final_metrics = trainer.training_loop()

    # Store the trainer in session state for visualization
    st.session_state.trainer = trainer
    st.session_state.final_metrics = final_metrics

    st.session_state.training_status = "RL agent trained successfully!"

    return trainer, final_metrics

# Function to create interactive portfolio performance visualizations
def create_portfolio_visualizations():
    """Create interactive visualizations of portfolio performance"""
    if not hasattr(st.session_state, 'trainer') or st.session_state.trainer is None:
        st.warning("No training data available for visualization.")
        return

    trainer = st.session_state.trainer

    # Create a container for visualizations
    viz_container = st.container()

    with viz_container:
        st.subheader("Portfolio Performance Visualization")

        # Get evaluation data from the trainer
        if hasattr(trainer, 'final_eval_scores') and trainer.final_eval_scores:
            latest_eval = trainer.final_eval_scores[-1]

            # Extract portfolio return and buy-and-hold return
            portfolio_return = latest_eval.get('portfolio_return', 0.0)
            relative_return = latest_eval.get('relative_return', 0.0)
            buy_and_hold_return = portfolio_return - relative_return

            # Create comparison chart
            comparison_data = pd.DataFrame({
                'Strategy': ['RL Agent', 'Buy & Hold'],
                'Return (%)': [portfolio_return, buy_and_hold_return]
            })

            # Use Plotly for interactive bar chart
            fig1 = px.bar(
                comparison_data,
                x='Strategy',
                y='Return (%)',
                color='Strategy',
                title='Strategy Performance Comparison',
                color_discrete_map={'RL Agent': '#1f77b4', 'Buy & Hold': '#ff7f0e'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

            # Create metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Return", f"{portfolio_return:.2f}%", f"{relative_return:+.2f}% vs Buy & Hold")
            with col2:
                st.metric("Sharpe Ratio", f"{latest_eval.get('sharpe', 0.0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{latest_eval.get('max_drawdown', 0.0):.2%}")

            # Check if we have portfolio values over time
            if hasattr(trainer, 'episode_values') and len(trainer.episode_values) > 0:
                # Create a dataframe for the portfolio values
                portfolio_values = np.array(trainer.episode_values)

                # Calculate buy and hold values (assuming linear growth based on final return)
                initial_value = portfolio_values[0]
                final_value = portfolio_values[-1]
                steps = len(portfolio_values)

                # Generate buy and hold values
                buy_hold_values = []
                buy_hold_return = buy_and_hold_return / 100  # Convert from percentage to decimal

                # Linear approximation of buy and hold strategy
                for i in range(steps):
                    # Linear interpolation between initial value and calculated final value
                    t = i / (steps - 1) if steps > 1 else 0
                    buy_hold_value = initial_value * (1 + buy_hold_return * t)
                    buy_hold_values.append(buy_hold_value)

                # Create a dataframe for both strategies
                portfolio_df = pd.DataFrame({
                    'Time Step': range(len(portfolio_values)),
                    'RL Agent': portfolio_values,
                    'Buy & Hold': buy_hold_values
                })

                # Create interactive line chart with both strategies
                fig2 = px.line(
                    portfolio_df,
                    x='Time Step',
                    y=['RL Agent', 'Buy & Hold'],
                    title='Portfolio Value Over Time: RL Agent vs Buy & Hold',
                    labels={'value': 'Portfolio Value ($)', 'variable': 'Strategy'},
                    color_discrete_map={'RL Agent': '#1f77b4', 'Buy & Hold': '#ff7f0e'}
                )

                # Add hover information
                fig2.update_traces(
                    hovertemplate='<b>%{y:$,.2f}</b><br>Time Step: %{x}<extra></extra>'
                )

                fig2.update_layout(
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Add a toggle for logarithmic scale
                if st.checkbox("Show logarithmic scale"):
                    fig2.update_layout(yaxis_type="log")
                    st.plotly_chart(fig2, use_container_width=True)

                # Calculate and show cumulative returns
                st.subheader("Cumulative Returns")

                # Calculate cumulative returns for both strategies
                rl_cum_returns = [(val / initial_value - 1) * 100 for val in portfolio_values]
                bh_cum_returns = [(val / initial_value - 1) * 100 for val in buy_hold_values]

                # Create dataframe for cumulative returns
                returns_df = pd.DataFrame({
                    'Time Step': range(len(portfolio_values)),
                    'RL Agent': rl_cum_returns,
                    'Buy & Hold': bh_cum_returns
                })

                # Create interactive line chart for cumulative returns
                fig3 = px.line(
                    returns_df,
                    x='Time Step',
                    y=['RL Agent', 'Buy & Hold'],
                    title='Cumulative Returns (%): RL Agent vs Buy & Hold',
                    labels={'value': 'Return (%)', 'variable': 'Strategy'},
                    color_discrete_map={'RL Agent': '#1f77b4', 'Buy & Hold': '#ff7f0e'}
                )

                # Add hover information
                fig3.update_traces(
                    hovertemplate='<b>%{y:.2f}%</b><br>Time Step: %{x}<extra></extra>'
                )

                fig3.update_layout(
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig3, use_container_width=True)

            # If we have regime transitions, show them
            if 'regime_transitions' in latest_eval and latest_eval['regime_transitions']:
                st.subheader("Market Regime Transitions")
                regime_df = pd.DataFrame(latest_eval['regime_transitions'])
                if not regime_df.empty:
                    st.dataframe(regime_df[['from_regime', 'to_regime', 'step']])

            # If we have trading actions, show distribution
            if hasattr(trainer, 'trading_actions_summary') and trainer.trading_actions_summary:
                st.subheader("Trading Actions Distribution")
                actions_df = pd.DataFrame(trainer.trading_actions_summary)
                if not actions_df.empty:
                    fig3 = px.bar(
                        actions_df,
                        x='ticker',
                        y=['buys', 'holds', 'sells'],
                        title='Trading Actions by Ticker',
                        barmode='group'
                    )
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)

            # Add drawdown visualization if we have portfolio values
            if hasattr(trainer, 'episode_values') and len(trainer.episode_values) > 0:
                st.subheader("Portfolio Drawdown")

                # Calculate drawdown for both strategies
                def calculate_drawdown(values):
                    peak = np.maximum.accumulate(values)
                    drawdown = -((values - peak) / peak) * 100  # Convert to percentage and make positive
                    return drawdown

                portfolio_values = np.array(trainer.episode_values)
                initial_value = portfolio_values[0]

                # Generate buy and hold values (reusing from earlier)
                buy_hold_values = []
                buy_hold_return = (portfolio_return - relative_return) / 100  # Convert from percentage to decimal

                # Linear approximation of buy and hold strategy
                steps = len(portfolio_values)
                for i in range(steps):
                    # Linear interpolation between initial value and calculated final value
                    t = i / (steps - 1) if steps > 1 else 0
                    buy_hold_value = initial_value * (1 + buy_hold_return * t)
                    buy_hold_values.append(buy_hold_value)

                rl_drawdown = calculate_drawdown(portfolio_values)
                bh_drawdown = calculate_drawdown(np.array(buy_hold_values))

                # Create dataframe for drawdown
                drawdown_df = pd.DataFrame({
                    'Time Step': range(len(portfolio_values)),
                    'RL Agent': rl_drawdown,
                    'Buy & Hold': bh_drawdown
                })

                # Create interactive line chart for drawdown
                fig4 = px.line(
                    drawdown_df,
                    x='Time Step',
                    y=['RL Agent', 'Buy & Hold'],
                    title='Portfolio Drawdown (%): RL Agent vs Buy & Hold',
                    labels={'value': 'Drawdown (%)', 'variable': 'Strategy'},
                    color_discrete_map={'RL Agent': '#1f77b4', 'Buy & Hold': '#ff7f0e'}
                )

                # Add hover information
                fig4.update_traces(
                    hovertemplate='<b>%{y:.2f}%</b><br>Time Step: %{x}<extra></extra>'
                )

                # Invert y-axis to show drawdown as negative values going down
                fig4.update_layout(
                    height=500,
                    yaxis=dict(autorange="reversed"),  # Invert y-axis
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig4, use_container_width=True)

# Function to run training in a separate thread
def run_training_thread(tickers, initial_capital=10000):
    try:
        # Train the StockAnalyzer
        analyzer = train_stock_analyzer(tickers)

        # Train the RL agent with the specified initial capital and tickers
        trainer, final_metrics = train_rl_agent(tickers, initial_capital)

        # Update overall progress to completed
        import time
        progress_tracker.update_overall_progress(
            status="completed",
            message="Training completed successfully!",
            progress=1.0,
            end_time=time.time()
        )

        st.session_state.training_complete = True
    except Exception as e:
        error_message = f"Error during training: {str(e)}"
        st.session_state.training_status = error_message
        st.session_state.training_error = True

        # Update progress trackers with error
        import time
        progress_tracker.update_overall_progress(
            status="error",
            message=error_message,
            error=str(e),
            end_time=time.time()
        )

        # Update the current phase tracker with error
        current_phase = progress_tracker.get_overall_progress().get("current_phase", 1)
        if current_phase == 1:
            progress_tracker.update_forecaster_progress(
                status="error",
                message=error_message,
                error=str(e),
                end_time=time.time()
            )
        else:
            progress_tracker.update_rl_agent_progress(
                status="error",
                message=error_message,
                error=str(e),
                end_time=time.time()
            )

# Function to evaluate a pretrained model
def evaluate_pretrained_model_ui(model_path, tickers, start_date, end_date, initial_capital=10000):
    st.session_state.evaluation_status = "Evaluating pretrained model..."

    # Update overall progress
    progress_tracker.update_overall_progress(
        status="running",
        message="Evaluating pretrained model...",
        progress=0.0,
        current_phase=3,  # Using phase 3 for evaluation
        start_time=time.time()
    )

    try:
        # Check if user wants to use the model's tickers
        use_model_tickers = False
        if not tickers or len(tickers) == 0 or (len(tickers) == 1 and tickers[0].strip() == ""):
            use_model_tickers = True
            # Extract tickers from the model
            from evaluate_pretrained_model import extract_tickers_from_model
            model_tickers = extract_tickers_from_model(model_path)
            if model_tickers:
                tickers = model_tickers
            else:
                tickers = ['INTC', 'HPE']  # Default tickers

        # Update metadata files with the evaluation tickers
        if not use_model_tickers:
            update_metadata_files(tickers)

        # Call the evaluation function
        results = evaluate_pretrained_model(
            model_path=model_path,
            tickers=None if use_model_tickers else tickers,  # Pass None to use model's tickers
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            eval_episodes=10
        )

        if not results.get('success', False):
            st.session_state.evaluation_status = f"Error: {results.get('error', 'Unknown error')}"
            st.session_state.evaluation_error = True
            return None

        # Store results in session state for visualization
        st.session_state.evaluation_results = results
        st.session_state.trainer = type('obj', (object,), {
            'final_eval_scores': results.get('final_eval_scores', []),
            'episode_values': results.get('portfolio_values', [])
        })
        st.session_state.final_metrics = (
            results.get('mean_score', 0),
            results.get('sharpe_ratio', 0),
            results.get('max_drawdown', 0)
        )

        # Store the tickers used for evaluation
        if 'tickers' in results:
            st.session_state.evaluation_tickers = results['tickers']
            tickers_str = ', '.join(results['tickers'])
            st.session_state.evaluation_status = f"Model evaluation completed successfully on tickers: {tickers_str}!"
        else:
            st.session_state.evaluation_status = "Model evaluation completed successfully!"

        st.session_state.evaluation_complete = True

        return results
    except Exception as e:
        error_message = f"Error during model evaluation: {str(e)}"
        st.session_state.evaluation_status = error_message
        st.session_state.evaluation_error = True

        # Update progress trackers with error
        progress_tracker.update_overall_progress(
            status="error",
            message=error_message,
            error=str(e),
            end_time=time.time()
        )

        return None

# Function to run evaluation in a separate thread
def run_evaluation_thread(model_path, tickers, start_date, end_date, initial_capital=10000):
    try:
        # Evaluate the pretrained model
        results = evaluate_pretrained_model_ui(
            model_path=model_path,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )

        # Update overall progress to completed
        if results and 'tickers' in results:
            # Show which tickers were actually used for evaluation
            tickers_used = results['tickers']
            completion_message = f"Evaluation completed successfully on tickers: {', '.join(tickers_used)}!"
        else:
            completion_message = "Evaluation completed successfully!"

        progress_tracker.update_overall_progress(
            status="completed",
            message=completion_message,
            progress=1.0,
            end_time=time.time()
        )

        st.session_state.evaluation_complete = True
    except Exception as e:
        error_message = f"Error during evaluation: {str(e)}"
        st.session_state.evaluation_status = error_message
        st.session_state.evaluation_error = True

        # Update progress trackers with error
        progress_tracker.update_overall_progress(
            status="error",
            message=error_message,
            error=str(e),
            end_time=time.time()
        )

# Initialize session state variables
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []
if 'training_status' not in st.session_state:
    st.session_state.training_status = ""
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'training_error' not in st.session_state:
    st.session_state.training_error = False
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'evaluation_status' not in st.session_state:
    st.session_state.evaluation_status = ""
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'evaluation_error' not in st.session_state:
    st.session_state.evaluation_error = False
if 'evaluation_started' not in st.session_state:
    st.session_state.evaluation_started = False

# Main app layout
st.title("Stock Portfolio Selector")

# Sidebar for app mode selection
st.sidebar.header("App Mode")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Train New Model", "Evaluate Pretrained Model"],
    help="Choose whether to train a new model or evaluate a pretrained one"
)

if app_mode == "Train New Model":
    # Sidebar for sector selection and initial capital
    st.sidebar.header("Select Sector/Industry")
    sector_options = [
        "Technology",
        "Healthcare",
        "Financial",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Industrial",
        "Energy",
        "Utilities",
        "Real Estate",
        "Communication Services",
        "Basic Materials"
    ]
    selected_sector = st.sidebar.selectbox("Choose a sector", sector_options)

    # Add initial capital input
    st.sidebar.header("Investment Settings")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000, help="The amount of money you wish to invest initially")

    # Button to get recommendations
    if st.sidebar.button("Get Stock Recommendations"):
        with st.spinner(f"Fetching recommendations for {selected_sector} sector..."):
            st.session_state.recommendations = get_recommendations(selected_sector)
else:  # Evaluate Pretrained Model mode
    # Sidebar for evaluation settings
    st.sidebar.header("Evaluation Settings")

    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="memory/best_model.pth",
        help="Path to the pretrained model file"
    )

    # Try to extract tickers from the model
    model_tickers = None
    try:
        from evaluate_pretrained_model import extract_tickers_from_model
        model_tickers = extract_tickers_from_model(model_path)
        if model_tickers:
            default_tickers = ",".join(model_tickers)
        else:
            default_tickers = "INTC,HPE"
    except Exception as e:
        print(f"Error extracting tickers from model: {str(e)}")
        default_tickers = "INTC,HPE"

    # Ticker selection with model tickers as default
    ticker_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value=default_tickers,
        help="Comma-separated list of ticker symbols to evaluate. By default, uses the tickers the model was trained on."
    )
    evaluation_tickers = [ticker.strip() for ticker in ticker_input.split(",")]

    # Add a note about the model's tickers
    if model_tickers:
        st.sidebar.info(f"This model was trained on: {', '.join(model_tickers)}")
    else:
        st.sidebar.info("Could not determine which tickers this model was trained on. Using default tickers.")

    # Date range selection
    st.sidebar.subheader("Evaluation Period")

    # Calculate default dates (last 2 months)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=60)

    eval_start_date = st.sidebar.date_input(
        "Start Date",
        value=start_date,
        help="Start date for evaluation period"
    )

    eval_end_date = st.sidebar.date_input(
        "End Date",
        value=end_date,
        help="End date for evaluation period"
    )

    # Initial capital
    eval_initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="The amount of money to start with for evaluation"
    )

    # Button to start evaluation
    if st.sidebar.button("Evaluate Model"):
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file not found: {model_path}")
        elif len(evaluation_tickers) == 0:
            st.sidebar.error("Please enter at least one ticker symbol")
        elif eval_start_date >= eval_end_date:
            st.sidebar.error("Start date must be before end date")
        else:
            # Convert dates to string format
            start_date_str = eval_start_date.strftime("%Y-%m-%d")
            end_date_str = eval_end_date.strftime("%Y-%m-%d")

            # Start evaluation in a separate thread
            if not st.session_state.evaluation_started:
                st.session_state.evaluation_started = True
                evaluation_thread = threading.Thread(
                    target=run_evaluation_thread,
                    args=(model_path, evaluation_tickers, start_date_str, end_date_str, eval_initial_capital)
                )
                evaluation_thread.daemon = True
                evaluation_thread.start()

# Display content based on app mode
if app_mode == "Train New Model":
    # Display recommendations and selection interface
    if st.session_state.recommendations:
        st.header(f"Recommended Stocks in {selected_sector} Sector")

        # Create columns for better layout
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Select Stocks")

            # Create a dictionary to store selections
            selections = {}

            for stock in st.session_state.recommendations:
                ticker = stock.get('ticker', '')
                name = stock.get('name', '')

                # Create a checkbox for each stock
                selections[ticker] = st.checkbox(f"{ticker} - {name}", key=f"check_{ticker}")

        with col2:
            st.subheader("Investment Rationale")

            for stock in st.session_state.recommendations:
                ticker = stock.get('ticker', '')
                name = stock.get('name', '')
                rationale = stock.get('rationale', '')

                # Display rationale in an expander
                with st.expander(f"{ticker} - {name}"):
                    st.write(rationale)

        # Button to proceed with selected stocks
        if st.button("Create Portfolio with Selected Stocks"):
            # Get selected tickers
            selected_tickers = [ticker for ticker, selected in selections.items() if selected]

            if not selected_tickers:
                st.error("Please select at least one stock to proceed.")
            else:
                st.session_state.selected_tickers = selected_tickers
                st.success(f"Selected tickers: {', '.join(selected_tickers)}")

                # Start training in a separate thread
                if not st.session_state.training_started:
                    st.session_state.training_started = True
                    training_thread = threading.Thread(target=run_training_thread, args=(selected_tickers, initial_capital))
                    training_thread.daemon = True
                    training_thread.start()
else:  # Evaluate Pretrained Model mode
    # Display evaluation information
    st.header("Pretrained Model Evaluation")

    if not st.session_state.evaluation_started:
        st.info("Configure the evaluation settings in the sidebar and click 'Evaluate Model' to begin.")

        # Display information about the evaluation process
        st.markdown("""
        ### Evaluation Process

        This mode allows you to evaluate a pretrained RL agent model over a specific time period.
        The evaluation will:

        1. Load the pretrained model from the specified path
        2. Initialize the environment with data for the selected tickers and date range
        3. Run multiple evaluation episodes to assess model performance
        4. Generate visualizations comparing the RL agent to a buy-and-hold strategy

        For a showcase, you can use the last two months of data to demonstrate how the model
        would have performed in recent market conditions without having to train from scratch.
        """)
    else:
        # Show evaluation progress or results
        if not st.session_state.evaluation_complete and not st.session_state.evaluation_error:
            st.info("Evaluation in progress... Please wait.")
        elif st.session_state.evaluation_error:
            st.error(st.session_state.evaluation_status)
        else:
            st.success("Evaluation completed successfully!")

# Function to format time
def format_time(seconds):
    if seconds is None:
        return "N/A"
    return str(datetime.timedelta(seconds=int(seconds)))

# Display progress tracking for either training or evaluation
if st.session_state.training_started or st.session_state.evaluation_started:
    # Determine which mode we're in
    is_training = st.session_state.training_started
    is_evaluation = st.session_state.evaluation_started

    if is_training:
        st.header("Training Progress")
    else:
        st.header("Evaluation Progress")

    # Create containers for progress display
    overall_status_container = st.empty()
    overall_progress_container = st.empty()
    phase_status_container = st.empty()
    phase_progress_container = st.empty()
    metrics_container = st.empty()

    # Get initial progress data
    overall_progress = progress_tracker.get_overall_progress()
    forecaster_progress = progress_tracker.get_forecaster_progress()
    rl_agent_progress = progress_tracker.get_rl_agent_progress()

    # Check if process is already complete
    if overall_progress.get("status") == "completed":
        if is_training:
            st.session_state.training_complete = True
        else:
            st.session_state.evaluation_complete = True

    # Poll progress and update UI
    while ((is_training and not st.session_state.training_complete and not st.session_state.training_error) or
           (is_evaluation and not st.session_state.evaluation_complete and not st.session_state.evaluation_error)):
        # Get progress data
        overall_progress = progress_tracker.get_overall_progress()
        forecaster_progress = progress_tracker.get_forecaster_progress()
        rl_agent_progress = progress_tracker.get_rl_agent_progress()

        # Update overall status
        overall_status = overall_progress.get("message", "Process in progress...")
        overall_status_container.info(overall_status)

        # Update overall progress bar
        overall_progress_value = overall_progress.get("progress", 0.0)
        # Ensure progress value is between 0 and 1
        overall_progress_value = max(0.0, min(1.0, overall_progress_value))
        overall_progress_container.progress(overall_progress_value)

        # Determine current phase and update phase status
        current_phase = overall_progress.get("current_phase", 0)
        if current_phase == 1:
            # Forecaster phase
            phase_status = forecaster_progress.get("message", "Training forecasting model...")
            phase_progress_value = forecaster_progress.get("progress", 0.0)

            # Display forecaster metrics
            metrics_md = """
            ### Forecasting Model Metrics
            """

            if forecaster_progress.get("current_trial") is not None:
                metrics_md += f"""
                - **Current Trial**: {forecaster_progress.get("current_trial", 0)}/{forecaster_progress.get("total_trials", 0)}
                """

            if forecaster_progress.get("best_score") is not None:
                metrics_md += f"""
                - **Best Score (SMAPE)**: {forecaster_progress.get("best_score", 0):.4f}
                """

            elapsed_time = forecaster_progress.get("elapsed_time", 0)
            metrics_md += f"""
            - **Elapsed Time**: {format_time(elapsed_time)}
            """

            metrics_container.markdown(metrics_md)

        elif current_phase == 2:
            # RL agent phase
            phase_status = rl_agent_progress.get("message", "Training RL agent...")
            phase_progress_value = rl_agent_progress.get("progress", 0.0)

            # Display RL agent metrics
            metrics_md = """
            ### RL Agent Metrics
            """

            if rl_agent_progress.get("current_episode") is not None:
                metrics_md += f"""
                - **Current Episode**: {rl_agent_progress.get("current_episode", 0)}/{rl_agent_progress.get("total_episodes", 0)}
                """

            if rl_agent_progress.get("current_score") is not None:
                metrics_md += f"""
                - **Current Score**: {rl_agent_progress.get("current_score", 0):.2f}
                """

            if rl_agent_progress.get("sharpe_ratio") is not None:
                metrics_md += f"""
                - **Sharpe Ratio**: {rl_agent_progress.get("sharpe_ratio", 0):.2f}
                """

            if rl_agent_progress.get("max_drawdown") is not None:
                metrics_md += f"""
                - **Max Drawdown**: {rl_agent_progress.get("max_drawdown", 0):.2%}
                """

            elapsed_time = rl_agent_progress.get("elapsed_time", 0)
            metrics_md += f"""
            - **Elapsed Time**: {format_time(elapsed_time)}
            """

            metrics_container.markdown(metrics_md)
        elif current_phase == 3:
            # Evaluation phase
            phase_status = overall_progress.get("message", "Evaluating model...")
            phase_progress_value = overall_progress.get("progress", 0.0)

            # Display evaluation metrics
            metrics_md = """
            ### Evaluation Metrics
            """

            elapsed_time = overall_progress.get("elapsed_time", 0)
            metrics_md += f"""
            - **Elapsed Time**: {format_time(elapsed_time)}
            """

            metrics_container.markdown(metrics_md)
        else:
            phase_status = "Initializing..."
            phase_progress_value = 0.0
            metrics_container.empty()

        # Update phase status and progress
        phase_status_container.info(phase_status)
        # Ensure progress value is between 0 and 1
        phase_progress_value = max(0.0, min(1.0, phase_progress_value))
        phase_progress_container.progress(phase_progress_value)

        # Check if process is complete
        if overall_progress.get("status") == "completed":
            if is_training:
                st.session_state.training_complete = True
            else:
                st.session_state.evaluation_complete = True
            break

        # Sleep to avoid excessive polling
        time.sleep(1)

    # Display completion message
    if (is_training and st.session_state.training_complete) or (is_evaluation and st.session_state.evaluation_complete):
        # Clear progress displays
        overall_status_container.empty()
        overall_progress_container.empty()
        phase_status_container.empty()
        phase_progress_container.empty()

        # Show final metrics
        if is_training:
            final_metrics_md = """
            ## Training Completed Successfully!

            ### Final Metrics
            """
        else:
            final_metrics_md = """
            ## Evaluation Completed Successfully!

            ### Final Metrics
            """

        # Get final RL agent metrics
        rl_agent_progress = progress_tracker.get_rl_agent_progress()
        if rl_agent_progress.get("current_score") is not None:
            final_metrics_md += f"""
            - **Portfolio Score**: {rl_agent_progress.get("current_score", 0):.2f}
            - **Sharpe Ratio**: {rl_agent_progress.get("sharpe_ratio", 0):.2f}
            - **Max Drawdown**: {rl_agent_progress.get("max_drawdown", 0):.2%}
            """

        # Get total time
        overall_progress = progress_tracker.get_overall_progress()
        total_time = overall_progress.get("elapsed_time", 0)
        final_metrics_md += f"""
        - **Total Time**: {format_time(total_time)}
        """

        metrics_container.markdown(final_metrics_md)

        if is_training:
            st.success("Training completed successfully! Your portfolio is ready for trading.")
        else:
            st.success("Evaluation completed successfully! See the performance analysis below.")

        # Display interactive portfolio performance visualizations
        st.header("Portfolio Performance Analysis")
        st.write("The following interactive charts show the performance of your portfolio compared to a simple buy-and-hold strategy.")

        # Create and display interactive visualizations
        create_portfolio_visualizations()

    # Display error message
    if (is_training and st.session_state.training_error) or (is_evaluation and st.session_state.evaluation_error):
        if is_training:
            st.error("An error occurred during training. Please check the logs for details.")
        else:
            st.error("An error occurred during evaluation. Please check the logs for details.")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.header("About")

if app_mode == "Train New Model":
    st.sidebar.info(
        """
        **Train New Model Mode**

        This mode allows you to select stocks from a specific sector or industry
        for your portfolio. The recommendations are based on current market data
        and analysis.

        You can specify your initial investment capital, and after stock selection,
        the app trains forecasting models and an RL agent to optimize your portfolio
        based on your investment amount.
        """
    )
else:
    st.sidebar.info(
        """
        **Evaluate Pretrained Model Mode**

        This mode allows you to evaluate a pretrained RL agent model over a specific
        time period without having to train from scratch. This is perfect for showcases
        or demonstrations where you want to show how the model performs on recent data.

        You can specify:
        - The path to a pretrained model
        - The tickers to evaluate
        - The date range for evaluation (e.g., the last two months)
        - The initial capital for the evaluation

        The app will load the model, run it on the specified data, and generate
        visualizations comparing the RL agent to a buy-and-hold strategy.
        """
    )
