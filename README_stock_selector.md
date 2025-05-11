# Stock Portfolio Selector

A minimalist front-end application for selecting stocks by sector/industry, training forecasting models, and optimizing portfolios using reinforcement learning.

## Features

- Select stocks from various sectors/industries
- View AI-generated investment rationales for each stock
- Train forecasting models using historical data
- Optimize portfolio allocation using reinforcement learning

## Prerequisites

Before running the application, make sure you have the following:

1. Python 3.8 or higher
2. All required packages installed (see Installation section)
3. A Google API key for Gemini AI (for stock recommendations)

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.api_key.json` file in the root directory with your Google API key:

```json
{
  "api_key": "YOUR_GOOGLE_API_KEY"
}
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run stock_selector_app.py
```

2. Select a sector/industry from the dropdown menu in the sidebar
3. Click "Get Stock Recommendations" to fetch AI-generated stock recommendations
4. Select the stocks you want to include in your portfolio using the checkboxes
5. Click "Create Portfolio with Selected Stocks" to start the training process
6. Wait for the forecasting model and RL agent to complete training

## How It Works

1. **Stock Recommendations**: The application uses Google's Gemini AI to generate stock recommendations based on the selected sector/industry.

2. **Forecasting Model**: The selected stocks are used to train a TiDE (Time-series Dense Encoder) model using the Darts library. This model forecasts future stock prices based on historical data.

3. **Reinforcement Learning**: An RL agent is trained to optimize portfolio allocation based on the forecasted stock prices and other market indicators.

## Files

- `stock_selector_app.py`: The main Streamlit application
- `trading_recommender.py`: Module for generating stock recommendations using Gemini AI
- `forecasting.py`: Module for training and using the forecasting model
- `RL_training.py`: Module for training the reinforcement learning agent

## Notes

- The training process may take some time depending on your hardware and the number of selected stocks.
- Make sure you have enough disk space for storing the trained models.
- The application requires an internet connection to fetch stock data and recommendations.

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that your Google API key is valid and has access to the Gemini API
3. Ensure you have sufficient memory and disk space for training models
4. Check the console output for any error messages
