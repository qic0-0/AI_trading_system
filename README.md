# Quant Trading Agent System

A multi-agent system for news-aware quantitative trading using LLM-powered agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT SYSTEM                                       │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                        Data Agent                                 │     │
│    │            (Collect prices, news, fundamentals)                   │     │
│    └──────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                  Feature Engineering Agent                        │     │
│    │         (Compute factors + Process news for retrieval)           │     │
│    └───────────────────┬──────────────────────┬───────────────────────┘     │
│                        │                      │                              │
│           Numerical Features            Retrieved News Text                  │
│                        │                      │                              │
│                        ▼                      ▼                              │
│    ┌─────────────────────────┐    ┌─────────────────────────┐               │
│    │  Quant Modeling Agent   │    │   Market-Sense Agent    │               │
│    │  (Train/Tune ML Model)  │    │   (LLM + RAG Reasoning) │               │
│    └───────────┬─────────────┘    └───────────┬─────────────┘               │
│                │                              │                              │
│           Quant Signal                   Market Insight                      │
│                │                              │                              │
│                └──────────────┬───────────────┘                              │
│                               ▼                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │              Coordinator/Portfolio Agent                          │     │
│    │         (Signal aggregation + Portfolio management)              │     │
│    └──────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│                          Validated Trade Orders                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agents

| # | Agent | Role | Key Tools |
|---|-------|------|-----------|
| 1 | Data Agent | Collect raw data | fetch_price, fetch_news, fetch_fundamentals, fetch_economic |
| 2 | Feature Engineering Agent | Transform data | compute_factors, embed_text, retrieve_news |
| 3 | Quant Modeling Agent | ML prediction | read_document, execute_python, save_model, load_model |
| 4 | Market-Sense Agent | LLM reasoning (RAG) | retrieve_knowledge, retrieve_historical_events |
| 5 | Coordinator/Portfolio Agent | Portfolio management | get_portfolio_state, validate_trade, execute_trade |



## Usage

Before use the system, make sure you get API key: 

- FINNHUB_API_KEY from https://finnhub.io/register, 
- FRED_API_KEY from https://fred.stlouisfed.org/docs/api/api_key.html

then put in configuration
```bash
# Install dependencies
pip install -r requirements.txt

# Run single prediction
python main.py --mode predict --ticker AAPL

# Run backtest
python main.py --mode backtest --ticker AAPL --start 2024-01-01 --end 2024-06-01

# Train quant model
python main.py --mode train --config config/model_design.yaml
```


some code are generated with the help of chat-GPT like run_experiment