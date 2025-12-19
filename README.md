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

## Project Structure

```
quant_trading_agents/
│
├── config/
│   └── config.py              # Configuration settings
│
├── llm/
│   └── llm_client.py          # LLM client wrapper
│
├── agents/
│   ├── base_agent.py          # Base agent class
│   ├── data_agent.py          # Agent 1: Data collection
│   ├── feature_agent.py       # Agent 2: Feature engineering
│   ├── quant_model_agent.py   # Agent 3: Quant modeling
│   ├── market_sense_agent.py  # Agent 4: Market sense (RAG)
│   └── coordinator_agent.py   # Agent 5: Coordinator/Portfolio
│
├── knowledge/                 # RAG knowledge base (optional)
├── portfolio/                 # Portfolio state
├── data/                      # Data folder (optional)
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## Usage

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

## Key Design Decisions

1. **Factors stay numerical** — Exact values preserved for ML models
2. **Embeddings for retrieval only** — Find relevant news, then pass text to LLM
3. **RAG + Prompt Engineering** — Not fine-tuning for Market-Sense Agent
4. **Portfolio constraints** — Long-only, position limits, cash reserves
5. **Quant model trained independently** — Offline training, loaded for inference

## Status

This is the skeleton code structure. Each file contains:
- Class definitions
- Method signatures with docstrings
- TODO comments for implementation

## Next Steps

1. Implement LLM client for Argonne API
2. Implement data collection tools
3. Implement feature engineering
4. Create model design document
5. Implement RAG knowledge base
6. Implement portfolio management logic
7. Testing and evaluation
