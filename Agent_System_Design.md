# Multi-Agent System Design for News-Aware Quantitative Trading

## Project Design Document (Updated - Final Implementation)

---

## 1. Overview

This document summarizes the design of a multi-agent system for quantitative trading that integrates traditional numerical factors with news/text signals.

### Key Design Principles

- **Factors stay numerical** — exact values preserved for quantitative models
- **Embeddings for retrieval only** — find relevant news, then pass text to LLM
- **LLM reads text directly** — embeddings are for search, not for reasoning
- **Agent writes code once, reuses with new data** — no repeated LLM calls for model training
- **No look-ahead bias** — expanding window training during backtest

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT SYSTEM                                       │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                        Data Agent                                 │     │
│    │            (Collect prices, news, fundamentals)                   │     │
│    └──────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                          │
│                      Raw Data + Data Dictionary                              │
│                                   │                                          │
│                                   ▼                                          │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                  Feature Engineering Agent                        │     │
│    │         (Compute factors + Process news for retrieval)           │     │
│    └───────────────────┬──────────────────────┬───────────────────────┘     │
│                        │                      │                              │
│          Features + Feature Dict        Retrieved News Text                  │
│                        │                      │                              │
│                        ▼                      ▼                              │
│    ┌─────────────────────────┐    ┌─────────────────────────┐               │
│    │  Quant Modeling Agent   │    │   Market-Sense Agent    │               │
│    │  (HMM: 3 States)        │    │   (RAG + LLM)           │               │
│    │  Predict Expected       │    │   Qualitative Analysis  │               │
│    │  Returns ONLY           │    │                         │               │
│    └───────────┬─────────────┘    └───────────┬─────────────┘               │
│                │                              │                              │
│       Expected Returns               Market Insight                          │
│       + Regime + Confidence          + Risk Flags                            │
│                │                              │                              │
│                └──────────────┬───────────────┘                              │
│                               ▼                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                 Coordinator/Portfolio Agent                       │     │
│    │         Signal Aggregation + Risk Management                     │     │
│    │         DECIDES: BUY / SELL / HOLD based on aggregated signals   │     │
│    └──────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│                      BUY / SELL / HOLD Decisions                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTEST FRAMEWORK (Integrated)                           │
│                                                                              │
│    • Expanding window training (retrain every 5 days)                      │
│    • No look-ahead bias                                                     │
│    • Metrics: Sharpe ratio, returns, max drawdown, directional accuracy    │
│    • Comparison: Full System vs Buy-Hold / Quant-Only / LLM-Only           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Sources

The system targets the **U.S. stock market** using the following reliable free data sources:

### Price Data & Fundamentals

| Source | Library | Data Available | Notes |
|--------|---------|----------------|-------|
| Yahoo Finance | `yfinance` | OHLCV, adjusted prices, P/E, P/B, market cap | Free, no API key needed |

### News Data

| Source | Access | Data Available | Notes |
|--------|--------|----------------|-------|
| Finnhub | API | Company news, sentiment | Free tier: 60 calls/min |

### Economic Indicators

| Source | Library | Data Available | Notes |
|--------|---------|----------------|-------|
| FRED | `fredapi` | Interest rates, inflation, GDP | Federal Reserve data |

---

## 4. Agent Details

### Agent 1: Data Agent

**Purpose:** Collect and organize raw data from various sources. Outputs a **Data Dictionary** describing all available data.

| Attribute | Description |
|-----------|-------------|
| **Inputs** | Ticker symbols, date ranges |
| **Outputs** | Raw price data, raw news text, fundamental data, economic indicators, **Data Dictionary** |

**Tools:**
- `fetch_price_data(ticker, start_date, end_date)` — Get OHLCV data (via yfinance)
- `fetch_fundamentals(ticker)` — Get P/E, market cap, etc. (via yfinance)
- `fetch_news(ticker, start_date, end_date)` — Get news headlines (via Finnhub)
- `fetch_economic_indicators(indicator, start, end)` — Get interest rates, CPI, etc. (via FRED)
- `generate_data_dictionary()` — Create dictionary describing all available data

**LLM Role:** Minimal. Mainly orchestrates tool calls.

---

### Agent 2: Feature Engineering Agent

**Purpose:** Transform raw data into usable features. Outputs a **Feature Dictionary** describing all computed features for the Quant Model Agent.

| Attribute | Description |
|-----------|-------------|
| **Inputs** | Raw price data, raw news text, fundamental data, **Data Dictionary** |
| **Outputs** | Numerical feature matrix, retrieved relevant news text, **Feature Dictionary** |

**Dual Output Design:**

| Output Type | Description | Used By |
|-------------|-------------|---------|
| Numerical Features | Factors (momentum, value, volatility, etc.) | Quant Modeling Agent |
| Feature Dictionary | Schema describing each feature | Quant Modeling Agent |
| Retrieved News Text | Relevant historical news based on current context | Market-Sense Agent |

**Tools:**
- `compute_technical_indicators(prices)` — RSI, MACD, moving averages
- `compute_factors(prices, fundamentals)` — Momentum, value, quality factors
- `embed_text(news_text)` — Convert news to embedding vector
- `store_embedding(vector, metadata)` — Store in vector database
- `retrieve_similar_news(query, top_k)` — Find relevant past news
- `generate_feature_dictionary()` — Create dictionary describing all features

**Key Design Decision:** 
- Factors remain as **exact numerical values** (not embedded)
- Embeddings are used **only for retrieval**, not passed to downstream agents
- LLM receives **raw text**, not embeddings

---

### Agent 3: Quant Modeling Agent (UPDATED)

**Purpose:** Write training code once based on model design document, then reuse that code with new data. Predicts expected returns using Hidden Markov Models.

| Attribute | Description |
|-----------|-------------|
| **Inputs** | Price data, **Feature Dictionary**, **Model Design Document** |
| **Outputs** | **Expected returns** (per stock), confidence, regime, regime probabilities |

**Key Implementation Details:**

| Aspect | Value |
|--------|-------|
| Model Type | Gaussian Hidden Markov Model (HMM) |
| Number of States | 3 (Bull, Bear, Sideways) |
| Observation | 5-day stock returns |
| Training Data | Expanding window (all data up to current date) |
| Prediction Horizon | 5-day expected return |
| Training Frequency | **Every 5 days** (during backtest) |

**Agentic Code Generation & Reuse:**

```
FIRST TIME:
1. Read model_design.md using read_document tool
2. LLM writes Python training code based on design
3. Save code to models/train_hmm.py
4. Execute code with current data
5. Save trained model

SUBSEQUENT TIMES (Backtest - 50+ iterations):
1. Check: models/train_hmm.py exists? → YES
2. Load saved training code
3. Execute with NEW data (NO LLM call!)
4. Save updated model
5. Predict with fresh model

Result: Fast training after first time!
```

**No Look-Ahead Bias:**

During backtest, the agent retrains every 5 days using only data available up to that point:

```
Trading Day 5:  Train on data [start → Day 5]  → Predict Day 6-10
Trading Day 10: Train on data [start → Day 10] → Predict Day 11-15
Trading Day 15: Train on data [start → Day 15] → Predict Day 16-20
...
```

This ensures predictions use only past information (expanding window).

**Output Format:**

```python
{
    "AAPL": {
        "expected_return": 0.023,        # +2.3% expected 5-day return
        "confidence": 0.68,              # Confidence = max(regime_probabilities)
        "regime": "Bull",                # Current detected regime
        "regime_probabilities": {
            "Bull": 0.68,
            "Bear": 0.12,
            "Sideways": 0.20
        },
        "horizon": "5-day"
    }
}
```

**Tools:**
- `read_document(path)` — Read model design specifications
- `execute_python(code, variables)` — Run training code with data
- `load_model(path)` — Load previously trained model
- (Code is saved to file, reused automatically)

**LLM Role:**
- First time: Write training code based on design document
- Subsequent times: No LLM calls (code is reused)

**Why This Design:**
1. **True agentic approach** — Agent writes code autonomously
2. **Fast after first time** — No repeated LLM calls during backtest
3. **Flexible** — Change design doc → new code generated
4. **Realistic backtest** — No look-ahead bias with expanding window

---

### Agent 4: Market-Sense Agent

**Purpose:** Apply LLM reasoning with financial domain knowledge (via RAG) to interpret news and market context.

| Attribute | Description |
|-----------|-------------|
| **Inputs** | Market state, relevant news text, quant signal, ticker |
| **Outputs** | Market Insight (outlook, confidence, reasoning, risk flags) |

**RAG Implementation:**

| Knowledge Store | Contents | Size |
|----------------|----------|------|
| Financial Knowledge | Economic principles, financial concepts, investment strategies | 10 items |
| Historical Events | 2008 crisis, COVID crash, dot-com bubble, etc. | 6 major events |
| Embedding Model | sentence-transformers (all-MiniLM-L6-v2) | Lightweight |

**Retrieval Process:**

```
1. Get current market state + news
2. Embed query with sentence-transformers
3. Retrieve top-3 knowledge items (cosine similarity)
4. Retrieve top-2 historical events
5. Build LLM prompt with:
   - Market state (S&P 500, VIX, rates, ticker metrics)
   - News headlines
   - Quant signal (expected return, regime)
   - Retrieved knowledge
   - Retrieved historical comparisons
6. LLM reasons and outputs structured insight
```

**Output Format:**

```python
{
    "outlook": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence": 0.75,  # 0.0 to 1.0
    "reasoning": "Fed's dovish stance and strong earnings...",
    "risk_flags": ["Fed meeting next week", "High valuations"],
    "historical_comparison": "Similar to late 2019 environment..."
}
```

**Tools:**
- `retrieve_knowledge(query, top_k)` — Search financial knowledge base
- `retrieve_historical_events(query, top_k)` — Find similar past market situations
- `format_market_state(data)` — Format market data for LLM

**LLM Role (with System Prompt):**
- Interpret news in market context
- Apply financial reasoning using retrieved knowledge
- Compare current situation to historical patterns
- Assess risks and opportunities

**Why RAG instead of Fine-tuning:**

| Aspect | Fine-tuning | RAG (Chosen) |
|--------|-------------|--------------|
| Project feasibility | Difficult | Feasible |
| Knowledge updates | Requires retraining | Just update knowledge base |
| Transparency | Black-box | Can see what was retrieved |
| Compute cost | High | Low |

---

### Agent 5: Coordinator/Portfolio Agent (UPDATED)

**Purpose:** Aggregate signals from Quant Model and Market-Sense agents, resolve conflicts, and make all **BUY/SELL/HOLD decisions**. Manages portfolio state and validates trades within constraints.

| Attribute | Description |
|-----------|-------------|
| **Inputs** | Quant signal (expected return, regime, confidence), Market-Sense insight (outlook, confidence, risk flags), current portfolio state |
| **Outputs** | BUY/SELL/HOLD decision, validated trade order, updated portfolio state, reasoning |

**Trading Frequency:**

| Aspect | Value |
|--------|-------|
| Decision frequency | Every 5 trading days |
| Prediction horizon | 5-day expected return (from Quant Model) |
| Execution | Immediate on decision day |

**Signal Aggregation Logic:**

The Coordinator combines signals using this logic:

```
1. AGREEMENT (Quant & Market-Sense align):
   - Both say BUY → BUY with average confidence
   - Both say SELL → SELL with average confidence
   - Both say HOLD → HOLD

2. PARTIAL AGREEMENT (one is NEUTRAL):
   - Quant says BUY, Market says NEUTRAL → BUY with 70% confidence
   - Market says SELL, Quant says NEUTRAL → SELL with 70% confidence

3. DISAGREEMENT (conflict):
   - Take signal with HIGHER confidence, reduce to 50%
   - If confidence is close → HOLD (stay safe)

4. RISK ADJUSTMENT:
   - If Market-Sense flags risks → reduce confidence by 20%

5. POSITION SIZING:
   - signal_strength = |expected_return| × confidence × 10
   - Clamped to [0, 1]
   - Used to scale position size
```

**Example Decision Flows:**

```
Case 1: Agreement
Quant: BUY (+2.3% expected, 0.70 confidence)
Market: BULLISH (0.75 confidence)
→ Decision: BUY with 0.725 average confidence

Case 2: Disagreement
Quant: BUY (+1.8% expected, 0.60 confidence)
Market: BEARISH (0.75 confidence, "Fed hawkish")
→ Decision: SELL (Market has higher confidence at 0.75)
→ Confidence reduced to 0.50 due to conflict

Case 3: Risk Flags
Quant: BUY (+2.5% expected, 0.80 confidence)
Market: NEUTRAL (0.60 confidence, risk_flags=["Earnings volatility"])
→ Decision: BUY but confidence reduced: 0.80 → 0.64 (20% penalty)
```

**Portfolio State:**

```json
{
  "cash": 100000.00,
  "positions": {
    "AAPL": {"shares": 50, "cost_basis": 175.00, "current_price": 178.50},
    "MSFT": {"shares": 30, "cost_basis": 380.00, "current_price": 385.00}
  },
  "total_value": 128525.00
}
```

**Constraints:**

| Constraint | Value | Reason |
|-----------|-------|--------|
| Long-only | No shorting allowed | Can only sell owned shares |
| Max position | 30% of portfolio per ticker | Risk management |
| Min cash reserve | 10% always in cash | Maintain liquidity |
| Position sizing | Based on signal_strength² | Conservative scaling |

**Tools:**
- `get_portfolio_state()` — Get current positions and cash
- `validate_trade(action, ticker, amount)` — Check if trade is valid
- `execute_trade(action, ticker, shares, price)` — Execute and update state
- `calculate_position_size(ticker, expected_return, confidence)` — Determine shares to trade
- `aggregate_signals(quant, market)` — Combine and resolve conflicts

**Responsibilities:**

| Function | Description |
|----------|-------------|
| **Signal Aggregation** | Combine quant + market-sense signals, resolve conflicts |
| **Decision Making** | Convert aggregated signals → BUY/SELL/HOLD decisions |
| Trade Validation | Ensure we can execute (own shares to sell, have cash to buy) |
| Position Sizing | Determine how much to trade based on signal strength |
| Risk Management | Enforce position limits, cash reserves, risk flag penalties |

---

## 5. Data Flow Summary

| Stage | Data Type | Representation | From → To |
|-------|-----------|----------------|-----------|
| Raw prices | Numerical | Exact values | Data Agent → Feature Agent |
| Raw news | Text | String | Data Agent → Feature Agent |
| **Data Dictionary** | Schema | Dict | Data Agent → Feature Agent |
| Factors/Alpha | Numerical | **Exact values (NOT embedded)** | Feature Agent → Quant Model |
| **Feature Dictionary** | Schema | Dict | Feature Agent → Quant Model |
| News for search | Embedding | Vector (for retrieval only) | Feature Agent internal |
| News for reasoning | Text | Raw text (LLM reads directly) | Feature Agent → Market-Sense |
| **Expected returns** | Numerical | Per-stock predictions + regime | Quant Model → Coordinator |
| Market insight | Structured | outlook + confidence + reasoning | Market-Sense → Coordinator |
| **Aggregated signal** | Decision | BUY/SELL/HOLD + strength | Coordinator internal |
| **Trade order** | Action | BUY/SELL/HOLD + quantity + price | Coordinator → Output |

**Key Flow:**
```
Quant Model outputs: Expected returns + regime + confidence
Market-Sense outputs: Outlook + confidence + reasoning + risk flags
                              ↓
Coordinator aggregates: Resolves conflicts, calculates signal strength
                              ↓
Coordinator decides: BUY / SELL / HOLD + position size
```

---

## 6. Backtest Implementation (UPDATED)

**Expanding Window Training:**

To avoid look-ahead bias, the system retrains the HMM model every 5 days using only data available up to that point:

```python
for trading_day in [Day 5, Day 10, Day 15, ...]:
    # Get data ONLY up to current day
    historical_data = all_data[start_date : trading_day]
    
    # Retrain model with expanding window
    quant_agent.run(mode="train", price_data=historical_data)
    
    # Predict next 5 days
    prediction = quant_agent.run(mode="predict", price_data=historical_data)
    
    # Get market insight (simplified for backtest speed)
    market_insight = {"outlook": "NEUTRAL", "confidence": 0.5}
    
    # Make trading decision
    coordinator.run(ticker, price, prediction, market_insight)
```

**Comparison Baselines:**

| System | Description |
|--------|-------------|
| Buy-and-Hold | Equal-weight portfolio, no trading |
| Quant-Only | Uses only HMM signal, no Market-Sense |
| LLM-Only | Uses only Market-Sense, no quant model |
| Full System | Combines both with signal aggregation |

**Metrics:**

| Metric | What it Measures |
|--------|------------------|
| Cumulative Return | Total profit/loss |
| Annualized Sharpe Ratio | Risk-adjusted return |
| Maximum Drawdown | Worst peak-to-trough loss |
| Directional Accuracy | % of correct up/down predictions |
| Number of Trades | Trading activity |

---

## 7. Summary Table

| # | Agent | Primary Role | Key Innovation | Output |
|---|-------|--------------|----------------|--------|
| 1 | Data Agent | Collect raw data | Multi-source integration | Raw data + Data Dictionary |
| 2 | Feature Engineering Agent | Transform data | Dual output (numbers + text) | Features + Feature Dictionary + Retrieved News |
| 3 | Quant Modeling Agent | Predict returns | **Agent writes code once, reuses with new data** | Expected returns + regime + confidence |
| 4 | Market-Sense Agent | LLM reasoning | RAG with financial knowledge | Market Insight + risk flags |
| 5 | Coordinator Agent | Make trading decisions | **Signal aggregation with conflict resolution** | BUY/SELL/HOLD + position size |

---

## 8. Key Implementation Decisions

### Design Decisions:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Quant Model Training** | Expanding window, retrain every 5 days | Avoid look-ahead bias, adapt to new data |
| **Code Reuse** | Agent writes code once, saves to file | Fast after first time, no repeated LLM calls |
| **Signal Aggregation** | Coordinator resolves conflicts | Combines strengths of quant + qualitative |
| **Position Sizing** | Based on signal_strength² | Conservative, scales with confidence |
| **Risk Management** | 30% max position, 10% min cash | Portfolio stability |
| **Trading Frequency** | Every 5 days | Matches prediction horizon |

### Technology Stack:

| Component | Technology |
|-----------|------------|
| LLM Backend | Argonne API (Llama-3.1-70B-Instruct) |
| HMM Implementation | hmmlearn (Gaussian HMM) |
| Text Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Data Sources | yfinance, Finnhub, FRED |
| Language | Python 3.10+ |

---

*Document updated based on final implementation - December 2024*
