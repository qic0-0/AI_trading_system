# Quant Trading Agent System - Implementation TODO

> **How to continue in a new conversation:**
> Say: "Read the project files in this project folder, especially TODO.md and Agent_System_Design.pdf. Continue implementing from where we left off."

---

## Project Overview

- **Project:** Multi-Agent System for News-Aware Quantitative Trading
- **Course:** AI Agents for Science (CMSC 35370)
- **Target:** U.S. Stock Market

---

## Implementation Status

### Phase 1: Core Infrastructure (Priority: HIGH)

#### 1.1 LLM Client
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `llm/llm_client.py` | `__init__()` | ❌ TODO | Initialize HTTP client, set auth headers |
| `llm/llm_client.py` | `chat()` | ❌ TODO | Send chat completion request |
| `llm/llm_client.py` | `chat_with_tools()` | ❌ TODO | Implement ReAct loop |
| `llm/llm_client.py` | `_format_tools()` | ❌ TODO | Format tools for API |
| `llm/llm_client.py` | `_parse_response()` | ❌ TODO | Parse API response |

**Dependency:** Need Argonne API endpoint and authentication details

#### 1.2 Base Agent
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `agents/base_agent.py` | `think()` | ❌ TODO | Basic LLM reasoning |
| `agents/base_agent.py` | `think_with_tools()` | ❌ TODO | ReAct pattern implementation |

---

### Phase 2: Data Collection (Priority: HIGH)

#### 2.1 Data Agent
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `agents/data_agent.py` | `run()` | ❌ TODO | Orchestrate data collection |
| `agents/data_agent.py` | `fetch_price_data()` | ❌ TODO | Use yfinance |
| `agents/data_agent.py` | `fetch_fundamentals()` | ❌ TODO | Use yfinance |
| `agents/data_agent.py` | `fetch_news()` | ❌ TODO | Use Finnhub or Alpha Vantage |
| `agents/data_agent.py` | `fetch_economic_indicators()` | ❌ TODO | Use FRED API |
| `agents/data_agent.py` | `generate_data_dictionary()` | ❌ TODO | Create schema of available data |

**Dependencies:** 
- API keys for Finnhub/Alpha Vantage and FRED
- Install: `yfinance`, `fredapi`

---

### Phase 3: Feature Engineering (Priority: HIGH)

#### 3.1 Feature Agent
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `agents/feature_agent.py` | `run()` | ❌ TODO | Orchestrate feature pipeline |
| `agents/feature_agent.py` | `compute_technical_indicators()` | ❌ TODO | RSI, MACD, Bollinger Bands |
| `agents/feature_agent.py` | `compute_factors()` | ❌ TODO | Momentum, value, volatility |
| `agents/feature_agent.py` | `embed_text()` | ❌ TODO | Text to embedding |
| `agents/feature_agent.py` | `store_embedding()` | ❌ TODO | Store in vector DB |
| `agents/feature_agent.py` | `retrieve_similar_news()` | ❌ TODO | Search vector DB |
| `agents/feature_agent.py` | `generate_feature_dictionary()` | ❌ TODO | Create schema of features for Model Agent |

**Dependencies:**
- Vector database (ChromaDB)
- Embedding model (OpenAI or sentence-transformers)
- Technical analysis library (`ta`)

---

### Phase 4: Quant Modeling (Priority: HIGH)

#### 4.1 Model Design Document
| Task | Status | Notes |
|------|--------|-------|
| Create `config/model_design.md` | ✅ DONE | HMM model, 3 states, 5-day returns |
| Define observation variables | ✅ DONE | 5-day stock return |
| Define hidden states | ✅ DONE | Bull, Bear, Sideways |
| Define output format | ✅ DONE | Expected return per stock |

#### 4.2 Quant Model Agent
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `agents/quant_model_agent.py` | `run()` | ❌ TODO | Route to train/predict |
| `agents/quant_model_agent.py` | `train_model()` | ❌ TODO | LLM-driven training workflow |
| `agents/quant_model_agent.py` | `predict()` | ❌ TODO | Generate expected returns (NOT decisions) |
| `agents/quant_model_agent.py` | `read_document()` | ❌ TODO | Read design doc |
| `agents/quant_model_agent.py` | `read_feature_dictionary()` | ❌ TODO | Understand available features |
| `agents/quant_model_agent.py` | `execute_python()` | ❌ TODO | Safe code execution |
| `agents/quant_model_agent.py` | `save_model()` | ❌ TODO | Persist model |
| `agents/quant_model_agent.py` | `load_model()` | ❌ TODO | Load model |

**Dependencies:**
- Model design document
- ML libraries: `lightgbm`, `scikit-learn`

---

### Phase 5: Market-Sense RAG (Priority: MEDIUM)

#### 5.1 Knowledge Base Setup
| Task | Status | Notes |
|------|--------|-------|
| Create knowledge base structure | ❌ TODO | `knowledge/` directory |
| Populate economic principles | ❌ TODO | Fed policy, inflation, etc. |
| Populate financial concepts | ❌ TODO | Valuation, risk metrics |
| Populate historical events | ❌ TODO | 2008 crisis, COVID crash |
| Build vector index | ❌ TODO | Embed and index content |

#### 5.2 Market-Sense Agent
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `agents/market_sense_agent.py` | `run()` | ❌ TODO | Analyze market conditions |
| `agents/market_sense_agent.py` | `retrieve_knowledge()` | ❌ TODO | Search knowledge base |
| `agents/market_sense_agent.py` | `retrieve_historical_events()` | ❌ TODO | Find similar past events |
| `agents/market_sense_agent.py` | `format_market_state()` | ❌ TODO | Format for LLM prompt |

**Dependencies:**
- Knowledge base content
- Vector database (shared with Feature Agent)

---

### Phase 6: Portfolio Management (Priority: MEDIUM)

#### 6.1 Coordinator Agent
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `agents/coordinator_agent.py` | `run()` | ❌ TODO | Signal aggregation + trading |
| `agents/coordinator_agent.py` | `validate_trade()` | ❌ TODO | Check constraints |
| `agents/coordinator_agent.py` | `execute_trade()` | ❌ TODO | Update portfolio state |
| `agents/coordinator_agent.py` | `calculate_position_size()` | ❌ TODO | Position sizing logic |
| `agents/coordinator_agent.py` | `aggregate_signals()` | ❌ TODO | Combine quant + market sense |

**Dependencies:**
- Quant Model Agent (for signals)
- Market-Sense Agent (for insights)

---

### Phase 7: Main Pipeline & Backtest (Priority: LOW - after agents work)

#### 7.1 Main System
| File | Function | Status | Notes |
|------|----------|--------|-------|
| `main.py` | `run_single_prediction()` | ❌ TODO | Full pipeline for one ticker |
| `main.py` | `run_backtest()` | ❌ TODO | Historical simulation |
| `main.py` | `train_model()` | ❌ TODO | Trigger model training |

#### 7.2 Evaluation Framework
| Task | Status | Notes |
|------|--------|-------|
| Create evaluation metrics | ❌ TODO | Sharpe, returns, drawdown |
| Implement baseline (LLM-only) | ❌ TODO | For comparison |
| Generate performance reports | ❌ TODO | For paper/poster |

---

## Implementation Order (Recommended)

```
1. llm_client.py          ← Need this first for all agents
2. data_agent.py          ← Get real data flowing
3. feature_agent.py       ← Process data into features
4. quant_model_agent.py   ← ML predictions
5. market_sense_agent.py  ← RAG reasoning
6. coordinator_agent.py   ← Portfolio management
7. main.py                ← Tie it all together
8. Evaluation framework   ← For project submission
```

---

## Key Design Decisions (Reference)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Factors representation | Numerical (not embedded) | Preserve exact values for ML |
| News for retrieval | Embeddings | Semantic search |
| News for reasoning | Raw text | LLM reads text directly |
| Market-Sense approach | RAG + Prompt Engineering | Not fine-tuning |
| Portfolio | Long-only | No shorting |
| Quant model training | Independent/offline | Separate from trading loop |
| **Data Agent output** | Raw data + **Data Dictionary** | Describe available data schema |
| **Feature Agent output** | Features + **Feature Dictionary** | Describe each feature for Model Agent |
| **Quant Model output** | **Expected returns only** | No BUY/SELL decisions |
| **Coordinator role** | **Makes all BUY/SELL/HOLD decisions** | Single decision point |

---

## Files Reference

| File | Description |
|------|-------------|
| `Agent_System_Design.pdf` | Complete system design document |
| `Agent_System_Design.md` | Markdown version of design |
| `config/config.py` | Configuration settings |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Notes for Next Session

- Current status: **Code skeleton complete, no implementations yet**
- Next logical step: **Implement `llm_client.py`** (requires Argonne API details)
- Alternative next step: **Implement `data_agent.py`** (can test with yfinance without LLM)

---

*Last updated: Session 1 - Project structure and design complete*
