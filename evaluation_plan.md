# Evaluation Plan for Multi-Agent Quant Trading System

## Overview

This document outlines the evaluation experiments for the multi-agent trading system. The goal is to rigorously compare our full system against simpler alternatives and analyze performance across different market conditions.

---

## Experiment 1: Main System Comparison

### Objective
Compare three system configurations across different market conditions to demonstrate the value of combining quantitative models with LLM-based analysis.

### Test Period
- **Duration:** 2 years
- **Start Date:** 2023-01-01
- **End Date:** 2024-12-01

### Stock Selection (9 stocks total)

| Category | Criteria | Stocks (To Be Selected) |
|----------|----------|------------------------|
| 📈 Increasing (3) | Strong uptrend over 1 year | TBD |
| 📉 Decreasing (3) | Downtrend over 1 year | TBD |
| ➡️ Stable (3) | Low volatility, sideways movement | TBD |

**Stock Selection Criteria:**
- Increasing: >30% gain over the test period
- Decreasing: >20% loss over the test period
- Stable: <10% change with low volatility

### Systems to Compare

| System | Configuration | Description |
|--------|--------------|-------------|
| **Quant-Only** | Quant Agent enabled, Market-Sense disabled | HMM regime detection and return prediction only |
| **LLM-Only** | Market-Sense enabled, Quant Agent disabled | RAG + LLM qualitative analysis only |
| **Full System** | All agents enabled | Combined Quant + Market-Sense with Coordinator |

### Metrics to Collect

| Metric | Formula | Description |
|--------|---------|-------------|
| **Cumulative Return** | `(Final Value - Initial Value) / Initial Value` | Total profit/loss percentage |
| **Sharpe Ratio** | `(Annualized Return - Risk Free Rate) / Annualized Std Dev` | Risk-adjusted return (use 5% risk-free) |
| **Max Drawdown** | `max((Peak - Trough) / Peak)` | Largest peak-to-trough decline |
| **Directional Accuracy** | `Correct Predictions / Total Predictions` | % of times predicted direction was correct |
| **Win Rate** | `Profitable Trades / Total Trades` | % of trades that made money |
| **Number of Trades** | `count(trades)` | Total trades executed |

### Expected Output Table

| Stock | Category | System | Cumulative Return | Sharpe Ratio | Max Drawdown | Directional Accuracy | Win Rate | Num Trades |
|-------|----------|--------|-------------------|--------------|--------------|---------------------|----------|------------|
| TBD | Increasing | Quant-Only | | | | | | |
| TBD | Increasing | LLM-Only | | | | | | |
| TBD | Increasing | Full System | | | | | | |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Implementation Notes
- Run each stock × system combination separately
- Save all results to JSON for analysis
- Generate comparison charts

---

## Experiment 2: Ablation Study (Optional)

### Objective
Determine which components contribute most to system performance.

### Tests

| Test ID | Configuration | Purpose |
|---------|--------------|---------|
| 2a | Full System WITHOUT Fourier seasonality | Does Fourier seasonality improve predictions? |
| 2b | Full System WITHOUT RAG knowledge base | Does historical event retrieval help? |
| 2c | Full System WITHOUT news data | Does real-time news add value? |

### Method
- Select 2-3 representative stocks
- Run Full System with each component disabled
- Compare metrics to Full System baseline

### Decision
- [ ] Include this experiment
- [ ] Skip this experiment

---

## Experiment 3: Volatility Robustness

### Status: MERGED INTO EXPERIMENT 1

By selecting stocks across three categories (increasing, decreasing, stable), Experiment 1 already covers different market conditions. No separate experiment needed.

---

## Experiment 4: Deep Dive Analysis

### Objective
Detailed qualitative and quantitative analysis of system behavior on selected stocks.

### Stock Selection (4 stocks)

| Volatility | Trend | Stock (To Be Selected) | Selection Criteria |
|------------|-------|------------------------|-------------------|
| **High Volatility** | 📈 Uptrend | TBD | High beta, strong gains |
| **High Volatility** | 📉 Downtrend | TBD | High beta, significant losses |
| **Low Volatility** | 📈 Uptrend | TBD | Low beta, steady gains |
| **Low Volatility** | 📉 Downtrend | TBD | Low beta, gradual decline |

### Analysis Components

#### 4.1 Regime Detection Visualization
- Plot price chart with regime overlay (Bull/Bear/Sideways)
- Evaluate if regime changes align with actual market movements
- Calculate regime detection accuracy

#### 4.2 Trade Timing Analysis
- Plot BUY/SELL points on price chart
- Analyze if buys occur before price increases
- Analyze if sells occur before price decreases

#### 4.3 Signal Agreement Analysis
- Count times Quant and Market-Sense agree vs disagree
- Analyze outcomes when signals conflict
- Determine which agent is more reliable in conflicts

#### 4.4 Market-Sense Reasoning Quality
- Sample 5-10 Market-Sense outputs per stock
- Evaluate reasoning quality:
  - Is the reasoning logical?
  - Are risk flags appropriate?
  - Does confidence match actual outcome?

### Expected Visualizations
1. Price chart with regime coloring
2. Price chart with trade markers (BUY ▲, SELL ▼)
3. Signal agreement/disagreement timeline
4. Sample Market-Sense output screenshots

---

## Experiment 5: Market-Sense Agent Evaluation (Independent)

### Objective
Evaluate the LLM's ability to answer economic and market questions, independent of trading performance.

### Method
1. Create a set of 15-20 economic/market questions
2. Feed each question to the Market-Sense Agent
3. Evaluate response quality on multiple criteria
4. Compare to expected/correct answers

### Question Categories

#### Category A: Macroeconomic Understanding (5 questions)
| ID | Question |
|----|----------|
| A1 | How do rising interest rates typically affect growth stocks vs value stocks? |
| A2 | What is the relationship between inflation and stock market performance? |
| A3 | How does a strong US dollar impact multinational companies? |
| A4 | What economic indicators should investors watch for recession signals? |
| A5 | How do Federal Reserve policy decisions affect different market sectors? |

#### Category B: Sector Analysis (5 questions)
| ID | Question |
|----|----------|
| B1 | What factors drive semiconductor stock prices? |
| B2 | How do oil price changes affect airline stocks? |
| B3 | What are the key metrics for evaluating tech company valuations? |
| B4 | How does consumer confidence affect retail sector stocks? |
| B5 | What makes healthcare stocks defensive investments? |

#### Category C: Historical Events (5 questions)
| ID | Question |
|----|----------|
| C1 | What caused the 2022 tech stock crash? |
| C2 | How did COVID-19 initially impact different market sectors? |
| C3 | What happened during the 2023 banking crisis (SVB collapse)? |
| C4 | How did the market react to the 2024 Fed rate decisions? |
| C5 | What drove NVIDIA's stock price increase in 2023-2024? |

#### Category D: Company-Specific Analysis (5 questions)
| ID | Question |
|----|----------|
| D1 | What are the main growth drivers for Apple? |
| D2 | What risks does Tesla face as a company? |
| D3 | Why is NVIDIA considered a leader in AI? |
| D4 | What challenges does Intel face in the semiconductor market? |
| D5 | How does Amazon's cloud business (AWS) affect its stock? |

### Evaluation Criteria

| Criterion | Score (1-5) | Description |
|-----------|-------------|-------------|
| **Factual Accuracy** | 1-5 | Are the facts stated correct? |
| **Logical Reasoning** | 1-5 | Is the reasoning coherent and logical? |
| **Relevance** | 1-5 | Is the response relevant to the question? |
| **Completeness** | 1-5 | Does it cover the key points? |
| **Trading Applicability** | 1-5 | Would this help a trading decision? |

### Expected Output

| Question ID | Factual | Logical | Relevance | Completeness | Applicability | Total | Notes |
|-------------|---------|---------|-----------|--------------|---------------|-------|-------|
| A1 | | | | | | /25 | |
| A2 | | | | | | /25 | |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## Implementation Checklist

### Code Changes Needed

| Task | File | Description | Status |
|------|------|-------------|--------|
| Add Quant-Only mode | main.py | Skip Market-Sense Agent calls | [ ] |
| Add LLM-Only mode | main.py | Skip Quant Agent, use LLM signals only | [ ] |
| Implement metrics calculation | metrics.py (new) | Calculate all 6 metrics | [ ] |
| Add visualization functions | visualize.py (new) | Generate charts and plots | [ ] |
| Create experiment runner | run_experiments.py (new) | Automate experiment execution | [ ] |

### Data Collection

| Task | Description | Status |
|------|-------------|--------|
| Select 9 stocks for Exp 1 | Research and verify trend categories | [ ] |
| Select 4 stocks for Exp 4 | Match volatility/trend criteria | [ ] |
| Finalize question list for Exp 5 | Review and refine questions | [ ] |

### Execution Order

| Step | Experiment | Estimated Time |
|------|------------|----------------|
| 1 | Implement Quant-Only and LLM-Only modes | 1-2 hours |
| 2 | Implement metrics calculation | 1-2 hours |
| 3 | Select and verify all stocks | 1 hour |
| 4 | Run Experiment 1 (27 backtests: 9 stocks × 3 systems) | 4-6 hours |
| 5 | Run Experiment 4 (4 stocks, detailed analysis) | 2-3 hours |
| 6 | Run Experiment 5 (20 questions) | 1-2 hours |
| 7 | Generate visualizations | 2-3 hours |
| 8 | Write analysis and results | 3-4 hours |

---

## Summary

| Experiment | Focus | Stocks | Output |
|------------|-------|--------|--------|
| **1** | System Comparison | 9 (3↑, 3↓, 3→) | Metrics table, comparison charts |
| **2** | Ablation Study | 2-3 (optional) | Component contribution analysis |
| **4** | Deep Dive | 4 (2 high vol, 2 low vol) | Visualizations, qualitative analysis |
| **5** | LLM Quality | N/A | Question-answer evaluation table |

---

## Notes

- All experiments use 2-year test period (2023-01-01 to 2024-12-01)
- Initial capital: $100,000 per backtest
- Trading frequency: Every 5 days
- Risk-free rate for Sharpe: 5% annualized
