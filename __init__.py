"""
Quant Trading Agent System

A multi-agent system for quantitative trading that combines:
- Quantitative modeling (HMM for regime detection)
- LLM-based market analysis (RAG + reasoning)
- Portfolio management with risk controls

Agents:
    1. DataAgent - Collects price, fundamental, and news data
    2. FeatureEngineeringAgent - Computes technical indicators and factors
    3. QuantModelingAgent - Trains HMM and generates predictions
    4. MarketSenseAgent - RAG + LLM for qualitative market insight
    5. CoordinatorAgent - Aggregates signals and manages portfolio

Usage:
    python main.py --mode predict --ticker AAPL
    python main.py --mode backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01
    python main.py --mode train --config config/model_design.md
"""

__version__ = "0.1.0"
__author__ = "Qi Chen"