"""
Main entry point for the Quant Trading Agent System.

Usage:
    # Run backtest
    python main.py --mode backtest --ticker AAPL --start 2024-01-01 --end 2024-06-01
    
    # Single prediction
    python main.py --mode predict --ticker AAPL
    
    # Train quant model
    python main.py --mode train --config config/model_design.yaml

Output:
    - Console logs showing agent execution
    - Trading decisions with reasoning
    - Backtest results (if backtest mode)
    - Output files in outputs/ directory
"""

import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

from config.config import config, SystemConfig
from llm.llm_client import LLMClient
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureEngineeringAgent
from agents.quant_model_agent import QuantModelingAgent
from agents.market_sense_agent import MarketSenseAgent
from agents.coordinator_agent import CoordinatorAgent


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Main trading system orchestrating all agents.
    
    Agents:
        1. DataAgent - Collect raw data
        2. FeatureEngineeringAgent - Transform data
        3. QuantModelingAgent - ML predictions
        4. MarketSenseAgent - Qualitative analysis
        5. CoordinatorAgent - Portfolio management
    """

    def __init__(self, config: SystemConfig, system_mode: str = "full"):
        """
        Initialize the trading system.

        Args:
            config: System configuration
            system_mode: "full", "quant_only", or "llm_only"
        """
        self.config = config
        self.system_mode = system_mode  # NEW
        self.llm_client = LLMClient(config.llm)
        
        # Initialize agents
        self.data_agent = DataAgent(self.llm_client, config)
        self.feature_agent = FeatureEngineeringAgent(self.llm_client, config)
        self.quant_model_agent = QuantModelingAgent(self.llm_client, config)
        self.market_sense_agent = MarketSenseAgent(self.llm_client, config)
        self.coordinator_agent = CoordinatorAgent(self.llm_client, config)
        
        logger.info("Trading system initialized")

    def run_single_prediction(self, ticker: str) -> Dict[str, Any]:
        """
        Run a single prediction for one ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with trading decision and reasoning
        """
        logger.info(f"Running prediction for {ticker}")

        # Calculate date range (last 2 years for training, recent for prediction)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y-%m-%d")

        # Step 1: Collect data
        logger.info("[1/5] Data Agent: Collecting data...")
        data_result = self.data_agent.run({
            "tickers": [ticker],
            "start_date": start_date,
            "end_date": end_date,
            "collect_news": True,
            "collect_fundamentals": True,
            "economic_indicators": ["FEDFUNDS", "CPIAUCSL"]
        })

        if not data_result.success:
            logger.error(f"Data collection failed: {data_result.message}")
            return {"success": False, "message": data_result.message}

        collected_data = data_result.data.get("collected_data", {})
        data_dictionary = data_result.data.get("data_dictionary", {})

        # Step 2: Compute features
        logger.info("[2/5] Feature Engineering Agent: Computing features...")
        feature_result = self.feature_agent.run({
            "price_data": collected_data.get("price_data", {}),
            "fundamentals": collected_data.get("fundamentals", {}),
            "news": collected_data.get("news", {}),
            "data_dictionary": data_dictionary
        })

        if not feature_result.success:
            logger.error(f"Feature engineering failed: {feature_result.message}")
            return {"success": False, "message": feature_result.message}

        features = feature_result.data.get("features", {})
        retrieved_news = feature_result.data.get("retrieved_news", {})
        feature_dictionary = feature_result.data.get("feature_dictionary", {})

        # Step 3: Generate quant signal
        logger.info("[3/5] Quant Modeling Agent: Generating signal...")
        quant_result = self.quant_model_agent.run({
            "mode": "predict",
            "price_data": collected_data.get("price_data", {})
        })


        predictions = quant_result.data.get("predictions", {})
        quant_signal = predictions.get(ticker, {})

        # Step 4: Get market insight
        logger.info("[4/5] Market-Sense Agent: Analyzing market...")

        # Build market state for Market-Sense Agent
        ticker_features = features.get(ticker, {})
        ticker_fundamentals = collected_data.get("fundamentals", {}).get(ticker, {})
        ticker_news = collected_data.get("news", {}).get(ticker, [])

        # Get current price
        price_df = collected_data.get("price_data", {}).get(ticker)
        current_price = price_df['Close'].iloc[-1] if price_df is not None and not price_df.empty else 0

        market_state = {
            "ticker_data": {
                "ticker": ticker,
                "price": current_price,
                "pe_ratio": ticker_fundamentals.get("pe_ratio"),
                "momentum_1m": ticker_features.get("momentum_1m"),
                "volatility_30d": ticker_features.get("volatility_30d")
            }
        }

        # Extract news headlines
        news_texts = [n.get("headline", "") for n in ticker_news[:5]]

        market_result = self.market_sense_agent.run({
            "market_state": market_state,
            "news": news_texts,
            "quant_signal": quant_signal,
            "ticker": ticker
        })

        market_insight = market_result.data.get("insight", {})

        # Step 5: Make decision
        logger.info("[5/5] Coordinator Agent: Making decision...")
        decision_result = self.coordinator_agent.run({
            "ticker": ticker,
            "current_price": current_price,
            "quant_signal": quant_signal,
            "market_insight": market_insight
        })

        logger.info(f"Decision: {decision_result.message}")

        return {
            "success": True,
            "ticker": ticker,
            "current_price": current_price,
            "quant_signal": quant_signal,
            "market_insight": market_insight,
            "decision": decision_result.data,
            "logs": decision_result.logs
        }

    def run_backtest(
            self,
            tickers: List[str],
            start_date: str,
            end_date: str,
            system_mode: str = None  # NEW: override instance mode
    ) -> Dict[str, Any]:
        """
        Run backtest over historical period.

        Args:
            tickers: List of stock tickers
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            system_mode: "full", "quant_only", or "llm_only" (overrides instance mode)
        """
        # Use provided mode or fall back to instance mode
        mode = system_mode or self.system_mode
        logger.info(f"Running backtest from {start_date} to {end_date} [MODE: {mode}]")
        logger.info(f"Tickers: {tickers}")

        # Step 1: Collect all historical data
        logger.info("Collecting historical data...")
        data_result = self.data_agent.run({
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "collect_news": False,  # Skip news for backtest speed
            "collect_fundamentals": True,
            "economic_indicators": []
        })

        if not data_result.success:
            return {"success": False, "message": data_result.message}

        collected_data = data_result.data.get("collected_data", {})


        # Step 2: Training now happens inside the loop (expanding window)
        # No initial training needed


        # Step 3: Simulate trading every 5 days
        decisions = []
        portfolio_values = []

        for ticker in tickers:
            price_df = collected_data.get("price_data", {}).get(ticker)
            if price_df is None or price_df.empty:
                continue

            trading_days = price_df.index[::5]
            logger.info(f"Backtesting {ticker}: {len(trading_days)} trading periods")

            for i, date in enumerate(trading_days):
                if i < 10:  # Skip first 50 periods for warmup
                    continue

                # Get data ONLY up to this date
                historical_prices = price_df.loc[:date]
                current_price = historical_prices['Close'].iloc[-1]

                # Get fundamentals (needed by both QUANT and MARKET-SENSE)
                ticker_fundamentals = collected_data.get("fundamentals", {}).get(ticker, {})

                # ========== QUANT AGENT (skip if llm_only) ==========
                if mode != "llm_only":
                    # Compute features including embedding factors
                    feature_result = self.feature_agent.run({
                        "price_data": {ticker: historical_prices},
                        "fundamentals": {ticker: ticker_fundamentals},
                        "news": collected_data.get("news", {})
                    })
                    ticker_features = feature_result.data.get("features", {}).get(ticker, {})

                    # Pass features to Quant Model
                    train_result = self.quant_model_agent.run({
                        "mode": "train",
                        "price_data": {ticker: historical_prices},
                        "features": {ticker: ticker_features}  # NEW: includes embedding factors
                    })

                    # Predict using freshly trained model
                    quant_result = self.quant_model_agent.run({
                        "mode": "predict",
                        "price_data": {ticker: historical_prices},
                        "features": {ticker: ticker_features}  # Also pass for predict
                    })
                    predictions = quant_result.data.get("predictions", {})
                    quant_signal = predictions.get(ticker, {})
                else:
                    # Default neutral quant signal for LLM-only mode
                    quant_signal = {
                        "expected_return": 0.0,
                        "confidence": 0.0,
                        "regime": "DISABLED",
                        "regime_probabilities": {},
                        "model_type": "none"
                    }
                # ====================================================

                # Compute features for current window
                features_df = self.feature_agent.compute_technical_indicators(historical_prices)

                # ========== MARKET-SENSE AGENT (skip if quant_only) ==========
                if mode != "quant_only":
                    # Build market state
                    market_state = {
                        "ticker_data": {
                            "ticker": ticker,
                            "price": current_price,
                            "pe_ratio": ticker_fundamentals.get("pe_ratio"),
                            "momentum_1m": features_df['Close'].pct_change(20).iloc[-1] if len(features_df) > 20 else 0,
                            "volatility_30d": features_df['Close'].pct_change().std() * (252 ** 0.5) if len(
                                features_df) > 1 else 0
                        }
                    }

                    # Call Market-Sense Agent
                    market_result = self.market_sense_agent.run({
                        "market_state": market_state,
                        "news": [],
                        "quant_signal": quant_signal,
                        "ticker": ticker
                    })
                    market_insight = market_result.data.get("insight", {})
                else:
                    # Default neutral market insight for Quant-only mode
                    market_insight = {
                        "outlook": "NEUTRAL",
                        "confidence": 0.0,
                        "reasoning": "Market-Sense disabled in quant_only mode",
                        "risk_flags": []
                    }
                # =============================================================

                # Make decision
                decision_result = self.coordinator_agent.run({
                    "ticker": ticker,
                    "current_price": current_price,
                    "quant_signal": quant_signal,
                    "market_insight": market_insight
                })

                # Get aggregated signal for reasoning
                aggregated_signal = decision_result.data.get("aggregated_signal", {})

                decisions.append({
                    "date": str(date),
                    "ticker": ticker,
                    "price": current_price,
                    "decision": decision_result.data.get("trade_result", {}).get("message", ""),
                    "regime": quant_signal.get("regime", "UNKNOWN"),
                    "reasoning": aggregated_signal.get("reasoning", ""),
                    "market_outlook": market_insight.get("outlook", "UNKNOWN"),
                    "market_confidence": market_insight.get("confidence", 0),
                    "signal_direction": aggregated_signal.get("direction", "UNKNOWN"),
                    "signal_strength": aggregated_signal.get("strength", 0)
                })

                # DEBUG: Log for llm_only mode
                if mode == "llm_only":
                    logger.info(f"  LLM-ONLY Debug: outlook={market_insight.get('outlook')}, "
                                f"confidence={market_insight.get('confidence')}, "
                                f"direction={aggregated_signal.get('direction')}, "
                                f"strength={aggregated_signal.get('strength', 0):.3f}")

                portfolio_values.append({
                    "date": str(date),
                    "value": self.coordinator_agent.portfolio.total_value
                })

        # Update all position prices to final market prices
        for ticker in tickers:
            price_df = collected_data.get("price_data", {}).get(ticker)
            if price_df is not None and not price_df.empty:
                final_price = float(price_df['Close'].iloc[-1])
                if ticker in self.coordinator_agent.portfolio.positions:
                    self.coordinator_agent.portfolio.positions[ticker].current_price = final_price

        # Step 4: Calculate metrics
        initial_value = self.config.portfolio.initial_cash
        final_value = self.coordinator_agent.portfolio.total_value
        total_return = (final_value - initial_value) / initial_value

        metrics = {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": f"{total_return * 100:.2f}%",
            "num_trades": len(self.coordinator_agent.trade_history),
            "num_decisions": len(decisions),
            "system_mode": mode  # NEW: track which mode was used
        }

        logger.info(f"Backtest complete. Return: {metrics['total_return_pct']}")

        return {
            "success": True,
            "decisions": decisions,
            "portfolio_values": portfolio_values,
            "metrics": metrics,
            "final_portfolio": self.coordinator_agent.portfolio.to_dict()
        }

    def train_model(self, design_doc_path: str) -> Dict[str, Any]:
        """
        Train the quant model independently.

        Args:
            design_doc_path: Path to model design document

        Returns:
            Dict with training results
        """
        logger.info(f"Training model from {design_doc_path}")

        # Read model design document
        model_design = self.quant_model_agent.read_document(design_doc_path)
        if not model_design:
            return {"success": False, "message": f"Could not read design doc: {design_doc_path}"}

        logger.info("Model design loaded")

        # Get tickers from config
        tickers = self.config.tickers

        # Calculate training date range (2 years of data)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y-%m-%d")

        # Collect training data
        logger.info(f"Collecting training data for {tickers}...")
        data_result = self.data_agent.run({
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "collect_news": False,
            "collect_fundamentals": False,
            "economic_indicators": []
        })

        if not data_result.success:
            return {"success": False, "message": f"Data collection failed: {data_result.message}"}

        collected_data = data_result.data.get("collected_data", {})

        # Train model
        logger.info("Training HMM model...")
        train_result = self.quant_model_agent.run({
            "mode": "train",
            "price_data": collected_data.get("price_data", {})
        })

        if not train_result.success:
            return {"success": False, "message": f"Training failed: {train_result.message}"}

        training_results = train_result.data.get("training_results", {})

        logger.info("Training complete!")
        for ticker, result in training_results.items():
            logger.info(f"  {ticker}: log_likelihood={result.get('log_likelihood', 'N/A'):.2f}")

        return {
            "success": True,
            "design_doc": design_doc_path,
            "tickers": tickers,
            "training_results": training_results,
            "date_range": {"start": start_date, "end": end_date}
        }



def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Quant Trading Agent System"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "backtest", "train"],
        default="predict",
        help="Running mode"
    )
    parser.add_argument(
        "--system-mode",
        choices=["full", "quant_only", "llm_only"],
        default="full",
        help="System mode: full (both agents), quant_only, or llm_only"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker (for predict mode)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Stock tickers (for backtest mode)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model design config (for train mode)"
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("             Quant Trading Agent System")
    print("=" * 70)
    print()

    # Initialize system
    system = TradingSystem(config, system_mode=args.system_mode)

    # Run based on mode
    if args.mode == "predict":
        result = system.run_single_prediction(args.ticker)
        print_prediction_result(result)

    elif args.mode == "backtest":
        if not args.start or not args.end:
            parser.error("Backtest mode requires --start and --end dates")
        tickers = args.tickers or [args.ticker]
        result = system.run_backtest(tickers, args.start, args.end)
        print_backtest_result(result)

    elif args.mode == "train":
        if not args.config:
            parser.error("Train mode requires --config path")
        result = system.train_model(args.config)
        print_training_result(result)


def print_prediction_result(result: Dict[str, Any]):
    """Print prediction result to console."""
    print()
    print("=" * 60)
    print("                   TRADING DECISION")
    print("=" * 60)

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return

    ticker = result.get("ticker", "N/A")
    price = result.get("current_price", 0)

    print(f"Ticker: {ticker}")
    print(f"Current Price: ${price:.2f}")
    print()

    # Quant Signal
    quant = result.get("quant_signal", {})
    print("--- Quant Model Signal ---")
    print(f"  Expected Return: {quant.get('expected_return', 0) * 100:.2f}%")
    print(f"  Regime: {quant.get('regime', 'N/A')}")
    print(f"  Confidence: {quant.get('confidence', 0):.2f}")
    print()

    # Market Insight
    insight = result.get("market_insight", {})
    print("--- Market-Sense Insight ---")
    print(f"  Outlook: {insight.get('outlook', 'N/A')}")
    print(f"  Confidence: {insight.get('confidence', 0):.2f}")
    print(f"  Reasoning: {insight.get('reasoning', 'N/A')[:100]}...")
    if insight.get("risk_flags"):
        print(f"  Risk Flags: {', '.join(insight['risk_flags'])}")
    print()

    # Decision
    decision = result.get("decision", {})
    trade_result = decision.get("trade_result", {})
    portfolio = decision.get("portfolio_state", {})

    print("--- FINAL DECISION ---")
    print(f"  Action: {trade_result.get('message', 'N/A')}")
    if trade_result.get("order"):
        order = trade_result["order"]
        print(f"  Order: {order['action']} {order['shares']} shares @ ${order['price']:.2f}")
    print()

    print("--- Portfolio State ---")
    print(f"  Cash: ${portfolio.get('cash', 0):,.2f}")
    print(f"  Total Value: ${portfolio.get('total_value', 0):,.2f}")
    if portfolio.get("positions"):
        print("  Positions:")
        for ticker, pos in portfolio["positions"].items():
            print(f"    {ticker}: {pos['shares']} shares @ ${pos['current_price']:.2f} = ${pos['market_value']:,.2f}")

    print("=" * 60)


def print_backtest_result(result: Dict[str, Any]):
    """Print backtest result to console."""
    print()
    print("=" * 70)
    print("                       BACKTEST RESULTS")
    print("=" * 70)

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return

    metrics = result.get("metrics", {})

    print()
    print("--- Performance Metrics ---")
    print(f"  Initial Portfolio Value: ${metrics.get('initial_value', 0):,.2f}")
    print(f"  Final Portfolio Value:   ${metrics.get('final_value', 0):,.2f}")
    print(f"  Total Return:            {metrics.get('total_return_pct', 'N/A')}")
    print(f"  Number of Trades:        {metrics.get('num_trades', 0)}")
    print(f"  Number of Decisions:     {metrics.get('num_decisions', 0)}")
    print()

    # Final portfolio
    portfolio = result.get("final_portfolio", {})
    print("--- Final Portfolio ---")
    print(f"  Cash: ${portfolio.get('cash', 0):,.2f}")
    if portfolio.get("positions"):
        print("  Positions:")
        for ticker, pos in portfolio["positions"].items():
            print(f"    {ticker}: {pos['shares']} shares @ ${pos['current_price']:.2f} = ${pos['market_value']:,.2f}")
    else:
        print("  Positions: None")
    print()

    # Recent decisions
    decisions = result.get("decisions", [])
    if decisions:
        print("--- Recent Decisions (last 10) ---")
        for d in decisions[-10:]:
            print(f"  {d['date']}: {d['ticker']} @ ${d['price']:.2f} - {d['decision']} (Regime: {d['regime']})")

    print()
    print("=" * 70)


def print_training_result(result: Dict[str, Any]):
    """Print training result to console."""
    print()
    print("=" * 70)
    print("                       TRAINING RESULTS")
    print("=" * 70)

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'Unknown error')}")
        return

    print()
    print(f"Design Document: {result.get('design_doc', 'N/A')}")
    print(f"Tickers: {', '.join(result.get('tickers', []))}")

    date_range = result.get("date_range", {})
    print(f"Training Period: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
    print()

    print("--- Model Training Results ---")
    training_results = result.get("training_results", {})
    for ticker, res in training_results.items():
        print(f"\n  {ticker}:")
        print(f"    Model Type: {res.get('model_type', 'N/A')}")
        print(f"    States: {res.get('n_states', 'N/A')}")
        print(f"    Log-Likelihood: {res.get('log_likelihood', 'N/A'):.2f}")
        print(f"    Training Samples: {res.get('n_samples', 'N/A')}")
        print(f"    Model Path: {res.get('model_path', 'N/A')}")

        if res.get("emission_means"):
            print("    Emission Means (Expected Returns):")
            for regime, mean in res["emission_means"].items():
                print(f"      {regime}: {mean * 100:.2f}%")

    print()
    print("=" * 70)
    print("Training complete! Models saved to models/ directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()