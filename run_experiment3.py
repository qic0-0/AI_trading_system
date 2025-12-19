"""
Experiment 3: LLM-Only Mode Test
=================================

Tests llm_only mode on all 6 tickers from Experiments 1 and 2.

Stocks:
    - From Exp1: NVDA (increasing), MRNA (decreasing), KO (stable)
    - From Exp2: JNJ (increasing), VZ (decreasing), F (stable)

Configuration:
    - llm_only mode only (Market-Sense agent only)

Usage:
    python run_experiment3.py
    python run_experiment3.py --output results/experiment3
    python run_experiment3.py --stocks NVDA MRNA
"""

import argparse
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from copy import deepcopy

# Import trading system components
from config.config import config, SystemConfig, PortfolioConfig
from main import TradingSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== EXPERIMENT CONFIGURATION ==============

EXPERIMENT_CONFIG = {
    "name": "Experiment 3: LLM-Only Mode Test",
    "start_date": "2024-01-01",
    "end_date": "2024-12-01",
    "initial_cash": 100000.0,
    "risk_free_rate": 0.05,  # 5% annualized for Sharpe ratio
    
    # All 6 tickers from Experiments 1 and 2
    "stocks": {
        "exp1_increasing": ["NVDA"],
        "exp1_decreasing": ["MRNA"],
        "exp1_stable": ["KO"],
        "exp2_increasing": ["JNJ"],
        "exp2_decreasing": ["VZ"],
        "exp2_stable": ["F"]
    },
    
    "system_modes": ["llm_only"],
    
    "model_types": ["none"]  # Not used for llm_only
}


# ============== METRICS CALCULATION ==============

def calculate_metrics(
    decisions: List[Dict],
    portfolio_values: List[Dict],
    trade_history: List,
    initial_value: float,
    final_value: float,
    risk_free_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a backtest run.
    """
    metrics = {}
    
    # 1. Cumulative Return
    cumulative_return = (final_value - initial_value) / initial_value
    metrics["cumulative_return"] = cumulative_return
    metrics["cumulative_return_pct"] = f"{cumulative_return * 100:.2f}%"
    
    # 2. Sharpe Ratio
    if len(portfolio_values) > 1:
        values = [pv["value"] for pv in portfolio_values]
        returns = pd.Series(values).pct_change().dropna()
        
        if len(returns) > 0 and returns.std() > 0:
            periods_per_year = 252 / 5
            annualized_return = returns.mean() * periods_per_year
            annualized_std = returns.std() * np.sqrt(periods_per_year)
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
            metrics["sharpe_ratio"] = round(sharpe_ratio, 3)
        else:
            metrics["sharpe_ratio"] = 0.0
    else:
        metrics["sharpe_ratio"] = 0.0
    
    # 3. Max Drawdown
    if len(portfolio_values) > 0:
        values = [pv["value"] for pv in portfolio_values]
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        metrics["max_drawdown"] = round(max_drawdown, 4)
        metrics["max_drawdown_pct"] = f"{max_drawdown * 100:.2f}%"
    else:
        metrics["max_drawdown"] = 0.0
        metrics["max_drawdown_pct"] = "0.00%"
    
    # 4. Directional Accuracy
    correct_predictions = 0
    total_predictions = 0
    
    for i, decision in enumerate(decisions):
        if i == 0:
            continue
        
        decision_str = decision.get("decision", "")
        if "BUY" in decision_str:
            predicted_direction = "UP"
        elif "SELL" in decision_str:
            predicted_direction = "DOWN"
        else:
            continue
        
        current_price = decision.get("price", 0)
        prev_price = decisions[i-1].get("price", 0)
        
        if prev_price > 0:
            actual_direction = "UP" if current_price > prev_price else "DOWN"
            total_predictions += 1
            if predicted_direction == actual_direction:
                correct_predictions += 1
    
    if total_predictions > 0:
        directional_accuracy = correct_predictions / total_predictions
        metrics["directional_accuracy"] = round(directional_accuracy, 4)
        metrics["directional_accuracy_pct"] = f"{directional_accuracy * 100:.2f}%"
    else:
        metrics["directional_accuracy"] = 0.0
        metrics["directional_accuracy_pct"] = "N/A"
    
    # 5. Win Rate
    if len(trade_history) > 0:
        buy_trades = {}
        profitable_trades = 0
        total_closed_trades = 0
        
        for trade in trade_history:
            ticker = trade.ticker
            if trade.action == "BUY":
                if ticker not in buy_trades:
                    buy_trades[ticker] = []
                buy_trades[ticker].append(trade)
            elif trade.action == "SELL":
                if ticker in buy_trades and len(buy_trades[ticker]) > 0:
                    buy_trade = buy_trades[ticker].pop(0)
                    profit = (trade.price - buy_trade.price) * trade.shares
                    total_closed_trades += 1
                    if profit > 0:
                        profitable_trades += 1
        
        if total_closed_trades > 0:
            win_rate = profitable_trades / total_closed_trades
            metrics["win_rate"] = round(win_rate, 4)
            metrics["win_rate_pct"] = f"{win_rate * 100:.2f}%"
        else:
            metrics["win_rate"] = 0.0
            metrics["win_rate_pct"] = "N/A"
    else:
        metrics["win_rate"] = 0.0
        metrics["win_rate_pct"] = "N/A"
    
    # 6. Number of Trades
    metrics["num_trades"] = len(trade_history)
    
    # 7. Additional info
    metrics["initial_value"] = initial_value
    metrics["final_value"] = round(final_value, 2)
    metrics["num_decisions"] = len(decisions)
    
    return metrics


# ============== EXPERIMENT RUNNER ==============

class ExperimentRunner:
    """Runs Experiment 3: LLM-Only backtests across 6 stocks."""
    
    def __init__(self, output_dir: str = "results/experiment3"):
        self.output_dir = output_dir
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Experiment results will be saved to: {output_dir}")
    
    def save_incremental(self, result: Dict[str, Any]):
        """Save result immediately after each backtest completes."""
        json_path = os.path.join(self.output_dir, f"experiment3_full_{self.timestamp}.json")
        
        existing = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        existing.append(result)
        with open(json_path, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
        
        logger.info(f"  💾 Saved to {json_path} ({len(existing)} results)")
    
    def run_single_backtest(
        self,
        ticker: str,
        category: str,
        system_mode: str,
        model_type: str,
        start_date: str,
        end_date: str,
        initial_cash: float
    ) -> Dict[str, Any]:
        """Run a single backtest and return results."""
        logger.info(f"Running: {ticker} | Mode: {system_mode} | Model: {model_type} | Category: {category}")
        
        try:
            # Create fresh config with reset portfolio
            test_config = deepcopy(config)
            test_config.portfolio.initial_cash = initial_cash
            test_config.model.model_type = model_type
            
            # Create fresh trading system
            system = TradingSystem(test_config, system_mode=system_mode)
            
            # Run backtest
            result = system.run_backtest(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                system_mode=system_mode
            )
            
            if not result.get("success"):
                logger.error(f"Backtest failed for {ticker}/{system_mode}/{model_type}: {result.get('message')}")
                return {
                    "ticker": ticker,
                    "category": category,
                    "system_mode": system_mode,
                    "model_type": model_type,
                    "success": False,
                    "error": result.get("message", "Unknown error")
                }
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(
                decisions=result.get("decisions", []),
                portfolio_values=result.get("portfolio_values", []),
                trade_history=system.coordinator_agent.trade_history,
                initial_value=initial_cash,
                final_value=result.get("metrics", {}).get("final_value", initial_cash),
                risk_free_rate=EXPERIMENT_CONFIG["risk_free_rate"]
            )
            
            # Compile result
            backtest_result = {
                "ticker": ticker,
                "category": category,
                "system_mode": system_mode,
                "model_type": model_type,
                "success": True,
                "metrics": metrics,
                "decisions": result.get("decisions", []),
                "portfolio_values": result.get("portfolio_values", []),
                "final_portfolio": result.get("final_portfolio", {})
            }
            
            logger.info(f"  ✓ Return: {metrics['cumulative_return_pct']} | Sharpe: {metrics['sharpe_ratio']} | MaxDD: {metrics['max_drawdown_pct']}")
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Exception in {ticker}/{system_mode}/{model_type}: {str(e)}")
            return {
                "ticker": ticker,
                "category": category,
                "system_mode": system_mode,
                "model_type": model_type,
                "success": False,
                "error": str(e)
            }
    
    def run_all(
        self,
        stocks: Dict[str, List[str]] = None,
        modes: List[str] = None,
        model_types: List[str] = None
    ):
        """Run all backtests."""
        stocks = stocks or EXPERIMENT_CONFIG["stocks"]
        modes = modes or EXPERIMENT_CONFIG["system_modes"]
        model_types = model_types or EXPERIMENT_CONFIG["model_types"]
        
        start_date = EXPERIMENT_CONFIG["start_date"]
        end_date = EXPERIMENT_CONFIG["end_date"]
        initial_cash = EXPERIMENT_CONFIG["initial_cash"]
        
        total_stocks = sum(len(tickers) for tickers in stocks.values())
        total_runs = total_stocks
        
        logger.info("=" * 70)
        logger.info(f"EXPERIMENT 3: {EXPERIMENT_CONFIG['name']}")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Cash: ${initial_cash:,.2f}")
        logger.info(f"Stocks: {total_stocks} | Mode: llm_only")
        logger.info(f"Total Backtests: {total_runs}")
        logger.info("=" * 70)
        
        run_count = 0
        
        for category, tickers in stocks.items():
            logger.info(f"\n--- Category: {category.upper()} ---")
            
            for ticker in tickers:
                run_count += 1
                logger.info(f"\n[{run_count}/{total_runs}] {ticker} - llm_only (no quant model)")
                
                result = self.run_single_backtest(
                    ticker=ticker,
                    category=category,
                    system_mode="llm_only",
                    model_type="none",
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=initial_cash
                )
                self.results.append(result)
                self.save_incremental(result)
        
        logger.info("\n" + "=" * 70)
        logger.info("ALL BACKTESTS COMPLETE")
        logger.info("=" * 70)
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save final results to CSV and summary files."""
        json_path = os.path.join(self.output_dir, f"experiment3_full_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Full results saved to: {json_path}")
        
        # Save metrics summary to CSV
        csv_path = os.path.join(self.output_dir, f"experiment3_metrics_{self.timestamp}.csv")
        
        rows = []
        for r in self.results:
            if r.get("success"):
                metrics = r.get("metrics", {})
                rows.append({
                    "Ticker": r["ticker"],
                    "Category": r["category"],
                    "System_Mode": r["system_mode"],
                    "Model_Type": r.get("model_type", "none"),
                    "Cumulative_Return": metrics.get("cumulative_return", 0),
                    "Cumulative_Return_Pct": metrics.get("cumulative_return_pct", "N/A"),
                    "Sharpe_Ratio": metrics.get("sharpe_ratio", 0),
                    "Max_Drawdown": metrics.get("max_drawdown", 0),
                    "Max_Drawdown_Pct": metrics.get("max_drawdown_pct", "N/A"),
                    "Directional_Accuracy": metrics.get("directional_accuracy", 0),
                    "Win_Rate": metrics.get("win_rate", 0),
                    "Num_Trades": metrics.get("num_trades", 0),
                    "Final_Value": metrics.get("final_value", 0)
                })
            else:
                rows.append({
                    "Ticker": r["ticker"],
                    "Category": r["category"],
                    "System_Mode": r["system_mode"],
                    "Model_Type": r.get("model_type", "none"),
                    "Cumulative_Return": "ERROR",
                    "Error": r.get("error", "Unknown")
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics CSV saved to: {csv_path}")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, f"experiment3_summary_{self.timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_text())
        logger.info(f"Summary saved to: {summary_path}")
    
    def _generate_summary_text(self) -> str:
        """Generate text summary of results."""
        lines = []
        lines.append("=" * 90)
        lines.append("EXPERIMENT 3: LLM-ONLY MODE TEST - SUMMARY")
        lines.append("=" * 90)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Period: {EXPERIMENT_CONFIG['start_date']} to {EXPERIMENT_CONFIG['end_date']}")
        lines.append("")
        
        lines.append(f"\n{'='*50}")
        lines.append(f"MODE: LLM_ONLY")
        lines.append(f"{'='*50}")
        
        mode_results = [r for r in self.results if r.get("success")]
        
        if not mode_results:
            lines.append("No successful results")
        else:
            lines.append(f"{'Ticker':<8} {'Category':<18} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'DirAcc':>8} {'WinRate':>8} {'Trades':>7}")
            lines.append("-" * 85)
            
            for r in mode_results:
                m = r.get("metrics", {})
                lines.append(
                    f"{r['ticker']:<8} {r['category']:<18} "
                    f"{m.get('cumulative_return_pct', 'N/A'):>10} "
                    f"{m.get('sharpe_ratio', 0):>8.3f} "
                    f"{m.get('max_drawdown_pct', 'N/A'):>10} "
                    f"{m.get('directional_accuracy_pct', 'N/A'):>8} "
                    f"{m.get('win_rate_pct', 'N/A'):>8} "
                    f"{m.get('num_trades', 0):>7}"
                )
            
            avg_return = np.mean([r["metrics"]["cumulative_return"] for r in mode_results])
            avg_sharpe = np.mean([r["metrics"]["sharpe_ratio"] for r in mode_results])
            avg_dd = np.mean([r["metrics"]["max_drawdown"] for r in mode_results])
            avg_trades = np.mean([r["metrics"]["num_trades"] for r in mode_results])
            
            lines.append("-" * 85)
            lines.append(f"{'AVERAGE':<8} {'':<18} {avg_return*100:>9.2f}% {avg_sharpe:>8.3f} {avg_dd*100:>9.2f}% {'':<8} {'':<8} {avg_trades:>7.1f}")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print summary to console."""
        print(self._generate_summary_text())


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3: LLM-Only Mode Test")
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment3",
        help="Output directory for results"
    )
    parser.add_argument(
        "--stocks",
        type=str,
        nargs="+",
        help="Specific stocks to test (overrides default)"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output)
    
    stocks = None
    if args.stocks:
        stocks = {"custom": args.stocks}
    
    runner.run_all(stocks=stocks)


if __name__ == "__main__":
    main()
