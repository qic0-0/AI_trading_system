"""
Trade Visualization: BUY/SELL Points on Price Chart
=====================================================

Plots price charts with BUY (red ▲) and SELL (blue ▼) markers
from experiment results.

Usage:
    python plot_trades.py --results results/experiment1_results
    python plot_trades.py --results results/experiment1_results --ticker NVDA --mode full --model hmm
    python plot_trades.py --results results/experiment1_results --output plots/
"""

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf


def load_experiment_results(results_dir: str) -> List[Dict]:
    """Load the most recent experiment results JSON from a directory."""
    json_files = glob(os.path.join(results_dir, "experiment*_full_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No experiment JSON files found in {results_dir}")
    
    # Get most recent
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📂 Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"   Found {len(results)} backtest results")
    return results


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily price data for a ticker."""
    print(f"📈 Fetching price data for {ticker}...")
    
    # Extend dates slightly for context
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=7)
    
    df = yf.download(
        ticker, 
        start=start_dt.strftime("%Y-%m-%d"), 
        end=end_dt.strftime("%Y-%m-%d"), 
        progress=False
    )
    
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    
    df = df.reset_index()
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


def extract_trades(decisions: List[Dict]) -> tuple:
    """Extract BUY and SELL trades from decisions."""
    buys = []
    sells = []
    
    for d in decisions:
        decision_str = d.get("decision", "")
        date_str = d.get("date")
        price = d.get("price")
        
        if not date_str or not price:
            continue
        
        # Parse date
        try:
            if "T" in str(date_str):
                date = pd.to_datetime(date_str).to_pydatetime()
            else:
                date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        except:
            continue
        
        if "BUY" in decision_str.upper():
            buys.append({"date": date, "price": float(price)})
        elif "SELL" in decision_str.upper():
            sells.append({"date": date, "price": float(price)})
    
    return buys, sells


def plot_single_result(
    result: Dict,
    price_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    show: bool = True
) -> str:
    """
    Plot price chart with BUY/SELL markers for a single backtest result.
    
    Returns:
        Path to saved figure (if output_dir provided)
    """
    ticker = result["ticker"]
    mode = result["system_mode"]
    model = result.get("model_type", "none")
    decisions = result.get("decisions", [])
    metrics = result.get("metrics", {})
    
    # Extract trades
    buys, sells = extract_trades(decisions)
    
    print(f"   Found {len(buys)} BUY and {len(sells)} SELL trades")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot price line
    dates = pd.to_datetime(price_df["Date"])
    prices = price_df["Close"]
    
    ax.plot(dates, prices, color='#333333', linewidth=1.2, label=f'{ticker} Price', zorder=1)
    
    # Fill under the price line for better visibility
    ax.fill_between(dates, prices, alpha=0.1, color='gray')
    
    # Plot BUY markers (red triangles pointing up)
    if buys:
        buy_dates = [b["date"] for b in buys]
        buy_prices = [b["price"] for b in buys]
        ax.scatter(
            buy_dates, buy_prices,
            color='red', 
            marker='^',  # Triangle up
            s=150,  # Size
            label=f'BUY ({len(buys)})',
            zorder=3,
            edgecolors='darkred',
            linewidths=1.5
        )
    
    # Plot SELL markers (blue triangles pointing down)
    if sells:
        sell_dates = [s["date"] for s in sells]
        sell_prices = [s["price"] for s in sells]
        ax.scatter(
            sell_dates, sell_prices,
            color='blue',
            marker='v',  # Triangle down
            s=150,  # Size
            label=f'SELL ({len(sells)})',
            zorder=3,
            edgecolors='darkblue',
            linewidths=1.5
        )
    
    # Format x-axis for 1-year data
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor tick every month
    
    # Rotate date labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    
    # Build title with metrics
    title = f"{ticker} - Mode: {mode.upper()}"
    if model != "none":
        title += f" | Model: {model.upper()}"
    
    # Add return info to title
    ret_pct = metrics.get("cumulative_return_pct", "N/A")
    sharpe = metrics.get("sharpe_ratio", "N/A")
    title += f"\nReturn: {ret_pct} | Sharpe: {sharpe}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Add annotation box with trade summary
    textstr = f"Total Trades: {len(buys) + len(sells)}\nBuys: {len(buys)}\nSells: {len(sells)}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save if output directory provided
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{ticker}_{mode}_{model}_trades.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def plot_all_results(
    results: List[Dict],
    output_dir: str,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-01"
):
    """Plot all backtest results."""
    # Group by ticker to avoid re-fetching price data
    ticker_results = {}
    for r in results:
        if not r.get("success"):
            continue
        ticker = r["ticker"]
        if ticker not in ticker_results:
            ticker_results[ticker] = []
        ticker_results[ticker].append(r)
    
    print(f"\n📊 Plotting {len(results)} results for {len(ticker_results)} tickers...")
    
    for ticker, ticker_runs in ticker_results.items():
        print(f"\n{'='*50}")
        print(f"Ticker: {ticker} ({len(ticker_runs)} configurations)")
        print('='*50)
        
        # Fetch price data once per ticker
        try:
            price_df = get_price_data(ticker, start_date, end_date)
        except Exception as e:
            print(f"   ❌ Error fetching price data: {e}")
            continue
        
        # Plot each configuration
        for r in ticker_runs:
            mode = r["system_mode"]
            model = r.get("model_type", "none")
            print(f"\n   📈 Plotting: {mode} / {model}")
            
            try:
                plot_single_result(r, price_df, output_dir=output_dir, show=False)
            except Exception as e:
                print(f"   ❌ Error plotting: {e}")
    
    print(f"\n✅ All plots saved to: {output_dir}")


def plot_comparison(
    results: List[Dict],
    ticker: str,
    output_dir: str,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-01"
):
    """
    Create a comparison plot with all configurations for a single ticker
    in a subplot grid.
    """
    # Filter results for this ticker
    ticker_results = [r for r in results if r.get("ticker") == ticker and r.get("success")]
    
    if not ticker_results:
        print(f"No results found for {ticker}")
        return
    
    # Fetch price data
    price_df = get_price_data(ticker, start_date, end_date)
    dates = pd.to_datetime(price_df["Date"])
    prices = price_df["Close"]
    
    # Create subplot grid (2 rows, 3 columns for 5 configurations)
    n_results = len(ticker_results)
    n_cols = 3
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_results > 1 else [axes]
    
    for idx, r in enumerate(ticker_results):
        ax = axes[idx]
        mode = r["system_mode"]
        model = r.get("model_type", "none")
        decisions = r.get("decisions", [])
        metrics = r.get("metrics", {})
        
        buys, sells = extract_trades(decisions)
        
        # Plot price
        ax.plot(dates, prices, color='#333333', linewidth=1, zorder=1)
        ax.fill_between(dates, prices, alpha=0.1, color='gray')
        
        # Plot BUY markers
        if buys:
            buy_dates = [b["date"] for b in buys]
            buy_prices = [b["price"] for b in buys]
            ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=80,
                      label=f'BUY ({len(buys)})', zorder=3, edgecolors='darkred', linewidths=1)
        
        # Plot SELL markers
        if sells:
            sell_dates = [s["date"] for s in sells]
            sell_prices = [s["price"] for s in sells]
            ax.scatter(sell_dates, sell_prices, color='blue', marker='v', s=80,
                      label=f'SELL ({len(sells)})', zorder=3, edgecolors='darkblue', linewidths=1)
        
        # Format
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=8)
        
        # Title with metrics
        title = f"{mode.upper()}"
        if model != "none":
            title += f" + {model.upper()}"
        ret_pct = metrics.get("cumulative_return_pct", "N/A")
        title += f"\nReturn: {ret_pct}"
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(ticker_results), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"{ticker} - Trade Timing Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{ticker}_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved comparison: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot BUY/SELL points on price charts")
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to experiment results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/trades",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Plot only specific ticker"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quant_only", "llm_only"],
        help="Plot only specific system mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hmm", "xgboost", "none"],
        help="Plot only specific model type"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create comparison subplot for each ticker"
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_experiment_results(args.results)
    
    # Filter if specific options provided
    if args.ticker:
        results = [r for r in results if r.get("ticker") == args.ticker]
    if args.mode:
        results = [r for r in results if r.get("system_mode") == args.mode]
    if args.model:
        results = [r for r in results if r.get("model_type") == args.model]
    
    if not results:
        print("❌ No matching results found")
        return
    
    print(f"📊 Found {len(results)} matching results")
    
    # Create comparison plots if requested
    if args.comparison:
        tickers = list(set(r["ticker"] for r in results if r.get("success")))
        for ticker in tickers:
            plot_comparison(results, ticker, args.output)
    else:
        # Plot all results individually
        plot_all_results(results, args.output)


if __name__ == "__main__":
    main()
