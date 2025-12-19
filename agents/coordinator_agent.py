"""
Coordinator/Portfolio Agent - Agent 5 in the Quant Trading System.

Responsibilities:
- Aggregate signals from Quant Model and Market-Sense agents
- Manage portfolio state (positions, cash)
- Validate trades (can only sell what we own)
- Execute trades within constraints
- Risk management (position limits, cash reserves)

Key Design:
- Combines signal aggregation + portfolio management
- Long-only (no shorting)
- Enforces position limits and cash reserves
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from .base_agent import BaseAgent, AgentOutput
from llm.llm_client import Message


@dataclass
class Position:
    """Represents a position in a single ticker."""
    ticker: str
    shares: int
    cost_basis: float  # Average cost per share
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.cost_basis) * self.shares


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    def to_dict(self) -> Dict:
        return {
            "cash": self.cash,
            "positions": {
                ticker: {
                    "shares": pos.shares,
                    "cost_basis": pos.cost_basis,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value
                }
                for ticker, pos in self.positions.items()
            },
            "total_value": self.total_value
        }


@dataclass
class TradeOrder:
    """A validated trade order."""
    ticker: str
    action: str  # "BUY" or "SELL"
    shares: int
    price: float
    reasoning: str
    timestamp: datetime = None
    
    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now()


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    order: Optional[TradeOrder]
    message: str
    new_portfolio_state: Optional[PortfolioState]


class CoordinatorAgent(BaseAgent):
    """
    Agent responsible for signal aggregation and portfolio management.
    
    Signal Aggregation:
    - Combines quant model signal with market-sense insight
    - Resolves conflicts between signals
    - Determines trade direction and size
    
    Portfolio Management:
    - Tracks current positions and cash
    - Validates trades (can only sell what we own)
    - Enforces position limits
    - Maintains minimum cash reserve
    
    Constraints:
    - Long-only (no shorting)
    - Max position size: 30% of portfolio
    - Min cash reserve: 10% of portfolio
    """
    
    def __init__(self, llm_client, config):
        super().__init__("CoordinatorAgent", llm_client, config)
        self.portfolio = PortfolioState(cash=config.portfolio.initial_cash)
        self.max_position_pct = config.portfolio.max_position_pct
        self.min_cash_reserve_pct = config.portfolio.min_cash_reserve_pct
        self.trade_history: List[TradeOrder] = []
        
    @property
    def system_prompt(self) -> str:
        return """You are a Coordinator/Portfolio Agent for a quantitative trading system.

Your role is to:
1. Aggregate signals from the Quant Model and Market-Sense agents
2. Resolve conflicts when signals disagree
3. Determine appropriate position sizes
4. Validate trades against portfolio constraints
5. Execute trades and update portfolio state

You have access to the following tools:
- get_portfolio_state: Get current positions and cash
- validate_trade: Check if a trade is valid
- execute_trade: Execute a validated trade
- calculate_position_size: Determine shares to trade

Constraints you MUST enforce:
- Can only SELL shares we currently own (no shorting)
- Can only BUY with available cash
- Max position size: 30% of portfolio in single ticker
- Min cash reserve: Always keep 10% in cash

When aggregating signals:
- If both Quant and Market-Sense agree: Higher confidence
- If they disagree: Use reasoning to resolve
- Consider confidence levels from both sources
- When uncertain, prefer smaller positions or no action

Always explain your reasoning for the final decision."""

    def _register_tools(self):
        """Register portfolio management tools."""
        
        self.register_tool(
            name="get_portfolio_state",
            func=self.get_portfolio_state,
            description="Get current portfolio state (positions and cash)",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        
        self.register_tool(
            name="validate_trade",
            func=self.validate_trade,
            description="Validate if a trade can be executed",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["BUY", "SELL"]},
                    "ticker": {"type": "string"},
                    "shares": {"type": "integer"},
                    "price": {"type": "number"}
                },
                "required": ["action", "ticker", "shares", "price"]
            }
        )
        
        self.register_tool(
            name="execute_trade",
            func=self.execute_trade,
            description="Execute a validated trade",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["BUY", "SELL"]},
                    "ticker": {"type": "string"},
                    "shares": {"type": "integer"},
                    "price": {"type": "number"},
                    "reasoning": {"type": "string"}
                },
                "required": ["action", "ticker", "shares", "price", "reasoning"]
            }
        )
        
        self.register_tool(
            name="calculate_position_size",
            func=self.calculate_position_size,
            description="Calculate appropriate position size based on signal strength",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "signal_strength": {"type": "number", "description": "Signal strength 0-1"},
                    "current_price": {"type": "number"}
                },
                "required": ["ticker", "signal_strength", "current_price"]
            }
        )

    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Aggregate signals and manage portfolio.

        Args:
            input_data: {
                "ticker": str,
                "current_price": float,
                "quant_signal": Dict,
                "market_insight": Dict
            }

        Returns:
            AgentOutput with trade decision and portfolio state
        """
        ticker = input_data.get("ticker", "")
        current_price = input_data.get("current_price", 0)
        quant_signal = input_data.get("quant_signal", {})
        market_insight = input_data.get("market_insight", {})

        logs = []

        try:
            # Update current prices in portfolio
            if ticker in self.portfolio.positions:
                self.portfolio.positions[ticker].current_price = current_price

            logs.append(f"Processing decision for {ticker} @ ${current_price:.2f}")
            logs.append(f"Portfolio value: ${self.portfolio.total_value:.2f}")

            # Aggregate signals
            aggregated = self.aggregate_signals(quant_signal, market_insight)
            logs.append(f"Aggregated signal: {aggregated['direction']} (confidence: {aggregated['confidence']:.2f})")
            logs.append(f"Reasoning: {aggregated['reasoning']}")

            direction = aggregated["direction"]
            signal_strength = aggregated["strength"]
            confidence = aggregated["confidence"]

            # Determine action based on direction and current position
            trade_result = None

            if direction == "HOLD":
                logs.append("Decision: HOLD - No trade executed")
                trade_result = TradeResult(
                    success=True,
                    order=None,
                    message="HOLD - No trade executed",
                    new_portfolio_state=self.portfolio
                )

            elif direction == "BUY":
                # Calculate position size
                shares = self.calculate_position_size(ticker, signal_strength, current_price)

                if shares > 0:
                    logs.append(f"Decision: BUY {shares} shares of {ticker}")
                    trade_result = self.execute_trade(
                        action="BUY",
                        ticker=ticker,
                        shares=shares,
                        price=current_price,
                        reasoning=aggregated["reasoning"]
                    )
                    logs.append(f"Trade result: {trade_result.message}")
                else:
                    logs.append("Decision: BUY signal but position size is 0 (constraints)")
                    trade_result = TradeResult(
                        success=True,
                        order=None,
                        message="BUY signal but cannot execute (constraints or insufficient funds)",
                        new_portfolio_state=self.portfolio
                    )

            elif direction == "SELL":
                # Check if we have a position to sell
                if ticker in self.portfolio.positions:
                    position = self.portfolio.positions[ticker]
                    # Sell proportional to signal strength
                    shares_to_sell = int(position.shares * signal_strength)
                    shares_to_sell = max(1, shares_to_sell)  # Sell at least 1 share
                    shares_to_sell = min(shares_to_sell, position.shares)  # Can't sell more than we own

                    logs.append(f"Decision: SELL {shares_to_sell} shares of {ticker}")
                    trade_result = self.execute_trade(
                        action="SELL",
                        ticker=ticker,
                        shares=shares_to_sell,
                        price=current_price,
                        reasoning=aggregated["reasoning"]
                    )
                    logs.append(f"Trade result: {trade_result.message}")
                else:
                    logs.append(f"Decision: SELL signal but no position in {ticker}")
                    trade_result = TradeResult(
                        success=True,
                        order=None,
                        message=f"SELL signal but no position in {ticker} (long-only)",
                        new_portfolio_state=self.portfolio
                    )

            return AgentOutput(
                success=True,
                data={
                    "trade_result": {
                        "success": trade_result.success,
                        "message": trade_result.message,
                        "order": {
                            "ticker": trade_result.order.ticker,
                            "action": trade_result.order.action,
                            "shares": trade_result.order.shares,
                            "price": trade_result.order.price,
                            "reasoning": trade_result.order.reasoning
                        } if trade_result.order else None
                    },
                    "portfolio_state": self.portfolio.to_dict(),
                    "aggregated_signal": aggregated,
                    "ticker": ticker
                },
                message=f"Decision for {ticker}: {direction}",
                logs=logs
            )

        except Exception as e:
            logger.error(f"Error in coordinator: {e}")
            return AgentOutput(
                success=False,
                data={"portfolio_state": self.portfolio.to_dict()},
                message=f"Coordinator error: {str(e)}",
                logs=logs
            )
    
    # ==================== Tool Implementations ====================
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Portfolio state as dict with cash, positions, total_value
        """
        return self.portfolio.to_dict()

    def validate_trade(
            self,
            action: str,
            ticker: str,
            shares: int,
            price: float
    ) -> Dict[str, Any]:
        """
        Validate if a trade can be executed.

        Args:
            action: "BUY" or "SELL"
            ticker: Stock ticker
            shares: Number of shares
            price: Price per share

        Returns:
            Dict with valid (bool) and reason (str)
        """
        trade_value = shares * price

        if action == "SELL":
            # Check if we own enough shares
            if ticker not in self.portfolio.positions:
                return {"valid": False, "reason": f"Cannot SELL {ticker}: no position held"}

            position = self.portfolio.positions[ticker]
            if position.shares < shares:
                return {"valid": False,
                        "reason": f"Cannot SELL {shares} shares of {ticker}: only own {position.shares}"}

            return {"valid": True, "reason": "Trade validated"}

        elif action == "BUY":
            # Check cash reserve requirement
            min_reserve = self.portfolio.total_value * self.min_cash_reserve_pct
            available_cash = self.portfolio.cash - min_reserve

            if trade_value > available_cash:
                return {"valid": False,
                        "reason": f"Insufficient cash: need ${trade_value:.2f}, available ${available_cash:.2f} (after {self.min_cash_reserve_pct * 100:.0f}% reserve)"}

            # Check position limit
            current_position_value = 0
            if ticker in self.portfolio.positions:
                current_position_value = self.portfolio.positions[ticker].market_value

            new_position_value = current_position_value + trade_value
            max_position_value = self.portfolio.total_value * self.max_position_pct

            if new_position_value > max_position_value:
                return {"valid": False,
                        "reason": f"Position limit exceeded: ${new_position_value:.2f} would exceed {self.max_position_pct * 100:.0f}% limit (${max_position_value:.2f})"}

            return {"valid": True, "reason": "Trade validated"}

        else:
            return {"valid": False, "reason": f"Invalid action: {action}. Must be BUY or SELL"}

    def execute_trade(
            self,
            action: str,
            ticker: str,
            shares: int,
            price: float,
            reasoning: str
    ) -> TradeResult:
        """
        Execute a validated trade.

        Args:
            action: "BUY" or "SELL"
            ticker: Stock ticker
            shares: Number of shares
            price: Price per share
            reasoning: Why this trade was made

        Returns:
            TradeResult with success status and updated portfolio
        """
        # Validate first
        validation = self.validate_trade(action, ticker, shares, price)
        if not validation["valid"]:
            return TradeResult(
                success=False,
                order=None,
                message=validation["reason"],
                new_portfolio_state=self.portfolio
            )

        trade_value = shares * price

        # Create order
        order = TradeOrder(
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            reasoning=reasoning
        )

        if action == "BUY":
            # Decrease cash
            self.portfolio.cash -= trade_value

            # Update or create position
            if ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                # Calculate new average cost basis
                total_cost = (pos.shares * pos.cost_basis) + trade_value
                total_shares = pos.shares + shares
                pos.cost_basis = total_cost / total_shares
                pos.shares = total_shares
                pos.current_price = price
            else:
                self.portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    cost_basis=price,
                    current_price=price
                )

            logger.info(f"BUY executed: {shares} shares of {ticker} @ ${price:.2f}")

        elif action == "SELL":
            # Increase cash
            self.portfolio.cash += trade_value

            # Update position
            pos = self.portfolio.positions[ticker]
            pos.shares -= shares
            pos.current_price = price

            # Remove position if fully sold
            if pos.shares == 0:
                del self.portfolio.positions[ticker]

            logger.info(f"SELL executed: {shares} shares of {ticker} @ ${price:.2f}")

        # Record in history
        self.trade_history.append(order)

        return TradeResult(
            success=True,
            order=order,
            message=f"{action} {shares} shares of {ticker} @ ${price:.2f}",
            new_portfolio_state=self.portfolio
        )

    def calculate_position_size(
            self,
            ticker: str,
            signal_strength: float,
            current_price: float
    ) -> int:
        """
        Calculate appropriate position size.

        Args:
            ticker: Stock ticker
            signal_strength: Signal confidence (0.0 to 1.0)
            current_price: Current stock price

        Returns:
            Number of shares to trade
        """
        if current_price <= 0:
            return 0

        # Calculate maximum position value (respecting position limit)
        max_position_value = self.portfolio.total_value * self.max_position_pct

        # Account for existing position
        current_position_value = 0
        if ticker in self.portfolio.positions:
            current_position_value = self.portfolio.positions[ticker].shares * current_price

        # Available room for this position
        available_position_value = max_position_value - current_position_value

        # Calculate available cash (respecting cash reserve)
        min_reserve = self.portfolio.total_value * self.min_cash_reserve_pct
        available_cash = max(0, self.portfolio.cash - min_reserve)

        # Take minimum of position limit and available cash
        max_trade_value = min(available_position_value, available_cash)

        # Scale by signal strength (stronger signal = larger position)
        # Use signal_strength squared to be more conservative
        scaled_value = max_trade_value * signal_strength

        # Convert to shares (round down)
        shares = int(scaled_value / current_price)

        logger.debug(
            f"Position size for {ticker}: {shares} shares (signal={signal_strength:.2f}, price=${current_price:.2f})")

        return shares

    def aggregate_signals(
            self,
            quant_signal: Dict,
            market_insight: Dict
    ) -> Dict[str, Any]:
        """
        Aggregate signals from Quant Model and Market-Sense agents.

        Handles three modes:
        - full: Both signals present, aggregate them
        - quant_only: Only quant signal, market_insight is neutral
        - llm_only: Only market_insight, quant_signal is neutral

        Args:
            quant_signal: For HMM: {expected_return, confidence, regime, model_type}
                          For XGBoost: {direction_probability, confidence, predicted_direction, model_type}
            market_insight: {outlook, confidence, reasoning, risk_flags}

        Returns:
            Aggregated signal with direction, strength, confidence, reasoning
        """
        model_type = quant_signal.get("model_type", "hmm")

        # Check if agents are disabled
        quant_disabled = (model_type == "none" or quant_signal.get("confidence", 0) == 0
                          and quant_signal.get("regime") == "DISABLED")
        market_disabled = (market_insight.get("outlook") == "NEUTRAL"
                           and market_insight.get("confidence", 0) == 0
                           and "disabled" in market_insight.get("reasoning", "").lower())

        # ========== QUANT-ONLY MODE ==========
        if market_disabled and not quant_disabled:
            # Use only quant signal
            if model_type == "hmm":
                expected_return = quant_signal.get("expected_return", 0)
                quant_confidence = quant_signal.get("confidence", 0.5)
                regime = quant_signal.get("regime", "Unknown")

                if expected_return > 0.01:
                    final_direction = "BUY"
                elif expected_return < -0.01:
                    final_direction = "SELL"
                else:
                    final_direction = "HOLD"
            elif model_type == "xgboost":
                direction_prob = quant_signal.get("direction_probability", 0.5)
                quant_confidence = quant_signal.get("confidence", 0.5)
                predicted_direction = quant_signal.get("predicted_direction", "NEUTRAL")
                regime = "N/A"
                expected_return = (direction_prob - 0.5) * 0.1

                if predicted_direction == "UP":
                    final_direction = "BUY"
                elif predicted_direction == "DOWN":
                    final_direction = "SELL"
                else:
                    final_direction = "HOLD"
            else:
                expected_return = 0
                quant_confidence = 0.5
                regime = "Unknown"
                final_direction = "HOLD"

            signal_strength = abs(expected_return) * quant_confidence * 50
            signal_strength = min(1.0, max(0.0, signal_strength))

            return {
                "direction": final_direction,
                "strength": signal_strength,
                "confidence": quant_confidence,
                "reasoning": f"[QUANT-ONLY] {model_type} signal: {final_direction} (regime: {regime})",
                "quant_direction": final_direction,
                "market_direction": "DISABLED",
                "regime": regime,
                "model_type": model_type
            }

        # ========== LLM-ONLY MODE ==========
        if quant_disabled and not market_disabled:
            # Use only market insight
            market_outlook = market_insight.get("outlook", "NEUTRAL")
            market_confidence = market_insight.get("confidence", 0.5)
            market_reasoning = market_insight.get("reasoning", "")
            risk_flags = market_insight.get("risk_flags", [])

            if market_outlook == "BULLISH":
                final_direction = "BUY"
                expected_return = 0.02 * market_confidence  # Estimate
            elif market_outlook == "BEARISH":
                final_direction = "SELL"
                expected_return = -0.02 * market_confidence
            else:
                final_direction = "HOLD"
                expected_return = 0

            # Reduce confidence if risk flags present
            if risk_flags:
                market_confidence *= 0.8

            signal_strength = abs(expected_return) * market_confidence * 50
            signal_strength = min(1.0, max(0.0, signal_strength))

            return {
                "direction": final_direction,
                "strength": signal_strength,
                "confidence": market_confidence,
                "reasoning": f"[LLM-ONLY] Market-Sense: {market_outlook}. {market_reasoning[:100]}",
                "quant_direction": "DISABLED",
                "market_direction": final_direction,
                "regime": "N/A",
                "model_type": "none"
            }

        # ========== FULL MODE (original logic) ==========
        # Extract quant signal based on model type
        if model_type == "hmm":
            expected_return = quant_signal.get("expected_return", 0)
            quant_confidence = quant_signal.get("confidence", 0.5)
            regime = quant_signal.get("regime", "Unknown")

            if expected_return > 0.01:
                quant_direction = "BUY"
            elif expected_return < -0.01:
                quant_direction = "SELL"
            else:
                quant_direction = "HOLD"

        elif model_type == "xgboost":
            direction_prob = quant_signal.get("direction_probability", 0.5)
            quant_confidence = quant_signal.get("confidence", 0.5)
            predicted_direction = quant_signal.get("predicted_direction", "NEUTRAL")
            regime = "N/A"

            if predicted_direction == "UP":
                quant_direction = "BUY"
                expected_return = (direction_prob - 0.5) * 0.1
            elif predicted_direction == "DOWN":
                quant_direction = "SELL"
                expected_return = (direction_prob - 0.5) * 0.1
            else:
                quant_direction = "HOLD"
                expected_return = 0
        else:
            expected_return = quant_signal.get("expected_return", 0)
            quant_confidence = quant_signal.get("confidence", 0.5)
            regime = quant_signal.get("regime", "Unknown")
            quant_direction = "HOLD"

        # Extract market insight info
        market_outlook = market_insight.get("outlook", "NEUTRAL")
        market_confidence = market_insight.get("confidence", 0.5)
        risk_flags = market_insight.get("risk_flags", [])

        if market_outlook == "BULLISH":
            market_direction = "BUY"
        elif market_outlook == "BEARISH":
            market_direction = "SELL"
        else:
            market_direction = "HOLD"

        # Aggregate based on agreement
        if quant_direction == market_direction:
            final_direction = quant_direction
            final_confidence = (quant_confidence + market_confidence) / 2
            reasoning = f"[FULL] Quant[{model_type}] ({quant_direction}) and Market-Sense ({market_outlook}) agree."
        elif quant_direction == "HOLD" or market_direction == "HOLD":
            if quant_direction != "HOLD":
                final_direction = quant_direction
                final_confidence = quant_confidence * 0.7
                reasoning = f"[FULL] Quant[{model_type}] suggests {quant_direction}, Market-Sense is neutral."
            else:
                final_direction = market_direction
                final_confidence = market_confidence * 0.7
                reasoning = f"[FULL] Market-Sense suggests {market_direction}, Quant[{model_type}] is neutral."
        else:
            if quant_confidence > market_confidence + 0.2:
                final_direction = quant_direction
                final_confidence = quant_confidence * 0.5
                reasoning = f"[FULL] Signals conflict. Going with Quant[{model_type}] ({quant_direction})."
            elif market_confidence > quant_confidence + 0.2:
                final_direction = market_direction
                final_confidence = market_confidence * 0.5
                reasoning = f"[FULL] Signals conflict. Going with Market-Sense ({market_direction})."
            else:
                final_direction = "HOLD"
                final_confidence = 0.3
                reasoning = f"[FULL] Signals conflict. Holding due to uncertainty."

        if risk_flags:
            final_confidence *= 0.8
            reasoning += f" Risk flags: {', '.join(risk_flags)}."

        signal_strength = abs(expected_return) * final_confidence * 50
        signal_strength = min(1.0, max(0.0, signal_strength))

        return {
            "direction": final_direction,
            "strength": signal_strength,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "quant_direction": quant_direction,
            "market_direction": market_direction,
            "regime": regime,
            "model_type": model_type
        }



