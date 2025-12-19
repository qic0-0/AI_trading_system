"""
Market-Sense Agent - Agent 4 in the Quant Trading System.

Responsibilities:
- LLM-based reasoning about market conditions
- News interpretation in market context
- Apply financial/economic knowledge (RAG)
- Compare to historical patterns
- Assess risks and opportunities

Key Design:
- Uses RAG + Prompt Engineering (NOT fine-tuning)
- Knowledge base contains financial concepts, historical events
- Receives market state from Data Agent
- Outputs qualitative market insight
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

from .base_agent import BaseAgent, AgentOutput
from llm.llm_client import Message

@dataclass
class MarketInsight:
    """Output from Market-Sense Agent."""
    outlook: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Explanation
    risk_flags: List[str]  # Any risks identified
    historical_comparison: Optional[str]  # Similar past situations


class MarketSenseAgent(BaseAgent):
    """
    Agent responsible for qualitative market analysis.
    
    Uses RAG to retrieve relevant:
    - Financial/economic concepts
    - Historical market events
    - Investment principles
    
    Then applies LLM reasoning to:
    - Interpret current news
    - Assess market conditions
    - Identify risks
    - Compare to historical patterns
    """
    
    def __init__(self, llm_client, config):
        super().__init__("MarketSenseAgent", llm_client, config)
        self.knowledge_store = None  # Vector store for financial knowledge
        
    @property
    def system_prompt(self) -> str:
        return """You are a Market-Sense Agent - an expert financial analyst for a quantitative trading system.

Your role is to:
1. Analyze market conditions using financial reasoning
2. Interpret news in the context of current market state
3. Apply economic principles to assess impact
4. Compare current situations to historical patterns
5. Identify risks and opportunities
6. Provide qualitative market outlook

You have access to:
- retrieve_knowledge: Search financial/economic knowledge base
- retrieve_historical_events: Find similar past market situations

Knowledge base includes:
- Economic principles (inflation, interest rates, business cycles)
- Financial concepts (valuation, risk, portfolio theory)
- Historical events (2008 crisis, COVID crash, Fed decisions)
- Sector-specific knowledge

When analyzing:
1. First retrieve relevant knowledge
2. Consider the quantitative signal (if provided)
3. Analyze news sentiment and implications
4. Compare to historical patterns
5. Identify any risk flags
6. Provide your outlook with confidence level

Be balanced and objective. Acknowledge uncertainty when appropriate."""

    def _register_tools(self):
        """Register market-sense tools."""
        
        self.register_tool(
            name="retrieve_knowledge",
            func=self.retrieve_knowledge,
            description="Retrieve relevant financial/economic knowledge",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query for knowledge retrieval"},
                    "top_k": {"type": "integer", "description": "Number of results"}
                },
                "required": ["query"]
            }
        )
        
        self.register_tool(
            name="retrieve_historical_events",
            func=self.retrieve_historical_events,
            description="Find similar historical market events",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Description of current situation"},
                    "top_k": {"type": "integer", "description": "Number of events to retrieve"}
                },
                "required": ["query"]
            }
        )
    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Analyze market conditions and provide insight.

        Args:
            input_data: {
                "market_state": Dict,
                "news": List[str],
                "quant_signal": Optional[Dict],
                "ticker": str
            }

        Returns:
            AgentOutput with MarketInsight
        """
        market_state = input_data.get("market_state", {})
        news = input_data.get("news", [])
        quant_signal = input_data.get("quant_signal", {})
        ticker = input_data.get("ticker", "Unknown")

        logs = []

        try:
            # Format market state
            market_state_str = self.format_market_state(market_state)
            logs.append("Formatted market state")

            # Build news summary
            news_str = "\n".join([f"- {n}" for n in news[:5]]) if news else "No recent news"

            # Retrieve relevant knowledge
            knowledge_query = f"{ticker} market conditions {news[0] if news else ''}"
            knowledge = self.retrieve_knowledge(knowledge_query, top_k=3)
            knowledge_str = "\n".join([f"- {k['text']}" for k in knowledge]) if knowledge else "No relevant knowledge found"
            logs.append(f"Retrieved {len(knowledge)} knowledge chunks")

            # Retrieve historical comparisons
            historical_query = f"Market situation: {market_state_str[:200]} {news[0] if news else ''}"
            historical = self.retrieve_historical_events(historical_query, top_k=2)
            historical_str = "\n".join([f"- {h['event']} ({h['date']}): {h['lessons']}" for h in historical]) if historical else "No similar historical events found"
            logs.append(f"Retrieved {len(historical)} historical events")

            # Format quant signal if available
            quant_str = ""
            if quant_signal:
                quant_str = f"""
Quantitative Model Signal:
- Expected Return: {quant_signal.get('expected_return', 'N/A')}
- Regime: {quant_signal.get('regime', 'N/A')}
- Confidence: {quant_signal.get('confidence', 'N/A')}
"""

            # Build prompt for LLM
            analysis_prompt = f"""Analyze the current market situation for {ticker}.

{market_state_str}

Recent News:
{news_str}

{quant_str}

Relevant Financial Knowledge:
{knowledge_str}

Similar Historical Events:
{historical_str}

Based on this information, provide your analysis:
1. What is your overall outlook? (BULLISH, BEARISH, or NEUTRAL)
2. What is your confidence level? (0.0 to 1.0)
3. What is your reasoning?
4. What risk flags do you see? (list any concerns)
5. How does this compare to historical situations?

Respond in the following JSON format:
{{
    "outlook": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "your explanation",
    "risk_flags": ["risk1", "risk2"],
    "historical_comparison": "comparison to past events"
}}"""

            # Call LLM
            messages = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=analysis_prompt)
            ]

            response = self.llm_client.chat(messages, temperature=0.3)
            logs.append("Received LLM response")

            # Parse LLM response
            try:
                import json
                # Try to extract JSON from response
                content = response.content
                # Find JSON in response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")

                insight = MarketInsight(
                    outlook=result.get("outlook", "NEUTRAL"),
                    confidence=float(result.get("confidence", 0.5)),
                    reasoning=result.get("reasoning", ""),
                    risk_flags=result.get("risk_flags", []),
                    historical_comparison=result.get("historical_comparison", "")
                )


            except (json.JSONDecodeError, ValueError) as e:

                logger.warning(f"Failed to parse LLM response as JSON: {e}")

                # Fallback: create insight from raw response

                insight = MarketInsight(

                    outlook="NEUTRAL",

                    confidence=0.5,

                    reasoning=response.content,

                    risk_flags=[],

                    historical_comparison=""

                )

                # FALLBACK: If LLM returns NEUTRAL, use technical indicators

            if insight.outlook == "NEUTRAL":

                ticker_data = market_state.get("ticker_data", {})

                momentum = ticker_data.get("momentum_1m", 0) or 0

                # Use momentum to determine outlook

                if momentum > 0.03:  # >3% monthly gain

                    insight = MarketInsight(

                        outlook="BULLISH",

                        confidence=min(0.7, 0.5 + abs(momentum)),

                        reasoning=f"Technical momentum positive ({momentum * 100:.1f}% 1M). " + insight.reasoning,

                        risk_flags=insight.risk_flags,

                        historical_comparison=insight.historical_comparison

                    )

                    logger.info(f"Fallback: NEUTRAL -> BULLISH based on momentum {momentum:.3f}")

                elif momentum < -0.03:  # <-3% monthly loss

                    insight = MarketInsight(

                        outlook="BEARISH",

                        confidence=min(0.7, 0.5 + abs(momentum)),

                        reasoning=f"Technical momentum negative ({momentum * 100:.1f}% 1M). " + insight.reasoning,

                        risk_flags=insight.risk_flags,

                        historical_comparison=insight.historical_comparison

                    )

                    logger.info(f"Fallback: NEUTRAL -> BEARISH based on momentum {momentum:.3f}")

            logs.append(f"Analysis complete: {insight.outlook} (confidence: {insight.confidence})")

            return AgentOutput(
                success=True,
                data={
                    "insight": {
                        "outlook": insight.outlook,
                        "confidence": insight.confidence,
                        "reasoning": insight.reasoning,
                        "risk_flags": insight.risk_flags,
                        "historical_comparison": insight.historical_comparison
                    },
                    "ticker": ticker
                },
                message=f"Market analysis for {ticker}: {insight.outlook}",
                logs=logs
            )

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return AgentOutput(
                success=False,
                data={},
                message=f"Market analysis failed: {str(e)}",
                logs=logs
            )
    # ==================== Tool Implementations ====================

    def retrieve_knowledge(
            self,
            query: str,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant financial/economic knowledge.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of knowledge chunks
        """
        # Initialize knowledge store if not exists
        if self.knowledge_store is None:
            self._initialize_knowledge_store()

        if not self.knowledge_store:
            logger.warning("Knowledge store is empty")
            return []

        try:
            # Embed query
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            query_embedding = self._embedding_model.encode(query)

            # Search knowledge store
            similarities = []
            for item in self.knowledge_store:
                similarity = np.dot(query_embedding, item["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
                )
                similarities.append({
                    "text": item["text"],
                    "source": item.get("source", ""),
                    "category": item.get("category", ""),
                    "similarity": float(similarity)
                })

            # Sort and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[:top_k]

            # Remove similarity score from output
            for r in results:
                del r["similarity"]

            logger.info(f"Retrieved {len(results)} knowledge chunks for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []



    def retrieve_historical_events(
            self,
            query: str,
            top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical market events.

        Args:
            query: Description of current situation
            top_k: Number of events to retrieve

        Returns:
            List of historical events
        """
        # Initialize historical events if not exists
        if not hasattr(self, '_historical_events'):
            self._initialize_historical_events()

        if not self._historical_events:
            return []

        try:
            # Embed query
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            query_embedding = self._embedding_model.encode(query)

            # Search historical events
            similarities = []
            for event in self._historical_events:
                similarity = np.dot(query_embedding, event["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(event["embedding"])
                )
                similarities.append({
                    "event": event["event"],
                    "date": event["date"],
                    "market_impact": event["market_impact"],
                    "lessons": event["lessons"],
                    "similarity": float(similarity)
                })

            # Sort and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[:top_k]

            # Remove similarity score
            for r in results:
                del r["similarity"]

            logger.info(f"Retrieved {len(results)} historical events")
            return results

        except Exception as e:
            logger.error(f"Error retrieving historical events: {e}")
            return []


    def format_market_state(self, market_state: Dict) -> str:
        """
        Format market state for LLM prompt.

        Args:
            market_state: Raw market state dict

        Returns:
            Formatted string for LLM context
        """
        lines = ["Current Market State:"]

        # Market-wide metrics
        if "sp500" in market_state:
            sp = market_state["sp500"]
            lines.append(f"- S&P 500: {sp.get('price', 'N/A')} ({sp.get('change_pct', 0):+.1f}% today)")

        if "vix" in market_state:
            vix = market_state["vix"]
            vix_level = "low" if vix < 15 else "moderate" if vix < 25 else "high"
            lines.append(f"- VIX: {vix} ({vix_level} volatility)")

        if "treasury_10y" in market_state:
            lines.append(f"- 10Y Treasury: {market_state['treasury_10y']}%")

        if "fed_rate" in market_state:
            lines.append(f"- Fed Funds Rate: {market_state['fed_rate']}%")

        if "cpi" in market_state:
            lines.append(f"- CPI: {market_state['cpi']}% YoY")

        # Ticker-specific data
        if "ticker_data" in market_state:
            td = market_state["ticker_data"]
            ticker = td.get("ticker", "Stock")
            lines.append(f"\n{ticker} Specific:")

            if "price" in td:
                lines.append(f"- Price: ${td['price']}")
            if "pe_ratio" in td:
                lines.append(f"- P/E Ratio: {td['pe_ratio']}")
            if "momentum_1m" in td:
                lines.append(f"- 1M Momentum: {td['momentum_1m'] * 100:+.1f}%")
            if "volatility_30d" in td:
                lines.append(f"- 30D Volatility: {td['volatility_30d'] * 100:.1f}%")

        return "\n".join(lines)

    def _initialize_knowledge_store(self):
        """Initialize the financial knowledge store with base knowledge."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Base financial knowledge
            knowledge_items = [
                {
                    "text": "When the Federal Reserve raises interest rates, it typically slows economic growth and can lead to lower stock valuations, especially for growth stocks.",
                    "category": "economic_principles", "source": "monetary_policy"},
                {
                    "text": "High inflation erodes purchasing power and can lead the Fed to tighten monetary policy. Sectors like utilities and consumer staples often outperform during inflationary periods.",
                    "category": "economic_principles", "source": "inflation"},
                {
                    "text": "The VIX index measures market volatility expectations. VIX above 30 indicates high fear, while below 15 suggests complacency.",
                    "category": "financial_concepts", "source": "volatility"},
                {
                    "text": "P/E ratio measures how much investors pay per dollar of earnings. High P/E suggests growth expectations or overvaluation, low P/E may indicate value or problems.",
                    "category": "financial_concepts", "source": "valuation"},
                {
                    "text": "Momentum investing relies on the tendency for winning stocks to continue winning. Strong 1-month and 3-month returns often predict near-term outperformance.",
                    "category": "investment_strategies", "source": "momentum"},
                {
                    "text": "Mean reversion suggests extreme price moves tend to reverse. Stocks far below their 52-week high may be oversold, while those at highs may be overbought.",
                    "category": "investment_strategies", "source": "mean_reversion"},
                {
                    "text": "Technology sector is sensitive to interest rates due to high growth expectations. Rising rates increase discount rates, reducing present value of future earnings.",
                    "category": "sector_knowledge", "source": "technology"},
                {
                    "text": "Financial sector often benefits from rising interest rates as banks earn more on loans. However, credit quality concerns can offset this benefit.",
                    "category": "sector_knowledge", "source": "financials"},
                {
                    "text": "Yield curve inversion (short-term rates above long-term) has historically preceded recessions within 12-18 months.",
                    "category": "economic_principles", "source": "yield_curve"},
                {
                    "text": "Earnings surprises drive short-term stock moves. Positive surprises lead to price increases, while negative surprises cause declines.",
                    "category": "financial_concepts", "source": "earnings"},
            ]

            # Embed and store
            self.knowledge_store = []
            for item in knowledge_items:
                embedding = self._embedding_model.encode(item["text"])
                self.knowledge_store.append({
                    "text": item["text"],
                    "category": item["category"],
                    "source": item["source"],
                    "embedding": embedding
                })

            logger.info(f"Initialized knowledge store with {len(self.knowledge_store)} items")

        except ImportError:
            logger.error("sentence-transformers not installed")
            self.knowledge_store = []
        except Exception as e:
            logger.error(f"Error initializing knowledge store: {e}")
            self.knowledge_store = []

    def _initialize_historical_events(self):
        """Initialize historical market events database."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Historical market events
            events = [
                {
                    "event": "2008 Financial Crisis - Lehman Brothers collapse",
                    "date": "September 2008",
                    "market_impact": "S&P 500 fell 57% from peak. Credit markets froze. Global recession.",
                    "lessons": "Systemic risk from interconnected financial institutions. Importance of liquidity."
                },
                {
                    "event": "COVID-19 Market Crash",
                    "date": "March 2020",
                    "market_impact": "S&P 500 fell 34% in 23 days. Fastest bear market in history. V-shaped recovery.",
                    "lessons": "Unprecedented Fed intervention can support markets. Technology stocks resilient."
                },
                {
                    "event": "Dot-com Bubble Burst",
                    "date": "2000-2002",
                    "market_impact": "NASDAQ fell 78%. Many tech companies went bankrupt. Multi-year bear market.",
                    "lessons": "Valuations matter eventually. Profitable companies outperform during downturns."
                },
                {
                    "event": "Fed Rate Hike Cycle 2022",
                    "date": "2022",
                    "market_impact": "S&P 500 fell 25%. Growth stocks hit hardest. Bonds also declined.",
                    "lessons": "Rising rates hurt growth stocks. Duration risk in bonds. Cash becomes attractive."
                },
                {
                    "event": "Black Monday 1987",
                    "date": "October 1987",
                    "market_impact": "Dow fell 22.6% in one day. Triggered by program trading and portfolio insurance.",
                    "lessons": "Market structure matters. Circuit breakers implemented afterward."
                },
                {
                    "event": "European Debt Crisis",
                    "date": "2011-2012",
                    "market_impact": "Global markets volatile. Flight to safety in US Treasuries.",
                    "lessons": "Sovereign debt risks can spread. Contagion effects across markets."
                },
            ]

            # Embed events
            self._historical_events = []
            for event in events:
                text = f"{event['event']} {event['market_impact']} {event['lessons']}"
                embedding = self._embedding_model.encode(text)
                self._historical_events.append({
                    **event,
                    "embedding": embedding
                })

            logger.info(f"Initialized {len(self._historical_events)} historical events")

        except ImportError:
            logger.error("sentence-transformers not installed")
            self._historical_events = []
        except Exception as e:
            logger.error(f"Error initializing historical events: {e}")
            self._historical_events = []
