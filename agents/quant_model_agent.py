"""
Quant Modeling Agent - Agent 3 in the Quant Trading System.

Responsibilities:
- Read model design document
- Write Python code to implement the model (ONCE)
- Reuse the code with new data for training
- Generate predictions with confidence scores

Key Design:
- Agent writes code based on design document
- Code is saved and reused (no LLM calls after first time)
- Only data changes between training runs
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os
import logging
import re

logger = logging.getLogger(__name__)

from .base_agent import BaseAgent, AgentOutput
from llm.llm_client import Message


@dataclass
class ModelPrediction:
    """Prediction output from the quant model."""
    ticker: str
    expected_return: float  # Predicted return (e.g., 0.023 = +2.3%)
    confidence: float  # 0.0 to 1.0
    horizon: str  # e.g., "5-day"
    regime: str  # Current regime: "Bull", "Bear", "Sideways"
    regime_probabilities: Dict[str, float]


class QuantModelingAgent(BaseAgent):
    """
    Agent that writes and reuses training code.

    Workflow:
    1. First time: Read design doc → Write code → Save code → Execute
    2. Subsequent: Load saved code → Execute with new data

    This makes training fast after the initial code generation.
    """

    def __init__(self, llm_client, config):
        super().__init__("QuantModelingAgent", llm_client, config)
        self.model = None
        self.model_path = config.model.model_path

        # Support multiple model designs
        self.design_docs = {
            "hmm": config.model.design_doc_path,  # Original HMM design
            "xgboost": getattr(config.model, 'xgboost_design_doc_path',
                               config.model.design_doc_path.replace('model_design.md', 'model_design_xgboost.md'))
        }

        # Default model type (can be overridden in run())
        self.default_model_type = getattr(config.model, 'model_type', 'hmm')

        # Create models directory if not exists
        os.makedirs(self.model_path, exist_ok=True)

    def _get_training_code_path(self, model_type: str) -> str:
        """Get training code path for specific model type."""
        return os.path.join(self.model_path, f"train_{model_type}.py")

    def _get_model_file_path(self, ticker: str, model_type: str) -> str:
        """Get model file path for specific ticker and model type."""
        return os.path.join(self.model_path, f"{ticker}_{model_type}.joblib")

    @property
    def system_prompt(self) -> str:
        return """You are a Quantitative Modeling Agent for a trading system.

Your role is to:
1. Read and understand model design documents
2. Write Python code to implement the specified model
3. The code should be reusable - same code, different data each time

When writing training code:
- Accept data as input variables (returns, ticker, model_path, n_states)
- Train the model on provided data
- Save trained model to model_path
- Return key metrics (log_likelihood, emission_means, etc.)
- Handle errors gracefully
- Use available imports: numpy (np), pandas (pd), joblib, hmmlearn.hmm

Be concise and write production-quality code."""

    def _register_tools(self):
        """Register quant modeling tools."""

        self.register_tool(
            name="read_document",
            func=self.read_document,
            description="Read model design document",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to design document"}
                },
                "required": ["path"]
            }
        )

        self.register_tool(
            name="execute_python",
            func=self.execute_python,
            description="Execute Python code and return output or errors",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        )

    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Run quant modeling task.

        Modes:
        - "train": Train models for each ticker
        - "predict": Generate predictions using trained models

        Args:
            input_data: {
                "mode": "train" or "predict",
                "price_data": Dict[str, pd.DataFrame],  # ticker -> price DataFrame
                "model_type": "hmm" or "xgboost" (optional, defaults to config)
                "features": Dict[str, pd.DataFrame]  # for XGBoost: ticker -> features DataFrame
            }

        Returns:
            AgentOutput with training results or predictions
        """
        mode = input_data.get("mode", "predict")
        price_data = input_data.get("price_data", {})
        model_type = input_data.get("model_type", self.default_model_type)
        features_data = input_data.get("features", {})  # For XGBoost

        logs = []
        logs.append(f"Using model type: {model_type}")

        if mode == "train":
            training_results = {}

            for ticker, prices_df in price_data.items():
                logger.info(f"Training {model_type} model for {ticker}...")
                logs.append(f"Training {model_type} model for {ticker}...")

                # Extract close prices
                if isinstance(prices_df, pd.DataFrame) and 'Close' in prices_df.columns:
                    prices = prices_df['Close']
                else:
                    prices = prices_df

                # Get features if available (for XGBoost)
                features = features_data.get(ticker, None)

                result = self.train_model(ticker, prices, model_type=model_type, features=features)

                if result.success:
                    training_results[ticker] = result.data
                    logs.extend(result.logs)
                else:
                    logs.append(f"ERROR training {ticker}: {result.message}")

            return AgentOutput(
                success=True,
                data={"training_results": training_results, "model_type": model_type},
                message=f"Trained {model_type} models for {len(training_results)}/{len(price_data)} tickers",
                logs=logs
            )

        elif mode == "predict":
            predictions = {}

            for ticker, prices_df in price_data.items():
                logger.info(f"Generating {model_type} prediction for {ticker}...")
                logs.append(f"Generating {model_type} prediction for {ticker}...")

                # Extract close prices
                if isinstance(prices_df, pd.DataFrame) and 'Close' in prices_df.columns:
                    prices = prices_df['Close']
                else:
                    prices = prices_df

                # Get features if available
                features = features_data.get(ticker, None)

                # Load model
                model_path = self._get_model_file_path(ticker, model_type)
                self.load_model(model_path)

                # Generate prediction based on model type
                if model_type == "hmm":
                    returns = prices.pct_change(periods=5).dropna().values.reshape(-1, 1)
                    prediction = self.predict_hmm(ticker, returns)
                elif model_type == "xgboost":
                    prediction = self.predict_xgboost(ticker, prices, features)
                else:
                    logger.error(f"Unknown model type: {model_type}")
                    continue

                predictions[ticker] = prediction
                logs.append(f"  {ticker}: {prediction}")

            return AgentOutput(
                success=True,
                data={"predictions": predictions, "model_type": model_type},
                message=f"Generated {model_type} predictions for {len(price_data)} tickers",
                logs=logs
            )

        else:
            return AgentOutput(
                success=False,
                data={},
                message=f"Unknown mode: {mode}. Use 'train' or 'predict'.",
                logs=logs
            )

    def train_model(self, ticker: str, prices: pd.Series, model_type: str = "hmm",
                    n_states: int = 3, features: pd.DataFrame = None) -> AgentOutput:
        """
        Train model using agent-written code (reused if exists).

        Args:
            ticker: Stock ticker
            prices: Price series
            model_type: "hmm" or "xgboost"
            n_states: Number of HMM states (for HMM only)
            features: Feature DataFrame (for XGBoost only)

        Returns:
            AgentOutput with training results
        """
        logs = []
        training_code_path = self._get_training_code_path(model_type)
        design_doc_path = self.design_docs.get(model_type, self.design_docs["hmm"])

        # Check if training code already exists
        if os.path.exists(training_code_path):
            logger.info(f"Reusing {model_type} training code from {training_code_path}")
            logs.append(f"Reusing existing {model_type} training code")

            with open(training_code_path, 'r') as f:
                training_code = f.read()
        else:
            # FIRST TIME: Agent writes the code
            logger.info(f"No {model_type} training code found. Agent will write code based on design doc.")
            logs.append(f"Writing new {model_type} training code based on design document...")

            # Read design document
            design_doc = self.read_document(design_doc_path)

            if not design_doc:
                return AgentOutput(
                    success=False,
                    data={},
                    message=f"Could not read design doc: {design_doc_path}",
                    logs=logs
                )

            # Generate appropriate prompt based on model type
            if model_type == "hmm":
                prompt = self._get_hmm_training_prompt(design_doc)
            elif model_type == "xgboost":
                prompt = self._get_xgboost_training_prompt(design_doc)
            else:
                return AgentOutput(
                    success=False,
                    data={},
                    message=f"Unknown model type: {model_type}",
                    logs=logs
                )

            messages = [
                Message("system", self.system_prompt),
                Message("user", prompt)
            ]

            response = self.llm_client.chat(messages)
            training_code = self._extract_code(response.content)

            if not training_code:
                return AgentOutput(
                    success=False,
                    data={},
                    message="Failed to extract code from LLM response",
                    logs=logs + [f"LLM response: {response.content}"]
                )

            # Save code for future reuse
            with open(training_code_path, 'w') as f:
                f.write(training_code)
            logger.info(f"Saved {model_type} training code to {training_code_path}")
            logs.append(f"Saved training code to {training_code_path}")

        # Execute the code
        try:
            model_file_path = self._get_model_file_path(ticker, model_type)

            # Prepare variables based on model type
            if model_type == "hmm":
                returns = prices.pct_change(periods=5).dropna().values.reshape(-1, 1)
                variables = {
                    'returns': returns,
                    'ticker': ticker,
                    'model_path': model_file_path,
                    'n_states': n_states
                }
            elif model_type == "xgboost":
                # Prepare data for XGBoost
                returns = prices.pct_change().dropna()
                # Create target (next day direction)
                target = (prices.shift(-1) > prices).astype(int).dropna()

                variables = {
                    'prices': prices,
                    'returns': returns.values,
                    'target': target.values[:-1] if len(target) > len(returns) else target.values,
                    'features': features,
                    'ticker': ticker,
                    'model_path': model_file_path
                }

            exec_result = self.execute_python(training_code, variables=variables)

            # DEBUG: Log execution details
            logger.info(f"Training exec success: {exec_result['success']}")
            if exec_result.get('error'):
                logger.error(f"Training error: {exec_result['error']}")

            if not exec_result['success']:
                return AgentOutput(
                    success=False,
                    data={},
                    message=f"Code execution failed: {exec_result['error']}",
                    logs=logs + [f"Error: {exec_result['error']}"]
                )

            result = exec_result['namespace'].get('result', {})

            if not result:
                return AgentOutput(
                    success=False,
                    data={},
                    message="Training code did not produce 'result' dictionary",
                    logs=logs + [f"Output: {exec_result['output']}"]
                )

            result['model_path'] = model_file_path
            result['model_type'] = model_type
            logs.append(f"Successfully trained {model_type} for {ticker}")

            return AgentOutput(
                success=True,
                data=result,
                message=f"Successfully trained {model_type} for {ticker}",
                logs=logs
            )

        except Exception as e:
            return AgentOutput(
                success=False,
                data={},
                message=f"Training failed: {str(e)}",
                logs=logs + [f"Exception: {str(e)}"]
            )

    def _get_hmm_training_prompt(self, design_doc: str) -> str:
        """Generate prompt for HMM training code."""
        return f"""Based on this model design document:
    {design_doc}

    Write Python code to train a Hidden Markov Model.

    CRITICAL REQUIREMENTS:
    1. Write DIRECTLY EXECUTABLE code - DO NOT define a function
    2. These variables are ALREADY DEFINED and available:
       - returns: numpy array, shape (n_samples, 1) - ALREADY numpy, DO NOT call .values on it
       - ticker: string
       - model_path: string  
       - n_states: int
    3. At the END of your code, you MUST create a variable called 'result' (not inside a function)

    The code should:
    1. Train a Gaussian HMM with n_states states on the 'returns' data
    2. Save the trained model to model_path using joblib.dump()
    3. Create a 'result' dictionary with these keys:
        - 'model_type': str ('GaussianHMM')
        - 'n_states': int
        - 'log_likelihood': float (use model.score(returns))
        - 'emission_means': dict mapping regime names to mean returns
        - 'n_samples': int (length of returns)

    Available imports (already imported): np, pd, joblib, hmm (from hmmlearn)

    Write ONLY executable Python code. NO function definitions. NO explanations."""

    def _get_xgboost_training_prompt(self, design_doc: str) -> str:
        """Generate prompt for XGBoost training code."""
        return f"""Based on this model design document:
    {design_doc}

    Write Python code to train an XGBoost classifier for stock direction prediction.

    CRITICAL REQUIREMENTS:
    1. Write DIRECTLY EXECUTABLE code - NO function definitions
    2. These variables are ALREADY DEFINED:
       - prices: pandas SERIES (not DataFrame!) of close prices with DatetimeIndex
       - ticker: string
       - model_path: string
    3. At the END, create a variable called 'result'

    EXACT CODE STRUCTURE TO FOLLOW:

    1. Create features DataFrame with prices.index:
       feature_df = pd.DataFrame(index=prices.index)

    2. Calculate RSI (14-day):
       delta = prices.diff()
       gain = delta.copy()
       loss = delta.copy()
       gain[gain < 0] = 0
       loss[loss > 0] = 0
       avg_gain = gain.rolling(window=14).mean()
       avg_loss = abs(loss).rolling(window=14).mean()
       rs = avg_gain / avg_loss
       feature_df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    3. Calculate MACD:
       ema_12 = prices.ewm(span=12, adjust=False).mean()
       ema_26 = prices.ewm(span=26, adjust=False).mean()
       feature_df['macd'] = ema_12 - ema_26

    4. Calculate SMA ratios (5, 10, 20, 50):
       feature_df['sma_5'] = prices / prices.rolling(window=5).mean()
       (similar for 10, 20, 50)

    5. Calculate Momentum (5-day, 10-day):
       feature_df['momentum_5'] = prices.pct_change(5)
       feature_df['momentum_10'] = prices.pct_change(10)

    6. Calculate Volatility:
       feature_df['volatility'] = prices.pct_change().rolling(window=14).std()

    7. Add embedding factor placeholders (for news sentiment during prediction):
       feature_df['news_sentiment'] = 0.0
       feature_df['news_magnitude'] = 0.0
       feature_df['news_novelty'] = 0.5

    8. Create target and ALIGN with features:
       target_series = (prices.shift(-1) > prices).astype(int)
       feature_df['target'] = target_series
       feature_df = feature_df.dropna()  # Drop NaN ONCE after combining

    9. Split, scale, train XGBoost

    10. Save as dict: joblib.dump({{'model': model, 'scaler': scaler, 'feature_names': list(X.columns)}}, model_path)

    11. Create result dict with: model_type, accuracy, n_features, feature_names, n_samples

    Available imports: np, pd, joblib, xgb, StandardScaler, train_test_split

    Write ONLY the executable Python code. NO markdown. NO explanations."""

    def predict_hmm(self, ticker: str, returns: np.ndarray) -> Dict[str, Any]:
        """
        Generate prediction using trained HMM model.
        """
        if self.model is None:
            model_path = self._get_model_file_path(ticker, "hmm")
            if not self.load_model(model_path):
                return {
                    "expected_return": 0.0,
                    "confidence": 0.0,
                    "regime": "Unknown",
                    "regime_probabilities": {},
                    "model_type": "hmm"
                }

        try:
            if returns.ndim == 1:
                returns = returns.reshape(-1, 1)

            regime_probs = self.model.predict_proba(returns)[-1]
            regime_names = ["Bull", "Bear", "Sideways"]
            current_regime_idx = np.argmax(regime_probs)
            current_regime = regime_names[current_regime_idx]

            means = self.model.means_.flatten()
            expected_return = float(np.dot(regime_probs, means))
            confidence = float(np.max(regime_probs))

            regime_probabilities = {
                regime_names[i]: float(regime_probs[i])
                for i in range(len(regime_names))
            }

            return {
                "expected_return": expected_return,
                "confidence": confidence,
                "regime": current_regime,
                "regime_probabilities": regime_probabilities,
                "model_type": "hmm"
            }

        except Exception as e:
            logger.error(f"Error in HMM prediction for {ticker}: {e}")
            return {
                "expected_return": 0.0,
                "confidence": 0.0,
                "regime": "Unknown",
                "regime_probabilities": {},
                "model_type": "hmm"
            }

    def predict_xgboost(self, ticker: str, prices: pd.Series, features: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate prediction using trained XGBoost model.

        Args:
            ticker: Stock ticker
            prices: Price series
            features: Dict of feature values including embedding factors (news_sentiment, news_magnitude, news_novelty)
        """
        if self.model is None:
            model_path = self._get_model_file_path(ticker, "xgboost")
            if not self.load_model(model_path):
                return {
                    "direction_probability": 0.5,
                    "confidence": 0.0,
                    "predicted_direction": "NEUTRAL",
                    "model_type": "xgboost"
                }

        try:
            # Extract model and scaler from saved data
            if isinstance(self.model, dict):
                model = self.model.get('model')
                scaler = self.model.get('scaler')
            else:
                model = self.model
                scaler = None

            # Extract embedding factors from features dict
            embedding_factors = None
            if features is not None:
                embedding_factors = {
                    "news_sentiment": features.get("news_sentiment", 0.0),
                    "news_magnitude": features.get("news_magnitude", 0.0),
                    "news_novelty": features.get("news_novelty", 0.5)
                }

            # Compute features from prices + embedding factors
            X = self._compute_basic_features(prices, embedding_factors)

            # Scale if scaler available
            if scaler is not None:
                X = scaler.transform(X)

            # Get prediction probability
            prob = model.predict_proba(X)[0]
            up_prob = prob[1] if len(prob) > 1 else prob[0]

            # Determine direction
            if up_prob > 0.55:
                direction = "UP"
            elif up_prob < 0.45:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            confidence = abs(up_prob - 0.5) * 2  # Scale to 0-1

            return {
                "direction_probability": float(up_prob),
                "confidence": float(confidence),
                "predicted_direction": direction,
                "model_type": "xgboost"
            }

        except Exception as e:
            logger.error(f"Error in XGBoost prediction for {ticker}: {e}")
            return {
                "direction_probability": 0.5,
                "confidence": 0.0,
                "predicted_direction": "NEUTRAL",
                "model_type": "xgboost"
            }


    def _compute_basic_features(self, prices: pd.Series, embedding_factors: Dict[str, float] = None) -> np.ndarray:
        """Compute technical features + embedding factors (12 features total)."""

        # Original 9 price-based features
        rsi_val = (prices.iloc[-1] - prices.tail(14).mean()) / prices.tail(14).std() if prices.tail(
            14).std() != 0 else 0
        macd_val = prices.tail(26).mean() - prices.tail(12).mean()
        sma_5 = prices.iloc[-1] / prices.tail(5).mean() if prices.tail(5).mean() != 0 else 1
        sma_10 = prices.iloc[-1] / prices.tail(10).mean() if prices.tail(10).mean() != 0 else 1
        sma_20 = prices.iloc[-1] / prices.tail(20).mean() if prices.tail(20).mean() != 0 else 1
        sma_50 = prices.iloc[-1] / prices.tail(50).mean() if prices.tail(50).mean() != 0 else 1
        momentum_5 = prices.pct_change(5).iloc[-1] if len(prices) > 5 else 0
        momentum_10 = prices.pct_change(10).iloc[-1] if len(prices) > 10 else 0
        volatility = prices.pct_change().tail(14).std() if len(prices) > 14 else 0

        # 3 embedding-based factors
        news_sentiment = 0.0
        news_magnitude = 0.0
        news_novelty = 0.5
        if embedding_factors:
            news_sentiment = embedding_factors.get("news_sentiment", 0.0)
            news_magnitude = embedding_factors.get("news_magnitude", 0.0)
            news_novelty = embedding_factors.get("news_novelty", 0.5)

        features = np.array([[
            rsi_val, macd_val, sma_5, sma_10, sma_20, sma_50,
            momentum_5, momentum_10, volatility,
            news_sentiment, news_magnitude, news_novelty  # NEW
        ]])

        features = np.nan_to_num(features, nan=0.0)
        return features

    # ==================== Tool Implementations ====================

    def read_document(self, path: str) -> str:
        """
        Read model design document.

        Args:
            path: Path to document (YAML, MD, or TXT)

        Returns:
            Document contents as string
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Document not found: {path}")
                return ""

            with open(path, 'r') as f:
                content = f.read()

            logger.info(f"Read document: {path} ({len(content)} chars)")
            return content

        except Exception as e:
            logger.error(f"Error reading document {path}: {e}")
            return ""

    def execute_python(self, code: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Python code with provided variables.

        Args:
            code: Python code to execute
            variables: Dict of variables to inject into execution namespace

        Returns:
            Dict with success, output, error, namespace
        """
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        try:
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Create execution namespace with common imports
            exec_namespace = {
                "pd": pd,
                "np": np,
                "datetime": datetime,
                "joblib": joblib,
                "__builtins__": __builtins__,
            }

            # Add hmmlearn
            try:
                from hmmlearn import hmm
                exec_namespace["hmm"] = hmm
            except ImportError:
                pass

            # Add xgboost and sklearn
            try:
                import xgboost as xgb
                exec_namespace["xgb"] = xgb
            except ImportError:
                pass

            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                exec_namespace["StandardScaler"] = StandardScaler
                exec_namespace["train_test_split"] = train_test_split
            except ImportError:
                pass

            # Inject provided variables
            if variables:
                exec_namespace.update(variables)

            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_namespace)

            return {
                "success": True,
                "output": stdout_capture.getvalue(),
                "error": None,
                "namespace": exec_namespace
            }

        except Exception as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue() if 'stdout_capture' in locals() else "",
                "error": str(e),
                "namespace": {}
            }

    def load_model(self, path: str) -> Any:
        """
        Load trained model from disk.
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Model file not found: {path}")
                return None

            model_data = joblib.load(path)

            # Handle different save formats
            if isinstance(model_data, dict) and "model" in model_data:
                # Dict format: {'model': model, 'scaler': scaler, ...}
                self.model = model_data
                logger.info(f"Loaded model (dict format) from {path}")
            elif isinstance(model_data, tuple) and len(model_data) == 2:
                # Tuple format from LLM-generated code: (model, scaler)
                self.model = {
                    'model': model_data[0],
                    'scaler': model_data[1],
                    'feature_names': []
                }
                logger.info(f"Loaded model (tuple format) from {path}")
            else:
                # HMM or other format - store directly
                self.model = model_data
                logger.info(f"Loaded model from {path}")

            return self.model

        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            return None



    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from LLM response.

        Handles responses with or without markdown code blocks.

        Args:
            text: LLM response text

        Returns:
            Extracted Python code
        """
        # Try to extract code from markdown blocks
        code_block_pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            # Return first code block
            return matches[0].strip()

        # No code blocks found, assume entire text is code
        return text.strip()

