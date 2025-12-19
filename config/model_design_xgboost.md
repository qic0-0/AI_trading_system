# XGBoost Factor Model Design

## Reference Implementation
**Source:** https://github.com/SahuH/Stock-prediction-XGBoost

The Quant Agent should read and adapt this implementation for our system.

---

## Model Overview

| Aspect | Description |
|--------|-------------|
| **Model Type** | XGBoost Classifier |
| **Task** | Predict next-day price direction (UP/DOWN) |
| **Output** | Probability of price increase |

---

## Input Features

### 1. Technical Indicators (from Feature Agent)

The model uses common quant factors. The agent should compute these from price data:

| Indicator | Description | Typical Parameters |
|-----------|-------------|-------------------|
| **RSI** | Relative Strength Index | 14-day period |
| **MACD** | Moving Average Convergence Divergence | 12, 26, 9 |
| **Bollinger Bands** | Volatility bands | 20-day, 2 std |
| **SMA** | Simple Moving Average | 5, 10, 20, 50 days |
| **EMA** | Exponential Moving Average | 12, 26 days |
| **ROC** | Rate of Change | 10-day |
| **Stochastic Oscillator** | %K and %D | 14, 3 |
| **ATR** | Average True Range | 14-day |
| **OBV** | On-Balance Volume | cumulative |
| **Momentum** | Price momentum | 10-day |

### 2. Text Embeddings (NEW - from Feature Agent)

In addition to technical indicators, include text embeddings from news/sentiment:

```
embedding_vector = feature_agent.embed_text(news_text)
# Returns: numpy array of shape (embedding_dim,)
```

Concatenate embedding vector to feature vector before training.

---

## Target Variable

```python
# Binary classification target
target = 1 if next_day_close > today_close else 0
```

---

## Training Process

```
1. Load price data (from Data Agent)
2. Compute technical indicators
3. Get text embeddings for each day's news (if available)
4. Create feature matrix X and target vector y
5. Train/test split (e.g., 80/20 chronological)
6. Standardize features (StandardScaler)
7. Train XGBoost classifier
8. Save model
```

---

## Model Parameters (Starting Point)

```python
params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

---

## Prediction Output

```python
def predict(features):
    """
    Args:
        features: array of [technical_indicators + text_embedding]
    
    Returns:
        probability: float between 0 and 1
            - > 0.5: predicted UP (bullish signal)
            - < 0.5: predicted DOWN (bearish signal)
        confidence: abs(probability - 0.5) * 2
    """
    prob = model.predict_proba(features)[0][1]
    confidence = abs(prob - 0.5) * 2
    return prob, confidence
```

---

## Integration with Trading System

The Coordinator Agent combines:
1. **HMM Model**: Regime detection (Bull/Bear/Sideways)
2. **XGBoost Model**: Direction prediction with confidence
3. **Market-Sense Agent**: LLM-based market analysis

Decision logic:
```
IF xgboost_prob > 0.6 AND hmm_regime == "Bull" AND market_sense == "Bullish":
    signal = STRONG_BUY
ELIF xgboost_prob < 0.4 AND hmm_regime == "Bear" AND market_sense == "Bearish":
    signal = STRONG_SELL
ELSE:
    signal = HOLD or weak signal
```

---

## Feature Importance

After training, extract and log feature importance:
```python
importance = model.feature_importances_
# Use this to understand which factors drive predictions
```

---

## Notes for Quant Agent

1. **Read the GitHub repo first** - understand the approach
2. **Adapt, don't copy** - modify for our data structure
3. **Add text embeddings** - this is our extension beyond the reference
4. **Handle missing data** - some days may not have news
5. **Save the model** - to `models/xgboost_model.pkl`
