# HMM Model Design Document

## 1. Model Overview

This model uses a Gaussian Hidden Markov Model (HMM) to predict expected stock returns. Each stock is modeled independently with three hidden regimes: Bull, Bear, and Sideways.

**Key Idea:** The market alternates between hidden regimes. Each regime has a characteristic return distribution. By inferring which regime we're in, we can predict expected returns.

```
Hidden States x(t):     [Bull] ──→ [Bull] ──→ [Bear] ──→ [Sideways]
                          │          │          │           │
                          ▼          ▼          ▼           ▼
Observable y(t):        +1.8%      +1.2%      -2.1%       +0.2%
(5-day return)
```

---

## 2. Model Type

- **Algorithm:** Gaussian Hidden Markov Model (HMM)
- **Library:** hmmlearn (GaussianHMM)
- **Scope:** Per-stock (train one HMM per ticker)

---

## 3. Hidden States

The model has 3 hidden states representing market regimes:

| State | Interpretation | Expected Return | Typical Volatility |
|-------|----------------|-----------------|-------------------|
| Bull | Upward trending, optimism | Positive (μ > 0) | Low to medium |
| Bear | Downward trending, fear | Negative (μ < 0) | High |
| Sideways | No clear trend, uncertainty | Near zero (μ ≈ 0) | Medium |

The model learns to identify these states automatically from the data.

---

## 4. Observations

The observable variable y(t) is the 5-day stock return:

- **Definition:** y(t) = 5-day return
- **Formula:** `y(t) = (Price(t) - Price(t-5)) / Price(t-5)`
- **Why 5-day:** Less noisy than daily returns, but still responsive

---

## 5. Model Components

### 5.1 Transition Matrix (A)

Probability of switching between regimes:

```
A[i,j] = P(x(t+1) = j | x(t) = i)

Example:
           To: Bull  Bear  Sideways
From: Bull    0.85  0.05    0.10
      Bear    0.10  0.80    0.10
      Sideways 0.15  0.15    0.70
```

The model learns this matrix during training.

### 5.2 Emission Distribution (B)

Each state emits returns from a Gaussian distribution:

```
P(y(t) | x(t) = k) ~ N(μ_k, σ_k²)
```

| State | Mean (μ) | Variance (σ²) |
|-------|----------|---------------|
| Bull | μ_bull (learned, typically > 0) | σ_bull² (learned) |
| Bear | μ_bear (learned, typically < 0) | σ_bear² (learned) |
| Sideways | μ_sideways (learned, typically ≈ 0) | σ_sideways² (learned) |

### 5.3 Initial Distribution (π)

Starting probabilities for each regime:

```
π = [P(x(0) = Bull), P(x(0) = Bear), P(x(0) = Sideways)]
```

---

## 6. Training

### 6.1 Input Data

- **Source:** Feature Dictionary provides available features
- **Required:** Historical price data for each stock
- **Minimum:** 2 years of data (approximately 500 trading days)
- **Computed:** 5-day returns from price data

### 6.2 Training Process

1. Load price data from Feature Agent output
2. Calculate 5-day returns: `returns = prices.pct_change(periods=5).dropna()`
3. Reshape for hmmlearn: `returns_array = returns.values.reshape(-1, 1)`
4. Initialize and fit GaussianHMM with 3 components
5. Model learns: transition matrix A, emission means μ_k, emission variances σ_k²

### 6.3 Training Code Example

```python
from hmmlearn import hmm
import numpy as np
import pandas as pd

def train_hmm(prices: pd.Series, n_states: int = 3) -> hmm.GaussianHMM:
    """
    Train a Gaussian HMM on stock returns.
    
    Args:
        prices: Historical price series
        n_states: Number of hidden states (default: 3)
    
    Returns:
        Trained GaussianHMM model
    """
    # Calculate 5-day returns
    returns = prices.pct_change(periods=5).dropna()
    
    # Reshape for hmmlearn: (n_samples, n_features)
    returns_array = returns.values.reshape(-1, 1)
    
    # Initialize model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=42
    )
    
    # Fit model
    model.fit(returns_array)
    
    return model
```

---

## 7. Prediction

### 7.1 Process

1. Get recent 5-day return observation
2. Use model to infer current regime probabilities: `P(Bull), P(Bear), P(Sideways)`
3. Calculate expected return as weighted average of state means
4. Calculate confidence as maximum regime probability

### 7.2 Expected Return Formula

```
E[return] = P(Bull) × μ_bull + P(Bear) × μ_bear + P(Sideways) × μ_sideways
```

Where:
- `P(state)` = probability of being in that state (from model inference)
- `μ_state` = mean return for that state (learned during training)

### 7.3 Confidence Calculation

```
Confidence = max(P(Bull), P(Bear), P(Sideways))
```

Interpretation:
- Confidence near 1.0: Model is very certain about current regime
- Confidence near 0.33: Model is uncertain (equal probability across states)

### 7.4 Prediction Code Example

```python
def predict_return(model: hmm.GaussianHMM, recent_returns: np.ndarray) -> dict:
    """
    Predict expected return using trained HMM.
    
    Args:
        model: Trained GaussianHMM
        recent_returns: Recent 5-day returns, shape (n_recent, 1)
    
    Returns:
        Dictionary with prediction results
    """
    # Get regime probabilities for latest observation
    regime_probs = model.predict_proba(recent_returns)[-1]
    
    # Get current regime (highest probability)
    current_regime_idx = np.argmax(regime_probs)
    regime_names = ["Bull", "Bear", "Sideways"]
    current_regime = regime_names[current_regime_idx]
    
    # Get emission means from model
    means = model.means_.flatten()
    
    # Calculate expected return (weighted average)
    expected_return = np.dot(regime_probs, means)
    
    # Confidence is max probability
    confidence = np.max(regime_probs)
    
    return {
        "regime": {
            "current": current_regime,
            "probabilities": {
                "Bull": float(regime_probs[0]),
                "Bear": float(regime_probs[1]),
                "Sideways": float(regime_probs[2])
            }
        },
        "expected_return": float(expected_return),
        "confidence": float(confidence)
    }
```

---

## 8. Output Format

The model outputs a dictionary for each stock:

```python
{
    "ticker": "AAPL",
    "timestamp": "2024-06-01T16:00:00",
    "horizon": "5-day",
    "regime": {
        "current": "Bull",
        "probabilities": {
            "Bull": 0.72,
            "Bear": 0.18,
            "Sideways": 0.10
        }
    },
    "expected_return": 0.023,  # +2.3% expected over 5 days
    "confidence": 0.72
}
```

**Important:** This is a prediction only. The Coordinator Agent decides whether to BUY/SELL/HOLD.

---

## 9. Input Requirements

### From Feature Dictionary

The model requires these features:

| Feature | Source | Description |
|---------|--------|-------------|
| `close_price` | Price data | Adjusted closing price |
| `return_5d` | Computed | 5-day return (can be computed from close_price) |

### Data Format

```python
# Input: DataFrame with price data
prices_df = {
    "AAPL": pd.Series([...]),  # Price series
    "MSFT": pd.Series([...]),
    ...
}
```

---

## 10. Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Directional Accuracy** | % of times predicted direction matches actual | > 55% |
| **MSE** | Mean squared error of predicted vs actual returns | Lower is better |
| **Log-Likelihood** | Model fit quality (from hmmlearn) | Higher is better |

### Evaluation Code Example

```python
def evaluate_model(model, test_returns, actual_future_returns):
    """Evaluate HMM prediction accuracy."""
    
    predictions = []
    for i in range(len(test_returns) - 1):
        pred = predict_return(model, test_returns[:i+1])
        predictions.append(pred["expected_return"])
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actual_future_returns[:-1])
    directional_accuracy = np.mean(pred_direction == actual_direction)
    
    # MSE
    mse = np.mean((np.array(predictions) - actual_future_returns[:-1])**2)
    
    return {
        "directional_accuracy": directional_accuracy,
        "mse": mse,
        "log_likelihood": model.score(test_returns)
    }
```

---

## 11. Hyperparameters

| Parameter | Default | Tuning Range | Description |
|-----------|---------|--------------|-------------|
| `n_components` | 3 | [2, 3, 4] | Number of hidden states |
| `covariance_type` | "diag" | ["diag", "full"] | Covariance matrix type |
| `n_iter` | 100 | [50, 100, 200] | Max training iterations |
| `random_state` | 42 | - | For reproducibility |

### Tuning Strategy

1. Start with defaults (3 states, diag covariance)
2. If log-likelihood is low, try more iterations
3. If overfitting, reduce n_components to 2
4. If underfitting, try "full" covariance

---

## 12. Retraining Policy

| Trigger | Action |
|---------|--------|
| **Scheduled** | Retrain monthly |
| **Performance** | Retrain if directional accuracy < 50% |
| **Data** | Retrain when new quarter of data available |

---

## 13. Summary

| Aspect | Value |
|--------|-------|
| Model | Gaussian HMM |
| Hidden States | 3 (Bull, Bear, Sideways) |
| Observable | 5-day stock return |
| Scope | Per-stock |
| Output | Expected return + confidence |
| Decision | None (Coordinator Agent decides) |
