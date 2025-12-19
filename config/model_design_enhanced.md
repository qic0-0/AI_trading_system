# HMM Model Design Document (Enhanced with Fourier Seasonality)

## 1. Model Overview

This model uses a **Multi-Dimensional Gaussian Hidden Markov Model (HMM)** enhanced with **Fourier series features** to capture market seasonality. Each stock is modeled independently with three hidden regimes: Bull, Bear, and Sideways.

**Key Innovation:** We augment the traditional HMM observation (returns) with Fourier terms that capture periodic patterns (weekly, monthly, quarterly, yearly), inspired by Facebook Prophet's approach to seasonality modeling.

```
Hidden States x(t):     [Bull] ──→ [Bull] ──→ [Bear] ──→ [Sideways]
                          │          │          │           │
                          ▼          ▼          ▼           ▼
Observable y(t):      [return]   [return]   [return]    [return]
(Multi-dim vector)    [sin_w]    [sin_w]    [sin_w]     [sin_w]
                      [cos_w]    [cos_w]    [cos_w]     [cos_w]
                      [sin_m]    [sin_m]    [sin_m]     [sin_m]
                      [cos_m]    [cos_m]    [cos_m]     [cos_m]
                        ...        ...        ...         ...
```

---

## 2. Model Type

- **Algorithm:** Multi-Dimensional Gaussian Hidden Markov Model (HMM)
- **Library:** hmmlearn (GaussianHMM)
- **Enhancement:** Fourier series features for seasonality (inspired by Facebook Prophet)
- **Scope:** Per-stock (train one HMM per ticker)

---

## 3. Hidden States

The model has 3 hidden states representing market regimes:

| State | Interpretation | Expected Return | Typical Volatility |
|-------|----------------|-----------------|-------------------|
| Bull | Upward trending, optimism | Positive (μ > 0) | Low to medium |
| Bear | Downward trending, fear | Negative (μ < 0) | High |
| Sideways | No clear trend, uncertainty | Near zero (μ ≈ 0) | Medium |

The model learns to identify these states automatically from the multi-dimensional observations.

---

## 4. Observations (Enhanced)

### 4.1 Original Observable
The base observable variable is the 5-day stock return:
- **Definition:** return(t) = 5-day return
- **Formula:** `return(t) = (Price(t) - Price(t-5)) / Price(t-5)`

### 4.2 Fourier Series Features (NEW)

We add Fourier terms to capture periodic seasonality patterns. For a given period P and Fourier order K:

```
For k = 1, 2, ..., K:
    sin_k(t) = sin(2πk × t / P)
    cos_k(t) = cos(2πk × t / P)
```

Where:
- t = time index (trading day number)
- P = period length in trading days
- K = Fourier order (number of harmonic terms)

### 4.3 Seasonality Periods for Stocks

| Seasonality | Period (Trading Days) | Fourier Order K | Features Added |
|-------------|----------------------|-----------------|----------------|
| Weekly | 5 | 2 | 4 (2 sin + 2 cos) |
| Monthly | 21 | 3 | 6 (3 sin + 3 cos) |
| Quarterly | 63 | 2 | 4 (2 sin + 2 cos) |
| Yearly | 252 | 4 | 8 (4 sin + 4 cos) |

**Total observation dimension:** 1 (return) + 4 + 6 + 4 + 8 = **23 features**

### 4.4 Why Fourier Terms?

1. **Capture Day-of-Week Effects:** Monday/Friday anomalies
2. **Month-End Effects:** Portfolio rebalancing, window dressing
3. **Quarterly Patterns:** Earnings seasons, dividend dates
4. **Yearly Seasonality:** Tax-loss harvesting, January effect, "Sell in May"

### 4.5 Complete Observation Vector

```python
y(t) = [
    return_5d,           # Base feature: 5-day return
    
    # Weekly seasonality (K=2)
    sin(2π × 1 × t / 5),
    cos(2π × 1 × t / 5),
    sin(2π × 2 × t / 5),
    cos(2π × 2 × t / 5),
    
    # Monthly seasonality (K=3)
    sin(2π × 1 × t / 21),
    cos(2π × 1 × t / 21),
    sin(2π × 2 × t / 21),
    cos(2π × 2 × t / 21),
    sin(2π × 3 × t / 21),
    cos(2π × 3 × t / 21),
    
    # Quarterly seasonality (K=2)
    sin(2π × 1 × t / 63),
    cos(2π × 1 × t / 63),
    sin(2π × 2 × t / 63),
    cos(2π × 2 × t / 63),
    
    # Yearly seasonality (K=4)
    sin(2π × 1 × t / 252),
    cos(2π × 1 × t / 252),
    sin(2π × 2 × t / 252),
    cos(2π × 2 × t / 252),
    sin(2π × 3 × t / 252),
    cos(2π × 3 × t / 252),
    sin(2π × 4 × t / 252),
    cos(2π × 4 × t / 252),
]
```

---

## 5. Model Components

### 5.1 Transition Matrix (A)

Probability of switching between regimes (same as before):

```
A[i,j] = P(x(t+1) = j | x(t) = i)

Example:
           To: Bull  Bear  Sideways
From: Bull    0.85  0.05    0.10
      Bear    0.10  0.80    0.10
      Sideways 0.15  0.15    0.70
```

### 5.2 Emission Distribution (B) - ENHANCED

Each state emits **multi-dimensional** observations from a Gaussian distribution:

```
P(y(t) | x(t) = k) ~ N(μ_k, Σ_k)
```

Where:
- `μ_k` = mean vector of dimension 23 (1 return + 22 Fourier features)
- `Σ_k` = covariance matrix (23 × 23) - using "diag" for efficiency

| State | Return Mean | Fourier Means |
|-------|-------------|---------------|
| Bull | μ_bull > 0 | Learned seasonality pattern for bull markets |
| Bear | μ_bear < 0 | Learned seasonality pattern for bear markets |
| Sideways | μ_sideways ≈ 0 | Learned seasonality pattern for sideways markets |

**Key Insight:** Different regimes may have different seasonal patterns! For example:
- Bull markets might show stronger January effect
- Bear markets might have more pronounced Monday effects

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
- **Computed:** 5-day returns + Fourier features from price data

### 6.2 Feature Generation Code

```python
import numpy as np
import pandas as pd

def generate_fourier_features(t: np.ndarray, periods: dict) -> np.ndarray:
    """
    Generate Fourier series features for seasonality.
    
    Args:
        t: Time index array (0, 1, 2, ..., n_samples)
        periods: Dict mapping period name to (period_length, fourier_order)
                 e.g., {'weekly': (5, 2), 'monthly': (21, 3)}
    
    Returns:
        Array of shape (n_samples, n_fourier_features)
    """
    features = []
    
    for name, (period, order) in periods.items():
        for k in range(1, order + 1):
            # Sine term
            sin_term = np.sin(2 * np.pi * k * t / period)
            features.append(sin_term)
            
            # Cosine term
            cos_term = np.cos(2 * np.pi * k * t / period)
            features.append(cos_term)
    
    return np.column_stack(features)


def prepare_observations(prices: pd.Series) -> np.ndarray:
    """
    Prepare multi-dimensional observation vector for HMM.
    
    Args:
        prices: Historical price series
    
    Returns:
        Array of shape (n_samples, n_features) where n_features = 23
    """
    # Calculate 5-day returns
    returns = prices.pct_change(periods=5).dropna()
    n_samples = len(returns)
    
    # Time index (trading day number)
    t = np.arange(n_samples)
    
    # Define seasonality periods and Fourier orders
    periods = {
        'weekly': (5, 2),      # 5 trading days, 2 harmonics
        'monthly': (21, 3),    # ~21 trading days, 3 harmonics  
        'quarterly': (63, 2),  # ~63 trading days, 2 harmonics
        'yearly': (252, 4),    # ~252 trading days, 4 harmonics
    }
    
    # Generate Fourier features
    fourier_features = generate_fourier_features(t, periods)
    
    # Combine: [return, fourier_features]
    returns_array = returns.values.reshape(-1, 1)
    observations = np.hstack([returns_array, fourier_features])
    
    return observations
```

### 6.3 Training Code

```python
from hmmlearn import hmm
import numpy as np
import pandas as pd
import joblib

def train_hmm_with_seasonality(
    prices: pd.Series,
    n_states: int = 3,
    model_path: str = None
) -> dict:
    """
    Train a multi-dimensional Gaussian HMM with Fourier seasonality features.
    
    Args:
        prices: Historical price series
        n_states: Number of hidden states (default: 3)
        model_path: Path to save trained model
    
    Returns:
        Dictionary with training results
    """
    # Prepare multi-dimensional observations
    observations = prepare_observations(prices)
    n_samples, n_features = observations.shape
    
    print(f"Training HMM with {n_samples} samples, {n_features} features")
    
    # Initialize model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",  # Diagonal covariance for efficiency
        n_iter=200,              # More iterations for convergence
        random_state=42,
        verbose=False
    )
    
    # Fit model
    model.fit(observations)
    
    # Save model
    if model_path:
        joblib.dump(model, model_path)
    
    # Extract regime characteristics
    regime_names = ["Bull", "Bear", "Sideways"]
    
    # Sort states by return mean (first feature)
    return_means = model.means_[:, 0]
    sorted_indices = np.argsort(return_means)[::-1]  # Descending
    
    # Map to regime names (highest mean = Bull, lowest = Bear)
    state_mapping = {sorted_indices[0]: "Bull", 
                     sorted_indices[2]: "Bear",
                     sorted_indices[1]: "Sideways"}
    
    emission_means = {}
    for state_idx, regime_name in state_mapping.items():
        emission_means[regime_name] = float(model.means_[state_idx, 0])
    
    return {
        'model_type': 'GaussianHMM_Fourier',
        'n_states': n_states,
        'n_features': n_features,
        'log_likelihood': model.score(observations),
        'emission_means': emission_means,
        'n_samples': n_samples,
        'state_mapping': state_mapping,
        'seasonality_periods': {
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63,
            'yearly': 252
        }
    }
```

---

## 7. Prediction

### 7.1 Process

1. Prepare recent observations (return + Fourier features)
2. Use model to infer current regime probabilities
3. Calculate expected return as weighted average of state means
4. Calculate confidence as maximum regime probability

### 7.2 Prediction Code

```python
def predict_return_with_seasonality(
    model: hmm.GaussianHMM,
    prices: pd.Series,
    state_mapping: dict
) -> dict:
    """
    Predict expected return using trained HMM with seasonality.
    
    Args:
        model: Trained GaussianHMM
        prices: Recent price series (for computing observation)
        state_mapping: Mapping from state index to regime name
    
    Returns:
        Dictionary with prediction results
    """
    # Prepare observations
    observations = prepare_observations(prices)
    
    # Get regime probabilities for latest observation
    regime_probs = model.predict_proba(observations)[-1]
    
    # Get current regime (highest probability)
    current_state_idx = np.argmax(regime_probs)
    
    # Reverse mapping: state_idx -> regime_name
    idx_to_name = {v: k for k, v in state_mapping.items()} if isinstance(list(state_mapping.keys())[0], str) else state_mapping
    
    # Get return means for each state (first column of means_)
    return_means = model.means_[:, 0]
    
    # Calculate expected return (weighted average)
    expected_return = np.dot(regime_probs, return_means)
    
    # Confidence is max probability
    confidence = np.max(regime_probs)
    
    # Build regime probabilities dict
    regime_probs_dict = {}
    for state_idx, prob in enumerate(regime_probs):
        regime_name = idx_to_name.get(state_idx, f"State_{state_idx}")
        regime_probs_dict[regime_name] = float(prob)
    
    return {
        "regime": idx_to_name.get(current_state_idx, f"State_{current_state_idx}"),
        "regime_probabilities": regime_probs_dict,
        "expected_return": float(expected_return),
        "confidence": float(confidence),
        "horizon": "5-day"
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
    "model_type": "GaussianHMM_Fourier",
    "regime": "Bull",
    "regime_probabilities": {
        "Bull": 0.72,
        "Bear": 0.18,
        "Sideways": 0.10
    },
    "expected_return": 0.023,  # +2.3% expected over 5 days
    "confidence": 0.72,
    "seasonality_features": {
        "weekly_phase": 0.4,    # Where in the week (0-1)
        "monthly_phase": 0.8,   # Where in the month (0-1)
        "quarterly_phase": 0.2, # Where in the quarter (0-1)
        "yearly_phase": 0.45    # Where in the year (0-1)
    }
}
```

---

## 9. Fourier Series Mathematical Foundation

### 9.1 General Fourier Series

Any periodic function s(t) with period P can be approximated by:

```
s(t) = a_0 + Σ [a_k × cos(2πkt/P) + b_k × sin(2πkt/P)]
       k=1 to K
```

Where:
- a_0 = constant (baseline)
- a_k, b_k = Fourier coefficients (learned by model)
- K = Fourier order (truncation point)
- Higher K = more detail, risk of overfitting

### 9.2 Facebook Prophet's Approach

Prophet models seasonality as:

```
s(t) = Σ [a_k × cos(2πkt/P) + b_k × sin(2πkt/P)]
       k=1 to K
```

Default settings:
- Weekly: K = 3, P = 7
- Yearly: K = 10, P = 365.25

### 9.3 Our Adaptation for Stocks

We use trading days instead of calendar days:
- Weekly: K = 2, P = 5 (trading days)
- Monthly: K = 3, P = 21
- Quarterly: K = 2, P = 63
- Yearly: K = 4, P = 252

**Key Difference:** Instead of fitting coefficients directly (like Prophet), we use Fourier terms as **observation features** in the HMM, allowing the model to learn **regime-dependent seasonality**.

---

## 10. Hyperparameters

| Parameter | Default | Tuning Range | Description |
|-----------|---------|--------------|-------------|
| `n_components` | 3 | [2, 3, 4] | Number of hidden states |
| `covariance_type` | "diag" | ["diag", "full"] | Covariance matrix type |
| `n_iter` | 200 | [100, 200, 500] | Max training iterations |
| `weekly_order` | 2 | [1, 2, 3] | Fourier order for weekly |
| `monthly_order` | 3 | [2, 3, 4] | Fourier order for monthly |
| `quarterly_order` | 2 | [1, 2, 3] | Fourier order for quarterly |
| `yearly_order` | 4 | [2, 4, 6] | Fourier order for yearly |

### Tuning Strategy

1. Start with defaults
2. If capturing too much noise, reduce Fourier orders
3. If missing seasonality patterns, increase Fourier orders
4. Monitor log-likelihood and directional accuracy
5. Use cross-validation on out-of-sample periods

---

## 11. Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Directional Accuracy** | % of times predicted direction matches actual | > 55% |
| **MSE** | Mean squared error of predicted vs actual returns | Lower is better |
| **Log-Likelihood** | Model fit quality (from hmmlearn) | Higher is better |
| **Seasonality Capture** | Visual inspection of regime patterns | Clear patterns |

### Evaluation Code

```python
def evaluate_model_with_seasonality(model, test_prices, state_mapping):
    """Evaluate HMM prediction accuracy with seasonality features."""
    
    observations = prepare_observations(test_prices)
    returns = test_prices.pct_change(periods=5).dropna().values
    
    predictions = []
    for i in range(50, len(observations)):  # Warmup period
        obs_so_far = observations[:i]
        regime_probs = model.predict_proba(obs_so_far)[-1]
        return_means = model.means_[:, 0]
        pred_return = np.dot(regime_probs, return_means)
        predictions.append(pred_return)
    
    actual_returns = returns[50:][:len(predictions)]
    predictions = np.array(predictions)
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actual_returns)
    directional_accuracy = np.mean(pred_direction == actual_direction)
    
    # MSE
    mse = np.mean((predictions - actual_returns) ** 2)
    
    return {
        "directional_accuracy": directional_accuracy,
        "mse": mse,
        "log_likelihood": model.score(observations),
        "n_predictions": len(predictions)
    }
```

---

## 12. Advantages of Fourier-Enhanced HMM

| Aspect | Basic HMM | Fourier-Enhanced HMM |
|--------|-----------|----------------------|
| Features | 1 (return only) | 23 (return + seasonality) |
| Seasonality | Not captured | Weekly, monthly, quarterly, yearly |
| Regime patterns | Return-based only | Return + time-based patterns |
| Market anomalies | Ignored | Monday effect, January effect, etc. |
| Complexity | Lower | Higher (more parameters) |

---

## 13. Known Stock Market Seasonalities

The Fourier features help capture these documented anomalies:

| Anomaly | Period | Description |
|---------|--------|-------------|
| **Monday Effect** | Weekly | Lower returns on Mondays |
| **Turn-of-Month** | Monthly | Higher returns at month start/end |
| **January Effect** | Yearly | Higher returns in January |
| **Sell in May** | Yearly | Lower returns May-October |
| **Earnings Seasons** | Quarterly | Volatility around earnings |
| **Tax-Loss Harvesting** | Yearly | Selling pressure in December |
| **Window Dressing** | Quarterly | Fund rebalancing at quarter-end |

---

## 14. Implementation Notes

### 14.1 Scaling Features

```python
from sklearn.preprocessing import StandardScaler

def prepare_observations_scaled(prices: pd.Series) -> tuple:
    """Prepare and scale observations."""
    observations = prepare_observations(prices)
    
    scaler = StandardScaler()
    observations_scaled = scaler.fit_transform(observations)
    
    return observations_scaled, scaler
```

### 14.2 Handling Different Start Dates

```python
def get_time_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Convert dates to continuous trading day index."""
    # Use day of year for yearly seasonality
    # Use weekday for weekly seasonality
    # etc.
    return np.arange(len(dates))
```

---

## 15. Summary

| Aspect | Value |
|--------|-------|
| Model | Gaussian HMM with Fourier Seasonality |
| Hidden States | 3 (Bull, Bear, Sideways) |
| Observable Dimension | 23 (1 return + 22 Fourier features) |
| Seasonality Captured | Weekly, Monthly, Quarterly, Yearly |
| Scope | Per-stock |
| Output | Expected return + confidence + regime |
| Decision | None (Coordinator Agent decides) |
| Inspired By | Facebook Prophet's Fourier approach |

---

## 16. References

1. Facebook Prophet Paper: Taylor, S.J. & Letham, B. (2018). "Forecasting at Scale"
2. Hyndman, R.J. "Forecasting: Principles and Practice" - Chapter on Dynamic Harmonic Regression
3. hmmlearn documentation: https://hmmlearn.readthedocs.io/
4. Stock market seasonality literature: French (1980), Lakonishok & Smidt (1988)
