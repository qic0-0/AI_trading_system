# Multivariate RNN Forecasting Model

## Overview

A multivariate time series forecasting model using RNN with Fourier features and cross-series attention. The model forecasts hourly values (24 hours) for multiple time series simultaneously, leveraging correlations between series.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Input Layer                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Shared Features ──────────────────────────→ rnn_shared ──→ h_shared            │
│                                                                                  │
│  Independent Features (per series):                                              │
│    x_0_s0 (1-dim) ──→ feature_embed[0] ─┐                                       │
│    x_1_s0 (1-dim) ──→ feature_embed[1] ─┼→ concat ──→ rnn_indep[0] ──→ h_indep[0]│
│    emb_0_s0 (768-dim) ──→ text_embed[0] ┘                                       │
│                                                                                  │
│    x_0_s1 (1-dim) ──→ feature_embed[0] ─┐                                       │
│    x_1_s1 (1-dim) ──→ feature_embed[1] ─┼→ concat ──→ rnn_indep[1] ──→ h_indep[1]│
│    emb_0_s1 (768-dim) ──→ text_embed[0] ┘                                       │
│                                                                                  │
│  Fourier Features ──→ fourier_layers[s] ──→ fourier_contrib[s]                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           Combine per Series                                     │
│  z[s] = h_shared + h_indep[s] + fourier_contrib[s]                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                        Feature Attention (per series)                            │
│  z_attended[s] = feature_attn_layers[s](z[s])                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                         Cross-Series Attention                                   │
│  z = cross_attn(stack(z_attended))                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                         Output Layer (per series)                                │
│  output[s] = output_layers[s](z[:,:,s,:]) → 24 hourly values                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Design:**
- `feature_embed[f]` - Shared MLP across all series for general features (learns feature type similarity)
- `text_embed[e]` - Shared MLP across all series for text embeddings (separate from general features)
- `rnn_indep[s]` - Independent RNN per series (learns series-specific dynamics)
- `z0[s]` - Independent initial state per series

---

## Model Output

The output is hourly data vector in each day (hourly data does not have to be 24 hours, e.g. data only have 10:00 an and 11:00 am, model is also able to predict)

---

## Input Data Format

### Required DataFrame Structure

The input DataFrame must have a **wide format** with columns organized as follows:

| Column Type | Naming Convention | Description                                                                                              |
|-------------|-------------------|----------------------------------------------------------------------------------------------------------|
| Day | `day` | Date column (datetime)                                                                                   |
| Shared Features | `share_0`, `share_1`, ... | Features common to all series (e.g., weather, holidays)                                                  |
| Independent Features | `x_{feat}_s{series}` | Features specific to each series                                                                         |
| Text Embeddings | `emb_{emb_idx}_s{series}_d{dim}` | High-dimensional embeddings (e.g., BERT, GPT)                                                            |
| Target Variables | `y_{hour}_s{series}` | Hourly target values for each series, it does not need to has 24 hours data, 7 hours daily data also work|

### Example DataFrame Structure

For 2 series with 2 independent features, 1 text embedding (16-dim), and no shared features:

```
| day        | x_0_s0  | x_1_s0  | x_0_s1  | x_1_s1  | emb_0_s0_d0 | ... | emb_0_s0_d15 | emb_0_s1_d0 | ... | emb_0_s1_d15 | y_0_s0 | ... | y_23_s0 | y_0_s1 | ... | y_23_s1 |
|------------|---------|---------|---------|---------|-------------|-----|--------------|-------------|-----|--------------|--------|-----|---------|--------|-----|---------|
| 2004-10-01 | 340923  | 341000  | 45876   | 45900   | 0.123       | ... | -0.456       | 0.789       | ... | 0.012        | 12379  | ... | 14067   | 1621   | ... | 1869    |
| 2004-10-02 | 311997  | 312100  | 39398   | 39450   | -0.234      | ... | 0.567        | -0.890      | ... | 0.345        | 13147  | ... | 14015   | 1700   | ... | 1590    |
```

### Column Naming Rules

- **Independent features**: `x_{feature_index}_s{series_index}`
  - `x_0_s0` = Feature 0 for Series 0
  - `x_1_s0` = Feature 1 for Series 0
  - `x_0_s1` = Feature 0 for Series 1

- **Text embeddings**: `emb_{emb_index}_s{series_index}_d{dimension}`
  - `emb_0_s0_d0` = Embedding 0, Series 0, Dimension 0
  - `emb_0_s0_d767` = Embedding 0, Series 0, Dimension 767
  - `emb_1_s0_d0` = Embedding 1, Series 0, Dimension 0

- **Shared features**: `share_{feature_index}`
  - `share_0` = Shared feature 0 (same value for all series)

- **Targets**: `y_{hour}_s{series_index}`
  - `y_0_s0` = Hour 0 value for Series 0
  - `y_23_s1` = Hour 23 value for Series 1

---

## Data Preparation

### Example Merging Multiple Datasets

**important** example data are in daily scale where 'y_{hour}' columns are hourly data, 'x_{feat}' are daily input

**important** all input are in daily resolution, if your data is in hourly resolution (same as y value), come up with a way to converge to daily resolution, or you can simply expand to multiple columns

```python
import pandas as pd
import numpy as np
import Multivariate_forcasting

# Load datasets
df_aep = pd.read_csv("data/AEP_DATA.csv")
df_dayton = pd.read_csv("data/DAYTON_DATA.csv")
df_shared = pd.read_csv("data/SHARE_DATA.csv")

# Convert day to datetime
df_aep['day'] = pd.to_datetime(df_aep['day'])
df_dayton['day'] = pd.to_datetime(df_dayton['day'])

# Rename columns for series 0 (AEP)
aep_rename = {'x_0': 'x_0_s0', 'x_1': 'x_1_s0'}
for h in range(24):
    aep_rename[f'y_{h}'] = f'y_{h}_s0'
df_aep = df_aep.rename(columns=aep_rename)

# Rename columns for series 1 (DAYTON)
dayton_rename = {'x_0': 'x_0_s1', 'x_1': 'x_1_s1'}
for h in range(24):
    dayton_rename[f'y_{h}'] = f'y_{h}_s1'
df_dayton = df_dayton.rename(columns=dayton_rename)

# Rename columns for Shared
df_shared.columns = ['day', 'share_0', 'share_0']

# Inner join on day
df_multi = pd.merge(df_aep, df_dayton, on='day', how='inner')
df_multi = pd.merge(df_multi, df_shared, on='day', how='inner')

# remove NA
df_multi.dropna(inplace=True)
```

### Adding Text Embeddings

**important:** below is just an example showing how to adapt to model's input, you should act on your embedding data instead of using random noise

```python
# Example: Add 1 text embedding (16-dim) per series
n_series = 2
emb_dim = 16

for s in range(n_series):
    for d in range(emb_dim):
        df_multi[f'emb_0_s{s}_d{d}'] = np.random.randn(len(df_multi))

# Example: Add 2 text embeddings (768-dim and 512-dim) per series
emb_dims = [768, 512]

for s in range(n_series):
    for e, dim in enumerate(emb_dims):
        for d in range(dim):
            df_multi[f'emb_{e}_s{s}_d{d}'] = np.random.randn(len(df_multi))

# Verify columns
print([c for c in df_multi.columns if c.startswith('emb_')])
```

---

## Model Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_series` | int | 1 | Number of time series |
| `n_shared` | int | 0 | Number of shared features |
| `n_indep` | List[int] | None | Number of independent features per series |
| `fourier_dim` | int | 0 | Dimension of Fourier features |
| `xf_mode` | str | "vector" | Fourier mode: "vector" or "matrix" |
| `latent_dim` | int | 24 | Hidden dimension of RNN |
| `d_model` | int | 128 | Dimension for attention |
| `nhead` | int | 4 | Number of attention heads |
| `H` | int | 24 | Number of output hours |
| `rnn_type` | str | "rnn" | RNN type: "rnn" or "gru" |
| `dropout` | float | 0.0 | Dropout rate |
| `learn_z0` | bool | True | Learn initial hidden state |
| `unique_alpha` | bool | False | Unique Fourier weights per hour |
| `embed_dim` | int | 32 | Output dimension of feature embeddings |
| `embed_hidden` | int | 64 | Hidden dimension of feature embedding MLP |
| `text_embed_dims` | List[int] | None | Input dimensions for each text embedding (e.g., [768, 512]) |
| `text_embed_hidden` | int | 256 | Hidden dimension of text embedding MLP |

### Training Configuration

```python
from Multivariate_forcasting import training_config

train_config = training_config(
    n_epochs=32,           # Number of training epochs
    device=torch.device("cuda"),  # Device (cuda/mps/cpu)
    T_hist=32,             # History length for sequences
    lr=5e-4,               # Learning rate
    lambda0=1e-5,          # L2 regularization
    lambdaf=5e-4,          # Fourier regularization
)
```

### Fourier Configuration

```python
from Multivariate_forcasting import fourier_config

fourier_conf = fourier_config(
    mode="matrix",         # keep "matrix"
    K_weekly=3,            # Weekly Fourier order
    K_monthly=6,           # Monthly Fourier order
    K_yearly=10,           # Yearly Fourier order
    P_WEEK=7.0,            # Weekly period (days)
    P_MONTH=365.25/12.0,   # Monthly period (days)
    P_yearly=365.25,       # Yearly period (days)
)
```

---

## Usage

### Basic Training and Forecasting (Without Text Embeddings)

```python
import Multivariate_forcasting
import torch

# Configuration
n_series = 2
n_indep = [2, 2]  # 2 features per series

K_total = fourier_conf.K_weekly + fourier_conf.K_monthly + fourier_conf.K_yearly
fourier_dim = 2 * K_total
```

### if use cuda

```python
# Model config
model_config = dict(
    fourier_dim=fourier_dim,
    xf_mode=fourier_conf.mode,
    d_model=128,
    latent_dim=32,
    nhead=4,
    n_series=n_series,
    n_shared=0,
    n_indep=n_indep,
)

# Create model
model = Multivariate_forcasting.build_model_dp(
    Multivariate_forcasting.RNN_fourier, 
    **model_config
)
```

### if use mps or cpu

```python
model = Multivariate_forcasting.RNN_fourier(
    fourier_dim=fourier_dim,
    xf_mode=fourier_conf.mode,
    d_model=128,
    latent_dim=32,
    nhead=4,
    n_series=n_series,
    n_shared=0,
    n_indep=n_indep,
)

```

### following are same for all device

```python
# Create trainer 
trainer = Multivariate_forcasting.RNN_train_fourier(
    model, 
    train_config, 
    fourier_conf, 
    deterministic=True  # Set False for prediction intervals
)

# Train
trainer(df_train)

# Forecast
prediction_results = trainer.forecast(df_test)
```

### Training with Text Embeddings

```python
import Multivariate_forcasting
import torch

# Configuration
n_series = 2
n_indep = [2, 2]  # 2 general features per series

K_total = fourier_conf.K_weekly + fourier_conf.K_monthly + fourier_conf.K_yearly
fourier_dim = 2 * K_total

# Model config with text embeddings
model_config = dict(
    fourier_dim=fourier_dim,
    xf_mode=fourier_conf.mode,
    d_model=128,
    latent_dim=32,
    nhead=4,
    n_series=n_series,
    n_shared=0,
    n_indep=n_indep,
    # Text embedding parameters
    text_embed_dims=[16],      # 1 embedding with 16 dimensions
    text_embed_hidden=256,     # Hidden dim for text MLP
    embed_dim=32,              # Output dim for all embeddings
    embed_hidden=64,           # Hidden dim for general feature MLP
)

# Create model
model = Multivariate_forcasting.build_model_dp(
    Multivariate_forcasting.RNN_fourier, 
    **model_config
)

# Create trainer
trainer = Multivariate_forcasting.RNN_train_fourier(
    model, 
    train_config, 
    fourier_conf, 
    deterministic=True
)

# Train
trainer(df_train)

# Forecast
prediction_results = trainer.forecast(df_test)
```

### Multiple Text Embeddings Example

```python
# For 2 text embeddings per series: 768-dim (BERT) and 512-dim (custom)
model_config = dict(
    fourier_dim=fourier_dim,
    xf_mode=fourier_conf.mode,
    d_model=128,
    latent_dim=32,
    nhead=4,
    n_series=n_series,
    n_shared=0,
    n_indep=n_indep,
    text_embed_dims=[768, 512],  # 2 embeddings with different dims
    text_embed_hidden=256,
)
```

---

## Output Format

### Deterministic Mode (`deterministic=True`)

Returns a dictionary with lists per series:

```python
{
    'test_pred': [
        array([...]),  # Series 0 predictions (24 values per day)
        array([...]),  # Series 1 predictions
        ...
    ],
    'test_true': [
        array([...]),  # Series 0 true values
        array([...]),  # Series 1 true values
        ...
    ]
}
```

### Probabilistic Mode (`deterministic=False`)

Returns additional uncertainty estimates:

```python
{
    'test_pred': [...],      # Point predictions per series
    'test_true': [...],      # True values per series
    'y_pred_lower': [...],   # Lower bound (95% CI) per series
    'y_pred_upper': [...],   # Upper bound (95% CI) per series
    'p_values': [...],       # P-values per series
}
```

### Accessing Results

```python
# Get predictions for series 0
pred_s0 = prediction_results['test_pred'][0]  # Shape: (n_days * 24,) or (24,)
true_s0 = prediction_results['test_true'][0]

# Get predictions for series 1
pred_s1 = prediction_results['test_pred'][1]
true_s1 = prediction_results['test_true'][1]

# Calculate RMSE per series
for s in range(n_series):
    rmse = np.sqrt(np.mean((prediction_results['test_pred'][s] - prediction_results['test_true'][s])**2))
    print(f"Series {s} RMSE: {rmse:.2f}")
```

---

## Notes

1. **Fourier Features**: Automatically generated based on `fourier_config`. No need to add manually.

2. **Scaling**: Data is automatically standardized internally. No need to pre-scale.

3. **Cross-Series Attention**: Only active when `n_series > 1`. Allows series to learn from each other.

4. **GPU Support**: Use `build_model_dp()` for automatic DataParallel when multiple GPUs available.

5. **Column Naming**: Must use exact naming convention:
   - General features: `x_{i}_s{s}`
   - Text embeddings: `emb_{e}_s{s}_d{d}`
   - Targets: `y_{h}_s{s}`
   - Shared: `share_{i}`

6. **Text Embeddings**: 
   - Shared MLP (`text_embed[e]`) learns feature type similarity across series
   - Separate from general feature MLP (`feature_embed[f]`)
   - Each series has independent RNN (`rnn_indep[s]`) after embedding

7. **Feature Embedding Design**:
   - General features (low-dim like scalars) → `feature_embed` (shared across series)
   - Text embeddings (high-dim like BERT) → `text_embed` (shared across series, separate MLP)
   - All embeddings concatenated → series-specific RNN
