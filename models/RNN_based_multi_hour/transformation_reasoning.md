# Data Transformation Reasoning

## 1. Source Data Analysis
- **Resolution and format**: The source data is in hourly resolution, with 7 bars per trading day.
- **Available factors**: The source data includes various independent factors (e.g., `compute_log_return_y`, `compute_return_1h`, `compute_return_5h`) and shared factors (e.g., `compute_spy_return`, `compute_vix_level`, `compute_qqq_return`).
- **Embeddings**: The source data includes embeddings with a dimension of 384.

## 2. Target Model Requirements
- **Expected input format**: The target model expects daily resolution input data.
- **Column naming**: The target model expects column names to follow the convention `x_{feat}_s{series}`.
- **Output format**: The target model expects output data with 7 steps.

## 3. Transformation Decisions

### Y Value Handling
- **Why this Y column**: The `compute_log_return_y` column is chosen as the target Y value because it represents the log return within the same hourly bar.
- **Window shift explanation**: To predict Y at time t, we use X from 1 step ago. This means that the X features are shifted forward by 1 step, while the Y values keep their true time index. This approach avoids look-ahead bias and ensures that the model only uses information available up to the previous time step to make predictions.
- **Output column format**: The Y values are expanded to 7 columns to match the output format expected by the target model.

### Independent Factors
| Factor | Aggregation | Reasoning |
|--------|-------------|-----------|
| compute_return_1h | mean | The mean aggregation is used to calculate the average 1-hour return over the 7 hourly bars in a day. |
| compute_return_5h | mean | The mean aggregation is used to calculate the average 5-hour return over the 7 hourly bars in a day. |
| compute_rsi | last | The last aggregation is used to take the last RSI value of the day, as it is a momentum indicator that is typically used to identify overbought or oversold conditions. |
| compute_volatility | mean | The mean aggregation is used to calculate the average volatility over the 7 hourly bars in a day. |
| compute_macd | last | The last aggregation is used to take the last MACD value of the day, as it is a trend-following indicator that is typically used to identify buying or selling opportunities. |
| compute_macd_signal | last | The last aggregation is used to take the last MACD signal value of the day, as it is a trend-following indicator that is typically used to identify buying or selling opportunities. |
| compute_macd_hist | last | The last aggregation is used to take the last MACD histogram value of the day, as it is a trend-following indicator that is typically used to identify buying or selling opportunities. |
| compute_volume_ratio | mean | The mean aggregation is used to calculate the average volume ratio over the 7 hourly bars in a day. |
| compute_price_range | mean | The mean aggregation is used to calculate the average price range over the 7 hourly bars in a day. |
| compute_sector_return | mean | The mean aggregation is used to calculate the average sector return over the 7 hourly bars in a day. |

### Shared Factors
| Factor | Aggregation | Reasoning |
|--------|-------------|-----------|
| compute_spy_return | mean | The mean aggregation is used to calculate the average SPY return over the 7 hourly bars in a day. |
| compute_vix_level | last | The last aggregation is used to take the last VIX level of the day, as it is a volatility indicator that is typically used to identify market sentiment. |
| compute_qqq_return | mean | The mean aggregation is used to calculate the average QQQ return over the 7 hourly bars in a day. |
| compute_iwm_return | mean | The mean aggregation is used to calculate the average IWM return over the 7 hourly bars in a day. |
| compute_fed_rate | last | The last aggregation is used to take the last Federal Funds Rate of the day, as it is an interest rate that is typically used to influence monetary policy. |
| compute_cpi | last | The last aggregation is used to take the last Consumer Price Index (CPI) of the day, as it is an inflation indicator that is typically used to identify changes in consumer prices. |

### Embeddings
- **How mapped to timestamps**: The embeddings are broadcast to all timestamps, assuming that the embeddings are static and do not change over time.
- **Dimension handling**: The embeddings are used as-is, with a dimension of 384.

## 4. Time Window Shifting
To avoid look-ahead bias and keep Y's true time index, we shift the X features forward by 1 step. This means that:

- Y values keep their TRUE time index (when the return actually occurred)
- X features are shifted FORWARD by 1 steps
- Row at time t has: X[t-1] and Y[t]
- This means: to predict Y at time t, we use X from 1 step(s) ago

Example:

| Time | X[t-1] | Y[t] |
| --- | --- | --- |
| 2022-01-01 09:30 | X[2022-01-01 09:30] | Y[2022-01-01 09:30] |
| 2022-01-01 10:30 | X[2022-01-01 09:30] | Y[2022-01-01 10:30] |
| ... | ... | ... |

## 5. Issues and Warnings
- **Source data has hourly resolution, but target model expects daily resolution**: We aggregate independent and shared factors using mean and last values respectively to convert the data to daily resolution.
- **Embeddings are broadcast to all timestamps**: Ensure that this is correct for the model, as embeddings are assumed to be static and do not change over time.