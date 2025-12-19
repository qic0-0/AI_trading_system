# Create features DataFrame with prices.index
feature_df = pd.DataFrame(index=prices.index)

# Calculate RSI (14-day)
delta = prices.diff()
gain = delta.copy()
loss = delta.copy()
gain[gain < 0] = 0
loss[loss > 0] = 0
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = abs(loss).ewm(span=14, adjust=False).mean()
rs = avg_gain / avg_loss
feature_df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

# Calculate MACD
ema_12 = prices.ewm(span=12, adjust=False).mean()
ema_26 = prices.ewm(span=26, adjust=False).mean()
feature_df['macd'] = ema_12 - ema_26

# Calculate SMA ratios (5, 10, 20, 50)
feature_df['sma_5'] = prices / prices.rolling(window=5).mean()
feature_df['sma_10'] = prices / prices.rolling(window=10).mean()
feature_df['sma_20'] = prices / prices.rolling(window=20).mean()
feature_df['sma_50'] = prices / prices.rolling(window=50).mean()

# Calculate Momentum (5-day, 10-day)
feature_df['momentum_5'] = prices.pct_change(5)
feature_df['momentum_10'] = prices.pct_change(10)

# Calculate Volatility
feature_df['volatility'] = prices.pct_change().ewm(span=14, adjust=False).std()

# Add embedding factor placeholders (for news sentiment during prediction)
feature_df['news_sentiment'] = 0.0
feature_df['news_magnitude'] = 0.0
feature_df['news_novelty'] = 0.5

# Create target and ALIGN with features
target_series = (prices.shift(-1) > prices).astype(int)
feature_df['target'] = target_series
feature_df = feature_df.dropna()  # Drop NaN ONCE after combining

# Split data
X = feature_df.drop('target', axis=1)
y = feature_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
xgb_model = xgb.XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Save model
joblib.dump({'model': xgb_model, 'scaler': scaler, 'feature_names': list(X.columns)}, model_path)

# Evaluate model
y_pred = xgb_model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)

# Create result dict
result = {
    'model_type': 'XGBoost',
    'accuracy': accuracy,
    'n_features': len(X.columns),
    'feature_names': list(X.columns),
    'n_samples': len(X)
}