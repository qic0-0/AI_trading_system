from hmmlearn.hmm import GaussianHMM

# Initialize and fit GaussianHMM with n_states components
model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
model.fit(returns)

# Save trained model to model_path using joblib.dump()
joblib.dump(model, model_path)

# Create 'result' dictionary with key metrics
result = {
    'model_type': 'GaussianHMM',
    'n_states': n_states,
    'log_likelihood': model.score(returns),
    'emission_means': {
        f'regime_{i}': model.means_[i][0] for i in range(n_states)
    },
    'n_samples': len(returns)
}