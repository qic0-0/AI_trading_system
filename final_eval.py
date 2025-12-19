import sys
print('=' * 60)
print('       FINAL EVALUATION - Quant Trading Agent System')
print('=' * 60)

results = []

# Test 1: Imports
print('[1/8] Testing imports...')
try:
    from config import config
    from llm.llm_client import LLMClient
    from agents.data_agent import DataAgent
    from agents.feature_agent import FeatureEngineeringAgent
    from agents.quant_model_agent import QuantModelingAgent
    from agents.market_sense_agent import MarketSenseAgent
    from agents.coordinator_agent import CoordinatorAgent
    print('  OK - All imports successful')
    results.append(('Imports', True))
except Exception as e:
    print(f'  FAIL - Import failed: {e}')
    results.append(('Imports', False))
    sys.exit(1)

# Test 2: LLM Client
print('[2/8] Testing LLM Client...')
try:
    llm = LLMClient(config.llm)
    from llm.llm_client import Message
    resp = llm.chat([Message('user', 'Say OK')])
    assert len(resp.content) > 0
    print('  OK - LLM Client working')
    results.append(('LLM Client', True))
except Exception as e:
    print(f'  FAIL - LLM Client failed: {e}')
    results.append(('LLM Client', False))

# Test 3: Data Agent
print('[3/8] Testing Data Agent...')
try:
    data_agent = DataAgent(llm, config)
    prices = data_agent.fetch_price_data('AAPL', '2024-01-01', '2024-06-01')
    assert len(prices) > 50
    print(f'  OK - Data Agent working ({len(prices)} days)')
    results.append(('Data Agent', True))
except Exception as e:
    print(f'  FAIL - Data Agent failed: {e}')
    results.append(('Data Agent', F    results.append(('Data Agent', F    [4/8] Testing Feature Agent...')
try:
    feature_agent = FeatureEngineeringAgent(llm, config)
    indicators = feature_agent.compute_technical_indicators(prices)
    assert 'RSI_14' in indicators.columns
    print(f'  OK - Feature Agent working ({len(indica    print(f'  OK - Feature Ag    results.append(('Feature Agent', True))
except Exceptioexcept Exceptioexcept Except Feature Agent failed: {e}')
    results.append(('Feature Agent', False))

# Test 5: Quant Model Agent - Tra# Test 5: Quant Model Agent - Tra# Testnt (Train)...')
try:
    quant_agent = QuantModelingAgent(llm, config)
    quant_agent = QuantM_agent.run({'mode'    quant_agent = QuantM_a'AAPL': prices}})
    assert train_result.success
    print('  OK - Quant Agent training working')
    results.append((    results.append((    results.append((    results.appf'  FA    results.append((    results.ap{e}')
    results.append(('Qua    results.append(('Qua    results.appen Agen    results.append(('Qua    results.append(('Qua    results.appen Agen    results.append(('Qua    resuln({'mode'    results.append(('Qua    results.append(('Qu   pred = pred_result.data.get('predictions', {}).get('AAPL', {})
    assert '    assert '    assert '     print(f'  OK - Quant Agent prediction working')
    results.append(('Quant Predict', True))
except Exception as e:
    print(f'  FAIL - Quant Agent prediction failed: {e}')
    results.append(('Quant Predict', False))

# Test 7: Market-Sense Agent
print('[7/8] Testing Market-Sense Agent...')
try:
    market_agent = MarketSenseAgent(llm, config)
    market_result = market_agent.run({
        'market_state': {'ticker_data': {'ticker': 'AAPL', 'price': 200.0}},
        'news': ['Apple reports earnings'],
        'quant_signal': pred,
        'ticker': 'AAPL'
    })
    insight = market_result.data.get('insight', {})
    assert 'outlook' in insight
    print(f'  OK - Market-Sense Agent working')
    results.append(('Market-Sense', True))
except Exception as e:
    print(f'  FAIL - Market-Sense Agent failed: {e}')
    results.append(('Market-Sense', False))

# Test 8: Coordinator Agent
print('[8/8] Testing Coordinator Agent...')
try:
    coordinator = CoordinatorAgent(llm, config)
    coord_result = coordinator.run({
        'ticker': 'AAPL',
        'current_p        'current_p        'cusignal': {'expected_return': 0.03, 'co        'current_p        'current_p        'cusignal'ight        'current_p        'current_p  : 0.7, 'risk_flags': []}
    })
    print(f'  OK - Coordinator Agent working')
    resu    resund(('Coordinator', True))
except Exception as e:
    print(f'  FAIL - Coordinator Agent failed: {e}')
    results.append(('Coordinator', False))

# Summary
print()
print('=' * 60)
print('                    EVALUATION SUMMARY')
print('=' * 60)
passed = sum(1 for passed = sum(1 for passed = sum(1 for passed = ults)passed = sum(1 for passed = s
                                              int(                                      {passed}/{total} tests passed')
if passed == total:
    print('ALL TESTS PASSED - System is fully functional!')
print('=' * 60)
