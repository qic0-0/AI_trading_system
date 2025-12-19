import sys
print("=" * 60)
print("       FINAL EVALUATION - Quant Trading Agent System")
print("=" * 60)

results = []

# Test 1: Imports
print("\n[1/8] Testing imports...")
try:
    from config import config
    from llm.llm_client import LLMClient
    from agents.data_agent import DataAgent
    from agents.feature_agent import FeatureEngineeringAgent
    from agents.quant_model_agent import QuantModelingAgent
    from agents.market_sense_agent import MarketSenseAgent
    from agents.coordinator_agent import CoordinatorAgent
    print("  ✓ All imports successful")
    results.append(("Imports", True))
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    results.append(("Imports", False))
    sys.exit(1)

# Test 2: LLM Client
print("\n[2/8] Testing LLM Client...")
try:
    llm = LLMClient(config.llm)
    from llm.llm_client import Message
    resp = llm.chat([Message("user", "Say OK")])
    assert len(resp.content) > 0
    print("  ✓ LLM Client working")
    results.append(("LLM Client", True))
except Exception as e:
    print(f"  ✗ LLM Client failed: {e}")
    results.append(("LLM Client", False))

# Test 3: Data Agent
print("\n[3/8] Testing Data Agent...")
try:
    data_agent = DataAgent(llm, config)
    prices = data_agent.fetch_pri    prices = data_agent.fetch_pri    6-01")
    assert len(prices) > 50
    print(f"  ✓ Data Agent work    print(f"  ✓ Data Agent work    print(f"  ✓ nd(("Data Agent", True))
except Exception as e:
    print(f"  ✗ Data Agent failed: {e}")
    results.append(("Data Agent", False))

# Test 4: Feature Agent
print("\n[print("\n[print("\n[prinnt...")
try:
    feature_agent = FeatureEngineeringAgent(llm, config)
    indicators = feature_agent.compute_technical_indicators(prices)
    assert "RSI_14" in indicators.columns
    print(f"  ✓ Feature Agent working ({len(indicators.columns)} indicators)")
    results.append(("Feature Agent", True))
except Exception as e:
    print(f"  ✗ Feature Agent failed: {e}")
    results.append(("Feature Agent", False))

# Test 5: Quant Model Agent - Train
print("\n[5/8] Testing Quant Model Agent (Train)...")
try:
    quant_agent = QuantModelingAgent(llm, config)
    train_result = quant_agent.run({"mode": "train", "price_data": {"AAPL": pr    train_result = quant_agent.run({"mode": "train", "price_data": {" trainin    train_result = quas.append(("Quant Train", True))
except Exception as e:
    print(f"  ✗ Quant Agent training failed: {e}")
    results.append(("Quant Tra    results.append(("Quant Tra    resulnt - Predict
print("\n[6/8] Testing Quant Modelprint("\n[6/8] )...")
trtrtrtrtrtrtrtrtrtrtrtrtrtrtrtrtrtrun({"mode": "predict", "price_data": {"AAPL": prices}})
    pred = pred_result.data.get("predictions", {}).get("AAPL", {})
    assert "expected_return" in pred
    assert "regime"     assert "regime"     assert "regime"     assert "regime"  urn={pred['expected_    assert "regime"     assert "rme']})")
                     Quant Predict", True))
except Exception as e:
    print(f"  ✗ Quant Agent prediction failed: {e}")
    results.append(("Quant Predict", False))

# Test 7: Market-Sense Agent
print("\n[7/8] Testing Market-Sense Agent...")
try:
    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    market_ag    marketquant_signal": pred,
        "ticker": "AAPL"
    })
    insight = market_result.data.get("insight", {})
    assert "outlook" in insight
    print(f"  ✓ Market-    print(f"  ✓ Market-    print(f"  ✓ Market-          print(f"  ✓ Market-    print(f"  ✓ Market-    print(f"  ✓ Market-          prinens    print(f"  ✓ Market-    print(f"  ✓ Market- ense", False))

# Test 8: Coordinator Agent
print("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprint("\nprinker": "AAPL",
        "current_price": 200.0,
        "quant_signal": {        "quant_signal": {        "quant_sig "regime": "        "quant_signal": {        "q"outlook": "BULLISH", "confidence": 0.7,        "quant_signal": {        "qulio = coord_result.data.get("portfolio_state", {})
    print(f"  ✓ Coordinator Agent working (decision={coord_result.message})")
    results.append(("Coordinator", True))
except Exception as e:
    print(f"  ✗ Coordinator Agent failed: {e}")
    results.append(("Coordinator", False))

# Summary
print("\n" + "=" * 60)
print("                    EVALUATION SUMMARY")
print("                    EVALUATION SUMMARY")
)
ion={coord_result.message})")
        "quant_signal": {        "q"outlook": "BULLISH", "confidence": 0.7,    print(f"  {icon} {name}")
print(f"\nResult: {passed}/{total} tests passed")
if passed == total:
    print("\n🎉 ALL TESTS PASSED - System is fully functional!")
else:
    print(f"\n⚠️  {total - passed} test(s) failed - needs fixing")
print("=" * 60)
