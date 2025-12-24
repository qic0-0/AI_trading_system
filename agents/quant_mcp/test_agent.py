"""
Test Agent - Tests the complete model pipeline.

This agent:
1. Imports generated/provided model code
2. Imports data adapter
3. Loads sample data and transforms it
4. Tests model training (with small sample)
5. Tests model prediction
6. Reports any errors with analysis

The Test Agent helps identify which component has bugs:
- Model code issues
- Data adapter issues
- Integration issues
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional, List

from .base_mcp_agent import BaseMCPAgent
from .mcp_config import AgentTask, AgentResponse

logger = logging.getLogger(__name__)


class TestAgent(BaseMCPAgent):
    """
    Agent that tests the complete model pipeline.
    
    Test steps:
    1. Import test - Can we import the modules?
    2. Data load test - Can we load and transform data?
    3. Train test - Can we train the model?
    4. Predict test - Can we make predictions?
    """
    
    def __init__(self, llm_client, config):
        super().__init__("TestAgent", llm_client, config)
    
    @property
    def system_prompt(self) -> str:
        return """You are a Test Agent specialized in testing quantitative model pipelines.

Your task is to:
1. Test that all code imports correctly
2. Test that data can be loaded and transformed
3. Test that the model can be trained
4. Test that predictions can be made
5. Analyze any errors and identify the likely source

When errors occur:
- Determine if it's a model code issue, data adapter issue, or integration issue
- Provide clear error messages with line numbers if possible
- Suggest fixes when appropriate

Be thorough but efficient - use sample data for testing, not full datasets."""
    
    def _register_tools(self):
        """Register Test Agent specific tools."""
        # Common tools already registered
        pass
    
    def run(self, task: AgentTask) -> AgentResponse:
        """
        Run tests on the model pipeline.
        
        Args:
            task: AgentTask with task_type="test"
            
        Returns:
            AgentResponse with test results
        """
        logs = []
        logs.append("=== Test Agent Started ===")
        
        model_dir = task.paths.get("model_dir")
        model_code_path = task.paths.get("model_code")
        data_adapter_path = task.paths.get("data_adapter")
        feature_dict_path = task.paths.get("feature_dictionary")
        test_results_path = task.paths.get("test_results")
        
        data_dir = task.data_dir
        y_column = task.y_column
        tickers = []
        
        # Load tickers from feature dictionary
        feature_dict_result = self.read_file(feature_dict_path)
        if feature_dict_result["success"]:
            feature_dict = feature_dict_result.get("parsed", {})
            tickers = feature_dict.get("tickers", [])
        
        model_provided = task.context.get("model_provided", False)
        
        test_results = {
            "import_test": {"passed": False, "error": None},
            "data_load_test": {"passed": False, "error": None},
            "transform_test": {"passed": False, "error": None},
            "train_test": {"passed": False, "error": None},
            "predict_test": {"passed": False, "error": None},
            "overall": {"passed": False, "error": None}
        }
        
        # ---------------------------------------------------------------------
        # Test 1: Import Test
        # ---------------------------------------------------------------------
        logs.append("\n--- Test 1: Import Test ---")
        
        import_test_code = f'''
import sys
sys.path.insert(0, "{model_dir}")
sys.path.insert(0, "{os.path.dirname(model_code_path)}")

# Try to import model
model_imported = False
model_error = None
try:
    # Get model filename without extension
    model_file = "{os.path.basename(model_code_path).replace('.py', '')}"
    model_module = __import__(model_file)
    model_imported = True
except Exception as e:
    model_error = str(e)

# Try to import data adapter
adapter_imported = False
adapter_error = None
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_adapter", "{data_adapter_path}")
    adapter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter_module)
    adapter_imported = True
except Exception as e:
    adapter_error = str(e)

result = {{
    "model_imported": model_imported,
    "model_error": model_error,
    "adapter_imported": adapter_imported,
    "adapter_error": adapter_error
}}
'''
        
        import_result = self.execute_python(import_test_code)
        
        if import_result["success"] and import_result.get("result"):
            result_data = import_result["result"]
            
            if result_data.get("model_imported") and result_data.get("adapter_imported"):
                test_results["import_test"]["passed"] = True
                logs.append("✓ Import test passed")
            else:
                if not result_data.get("model_imported"):
                    error = f"Model import failed: {result_data.get('model_error')}"
                    test_results["import_test"]["error"] = error
                    logs.append(f"✗ {error}")
                if not result_data.get("adapter_imported"):
                    error = f"Adapter import failed: {result_data.get('adapter_error')}"
                    test_results["import_test"]["error"] = error
                    logs.append(f"✗ {error}")
        else:
            test_results["import_test"]["error"] = import_result.get("error", "Unknown error")
            logs.append(f"✗ Import test failed: {import_result.get('error')}")
        
        if not test_results["import_test"]["passed"]:
            # Can't continue if imports fail
            test_results["overall"]["error"] = "Import test failed"
            self._save_test_results(test_results_path, test_results)
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Import test failed: {test_results['import_test']['error']}",
                error=test_results["import_test"]["error"],
                output_data=test_results,
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # Test 2: Data Load Test
        # ---------------------------------------------------------------------
        logs.append("\n--- Test 2: Data Load Test ---")
        
        data_load_code = f'''
import sys
import importlib.util

# Import data adapter
spec = importlib.util.spec_from_file_location("data_adapter", "{data_adapter_path}")
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)

# Try to create adapter and load data
try:
    adapter = adapter_module.DataAdapter(
        data_dir="{data_dir}",
        tickers={tickers},
        y_column="{y_column}"
    )
    
    # Try to load features
    adapter.load_features()
    
    result = {{
        "success": True,
        "error": None
    }}
except Exception as e:
    import traceback
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
'''
        
        load_result = self.execute_python(data_load_code)
        
        if load_result["success"] and load_result.get("result", {}).get("success"):
            test_results["data_load_test"]["passed"] = True
            logs.append("✓ Data load test passed")
        else:
            error = load_result.get("result", {}).get("error") or load_result.get("error")
            test_results["data_load_test"]["error"] = error
            logs.append(f"✗ Data load test failed: {error}")
            
            # Can't continue
            test_results["overall"]["error"] = "Data load test failed"
            self._save_test_results(test_results_path, test_results)
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Data load test failed: {error}",
                error=error,
                output_data=test_results,
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # Test 3: Transform Test
        # ---------------------------------------------------------------------
        logs.append("\n--- Test 3: Transform Test ---")
        
        transform_code = f'''
import sys
import importlib.util

# Import data adapter
spec = importlib.util.spec_from_file_location("data_adapter", "{data_adapter_path}")
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)

try:
    adapter = adapter_module.DataAdapter(
        data_dir="{data_dir}",
        tickers={tickers},
        y_column="{y_column}"
    )
    adapter.load_features()
    
    # Transform data
    transformed_df = adapter.transform()
    
    result = {{
        "success": True,
        "shape": transformed_df.shape if hasattr(transformed_df, 'shape') else None,
        "columns": list(transformed_df.columns)[:20] if hasattr(transformed_df, 'columns') else None,
        "error": None
    }}
except Exception as e:
    import traceback
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
'''
        
        transform_result = self.execute_python(transform_code)
        
        if transform_result["success"] and transform_result.get("result", {}).get("success"):
            test_results["transform_test"]["passed"] = True
            result_data = transform_result.get("result", {})
            logs.append(f"✓ Transform test passed")
            logs.append(f"  Shape: {result_data.get('shape')}")
            logs.append(f"  Columns (first 20): {result_data.get('columns')}")
        else:
            error = transform_result.get("result", {}).get("error") or transform_result.get("error")
            tb = transform_result.get("result", {}).get("traceback", "")
            test_results["transform_test"]["error"] = error
            logs.append(f"✗ Transform test failed: {error}")
            if tb:
                logs.append(f"Traceback:\n{tb[:500]}")
            
            # Can't continue
            test_results["overall"]["error"] = "Transform test failed"
            self._save_test_results(test_results_path, test_results)
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Transform test failed: {error}",
                error=error,
                output_data=test_results,
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # Test 4: Train Test (with sample data)
        # ---------------------------------------------------------------------
        logs.append("\n--- Test 4: Train Test (sample) ---")
        
        # Get model class name
        model_file = os.path.basename(model_code_path).replace('.py', '')
        
        train_code = f'''
import sys
import importlib.util
import numpy as np

# Import model
sys.path.insert(0, "{os.path.dirname(model_code_path)}")
model_module = __import__("{model_file}")

# Import data adapter
spec = importlib.util.spec_from_file_location("data_adapter", "{data_adapter_path}")
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)

try:
    # Load and transform data
    adapter = adapter_module.DataAdapter(
        data_dir="{data_dir}",
        tickers={tickers},
        y_column="{y_column}"
    )
    adapter.load_features()
    df = adapter.transform()
    
    # Use small sample for testing
    sample_size = min({self.config.test_sample_size}, len(df))
    df_sample = df.head(sample_size)
    
    # Get train/test split if available
    if hasattr(adapter, 'get_train_test_split'):
        train_df, test_df = adapter.get_train_test_split(test_ratio=0.2)
        train_df = train_df.head(sample_size)
    else:
        train_df = df_sample
    
    # Try to instantiate and train model
    if hasattr(model_module, 'Model'):
        model = model_module.Model()
    else:
        # Try to find any class
        for name in dir(model_module):
            obj = getattr(model_module, name)
            if isinstance(obj, type) and name != 'type':
                model = obj()
                break
    
    # Train
    train_result = model.train(train_df)
    
    result = {{
        "success": True,
        "train_result": str(train_result)[:500] if train_result else None,
        "error": None
    }}
except Exception as e:
    import traceback
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
'''
        
        train_result = self.execute_python(train_code)
        
        if train_result["success"] and train_result.get("result", {}).get("success"):
            test_results["train_test"]["passed"] = True
            logs.append("✓ Train test passed")
            logs.append(f"  Result: {train_result.get('result', {}).get('train_result', '')[:200]}")
        else:
            error = train_result.get("result", {}).get("error") or train_result.get("error")
            tb = train_result.get("result", {}).get("traceback", "")
            test_results["train_test"]["error"] = error
            logs.append(f"✗ Train test failed: {error}")
            if tb:
                logs.append(f"Traceback:\n{tb[:500]}")
            
            # Analyze error source
            error_analysis = self._analyze_error(error, tb, model_provided)
            logs.append(f"Error analysis: {error_analysis}")
            
            test_results["overall"]["error"] = "Train test failed"
            self._save_test_results(test_results_path, test_results)
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Train test failed: {error}",
                error=error,
                output_data={**test_results, "error_analysis": error_analysis},
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # Test 5: Predict Test
        # ---------------------------------------------------------------------
        logs.append("\n--- Test 5: Predict Test ---")
        
        predict_code = f'''
import sys
import importlib.util
import numpy as np

# Import model
sys.path.insert(0, "{os.path.dirname(model_code_path)}")
model_module = __import__("{model_file}")

# Import data adapter
spec = importlib.util.spec_from_file_location("data_adapter", "{data_adapter_path}")
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)

try:
    # Load and transform data
    adapter = adapter_module.DataAdapter(
        data_dir="{data_dir}",
        tickers={tickers},
        y_column="{y_column}"
    )
    adapter.load_features()
    df = adapter.transform()
    
    # Use small sample
    sample_size = min({self.config.test_sample_size}, len(df))
    df_sample = df.head(sample_size)
    
    # Instantiate and train model
    if hasattr(model_module, 'Model'):
        model = model_module.Model()
    else:
        for name in dir(model_module):
            obj = getattr(model_module, name)
            if isinstance(obj, type) and name != 'type':
                model = obj()
                break
    
    model.train(df_sample)
    
    # Predict on last few rows
    predict_df = df_sample.tail(10)
    prediction = model.predict(predict_df)
    
    result = {{
        "success": True,
        "prediction": str(prediction)[:500] if prediction else None,
        "error": None
    }}
except Exception as e:
    import traceback
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
'''
        
        predict_result = self.execute_python(predict_code)
        
        if predict_result["success"] and predict_result.get("result", {}).get("success"):
            test_results["predict_test"]["passed"] = True
            logs.append("✓ Predict test passed")
            logs.append(f"  Prediction: {predict_result.get('result', {}).get('prediction', '')[:200]}")
        else:
            error = predict_result.get("result", {}).get("error") or predict_result.get("error")
            tb = predict_result.get("result", {}).get("traceback", "")
            test_results["predict_test"]["error"] = error
            logs.append(f"✗ Predict test failed: {error}")
            if tb:
                logs.append(f"Traceback:\n{tb[:500]}")
            
            error_analysis = self._analyze_error(error, tb, model_provided)
            logs.append(f"Error analysis: {error_analysis}")
            
            test_results["overall"]["error"] = "Predict test failed"
            self._save_test_results(test_results_path, test_results)
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Predict test failed: {error}",
                error=error,
                output_data={**test_results, "error_analysis": error_analysis},
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # All tests passed!
        # ---------------------------------------------------------------------
        test_results["overall"]["passed"] = True
        logs.append("\n=== All Tests Passed! ===")
        
        self._save_test_results(test_results_path, test_results)
        
        return self.create_response(
            success=True,
            status="COMPLETED",
            message="All tests passed successfully",
            generated_files={"test_results": test_results_path},
            output_data=test_results,
            logs=logs
        )
    
    def _analyze_error(
        self,
        error: str,
        traceback_str: str,
        model_provided: bool
    ) -> Dict[str, Any]:
        """
        Analyze an error to determine its likely source.
        
        Args:
            error: Error message
            traceback_str: Full traceback
            model_provided: Whether model was provided by user
            
        Returns:
            Dict with analysis results
        """
        analysis = {
            "likely_source": "unknown",
            "confidence": "low",
            "suggestion": ""
        }
        
        error_lower = error.lower()
        tb_lower = traceback_str.lower()
        
        # Check for data adapter issues
        if "data_adapter" in tb_lower or "transform" in tb_lower:
            analysis["likely_source"] = "data_adapter"
            analysis["confidence"] = "high"
            analysis["suggestion"] = "Check data_adapter.py for column naming or transformation issues"
        
        # Check for shape mismatch (common integration issue)
        elif "shape" in error_lower or "dimension" in error_lower:
            analysis["likely_source"] = "integration"
            analysis["confidence"] = "medium"
            analysis["suggestion"] = "Data shape doesn't match model expectations. Check adapter config."
        
        # Check for missing columns
        elif "keyerror" in error_lower or "column" in error_lower:
            analysis["likely_source"] = "data_adapter"
            analysis["confidence"] = "high"
            analysis["suggestion"] = "Missing expected column. Check column naming in data_adapter.py"
        
        # Check for model-specific errors
        elif "model" in tb_lower and "train" in tb_lower:
            analysis["likely_source"] = "model_code"
            analysis["confidence"] = "medium"
            if model_provided:
                analysis["suggestion"] = "Error in provided model code. Please check model implementation."
            else:
                analysis["suggestion"] = "Error in generated model code. Will attempt to fix."
        
        # Import errors
        elif "import" in error_lower or "module" in error_lower:
            analysis["likely_source"] = "import"
            analysis["confidence"] = "high"
            analysis["suggestion"] = "Missing dependency or import error. Check required packages."
        
        return analysis
    
    def _save_test_results(self, path: str, results: Dict) -> None:
        """Save test results to JSON file."""
        try:
            self.write_file(path, json.dumps(results, indent=2))
        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")
