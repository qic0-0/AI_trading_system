"""
MCP Brain - The coordinator for the Quant Model Agent System.

The Brain orchestrates the entire workflow:
1. Validate inputs and check for provided files
2. Call Model Code Agent (if model not provided)
3. Call Data Adapt Agent (analyze → wait for user if issues → generate code)
4. Call Test Agent
5. Handle errors and route to appropriate agent for fixes

The Brain is LLM-powered - it uses reasoning to decide what to do next.
"""

import os
import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from .mcp_config import (
    MCPConfig, MCPResult, AgentTask, AgentResponse,
    get_model_paths, get_feature_dictionary_path, find_provided_files
)

logger = logging.getLogger(__name__)


class MCPBrain:
    """
    LLM-powered coordinator for the MCP Quant Agent System.
    
    Workflow:
    1. run() - Start fresh workflow
    2. If issues found, returns PAUSED with issues
    3. resume() - Continue with user overrides
    
    The Brain decides which agents to call based on:
    - What files exist (model provided vs need to generate)
    - What issues arise (blocking vs non-blocking)
    - What errors occur (which agent should fix)
    """
    
    def __init__(self, llm_client, config: Optional[MCPConfig] = None):
        """
        Initialize MCP Brain.
        
        Args:
            llm_client: LLMClient instance for LLM calls
            config: MCPConfig instance (uses defaults if None)
        """
        self.llm_client = llm_client
        self.config = config or MCPConfig()
        self.logger = logging.getLogger("MCPBrain")
        
        # Session state for pause/resume
        self._sessions: Dict[str, Dict] = {}
        
        # Initialize sub-agents (lazy loading)
        self._model_code_agent = None
        self._data_adapt_agent = None
        self._test_agent = None
    
    @property
    def model_code_agent(self):
        """Lazy load Model Code Agent."""
        if self._model_code_agent is None:
            from .model_code_agent import ModelCodeAgent
            self._model_code_agent = ModelCodeAgent(self.llm_client, self.config)
        return self._model_code_agent
    
    @property
    def data_adapt_agent(self):
        """Lazy load Data Adapt Agent."""
        if self._data_adapt_agent is None:
            from .data_adapt_agent import DataAdaptAgent
            self._data_adapt_agent = DataAdaptAgent(self.llm_client, self.config)
        return self._data_adapt_agent
    
    @property
    def test_agent(self):
        """Lazy load Test Agent."""
        if self._test_agent is None:
            from .test_agent import TestAgent
            self._test_agent = TestAgent(self.llm_client, self.config)
        return self._test_agent
    
    @property
    def system_prompt(self) -> str:
        """System prompt for Brain's LLM reasoning."""
        return """You are the MCP Brain - a coordinator for a quantitative model building system.

Your role is to:
1. Analyze the current state (what files exist, what's needed)
2. Decide which agent to call next
3. Handle errors and route them appropriately
4. Communicate clearly with the user

You coordinate three sub-agents:
- Model Code Agent: Writes model.py and usage_guide.md from model_description.md
- Data Adapt Agent: Writes data_adapter.py to transform features for the model
- Test Agent: Tests the complete pipeline and reports errors

Key principles:
- If model code is provided by user, skip Model Code Agent
- Always run Data Adapt Agent (even with provided models, data needs transformation)
- If blocking issues found, PAUSE and ask user
- If tests fail, analyze error and route to appropriate agent (or user if provided model)
- Be concise and clear in your decisions"""
    
    def run(
        self,
        model_name: str,
        data_dir: str,
        y_column: str,
        prediction_horizon: int = 1,
        user_overrides: Optional[Dict[str, Any]] = None
    ) -> MCPResult:
        """
        Run the MCP workflow from start.
        
        Args:
            model_name: Name of model (folder under models/)
            data_dir: Path to data directory (contains features/)
            y_column: Name of the Y column (target variable)
            prediction_horizon: How many steps ahead to predict
            user_overrides: Optional overrides for data adaptation
            
        Returns:
            MCPResult with status, issues, and generated files
        """
        session_id = str(uuid.uuid4())[:8]
        logs = []
        logs.append(f"=== MCP Run Started (Session: {session_id}) ===")
        logs.append(f"Model: {model_name}")
        logs.append(f"Data dir: {data_dir}")
        logs.append(f"Y column: {y_column}")
        logs.append(f"Prediction horizon: {prediction_horizon}")
        
        # Get paths
        model_paths = get_model_paths(self.config.models_dir, model_name)
        feature_dict_path = get_feature_dictionary_path(data_dir)
        
        # ---------------------------------------------------------------------
        # Step 1: Validate inputs
        # ---------------------------------------------------------------------
        logs.append("\n=== Step 1: Validating Inputs ===")
        
        validation_result = self._validate_inputs(
            model_name, data_dir, model_paths, feature_dict_path
        )
        
        if not validation_result["valid"]:
            logs.extend(validation_result["logs"])
            return MCPResult(
                status="FAILED",
                message=validation_result["error"],
                logs=logs
            )
        
        logs.extend(validation_result["logs"])
        provided_files = validation_result["provided_files"]
        
        # ---------------------------------------------------------------------
        # Step 2: Model Code Agent (if needed)
        # ---------------------------------------------------------------------
        logs.append("\n=== Step 2: Model Code ===")
        
        if "model_code" in provided_files and "usage_guide" in provided_files:
            logs.append("Model code and usage guide provided by user - skipping Model Code Agent")
            model_code_path = provided_files["model_code"]
            usage_guide_path = provided_files["usage_guide"]
        elif "model_code" in provided_files:
            logs.append("Model code provided but no usage guide - Model Code Agent will generate usage guide")
            # TODO: Have Model Code Agent analyze provided code and generate usage guide
            logs.append("ERROR: Usage guide required for provided model")
            return MCPResult(
                status="FAILED",
                message="Model code provided but usage_guide.md is missing. Please provide a usage guide.",
                logs=logs
            )
        else:
            logs.append("No model code provided - calling Model Code Agent")
            
            model_task = AgentTask(
                task_type="write_model",
                model_name=model_name,
                data_dir=data_dir,
                paths=model_paths
            )
            
            model_response = self.model_code_agent.run(model_task)
            logs.extend(model_response.logs)
            
            if not model_response.success:
                return MCPResult(
                    status="FAILED",
                    message=f"Model Code Agent failed: {model_response.message}",
                    logs=logs
                )
            
            model_code_path = model_response.generated_files.get("model_code")
            usage_guide_path = model_response.generated_files.get("usage_guide")
        
        # ---------------------------------------------------------------------
        # Step 3: Data Adapt Agent (Phase 1 - Analysis)
        # ---------------------------------------------------------------------
        logs.append("\n=== Step 3: Data Adaptation Analysis ===")
        
        adapt_task = AgentTask(
            task_type="analyze_data",
            model_name=model_name,
            data_dir=data_dir,
            y_column=y_column,
            prediction_horizon=prediction_horizon,
            user_overrides=user_overrides,
            paths={
                **model_paths,
                "model_code": model_code_path,
                "usage_guide": usage_guide_path,
                "feature_dictionary": feature_dict_path
            },
            context={
                "model_provided": "model_code" in provided_files
            }
        )
        
        adapt_response = self.data_adapt_agent.run(adapt_task)
        logs.extend(adapt_response.logs)
        
        # Check for blocking issues
        if adapt_response.blocking_issues:
            logs.append(f"\nFound {len(adapt_response.blocking_issues)} blocking issue(s)")
            
            # Save session for resume
            self._sessions[session_id] = {
                "model_name": model_name,
                "data_dir": data_dir,
                "y_column": y_column,
                "prediction_horizon": prediction_horizon,
                "model_code_path": model_code_path,
                "usage_guide_path": usage_guide_path,
                "model_paths": model_paths,
                "feature_dict_path": feature_dict_path,
                "provided_files": provided_files,
                "adapt_response": adapt_response
            }
            
            return MCPResult(
                status="PAUSED",
                message="Blocking issues found. Please review and provide overrides.",
                issues=adapt_response.blocking_issues,
                warnings=adapt_response.warnings,
                config_path=model_paths["adapter_config"],
                reasoning_path=model_paths["reasoning"],
                session_id=session_id,
                logs=logs,
                partial_work={
                    "adapter_config.json": "Created (may need edits)",
                    "transformation_reasoning.md": "Created"
                }
            )
        
        if not adapt_response.success:
            return MCPResult(
                status="FAILED",
                message=f"Data Adapt Agent failed: {adapt_response.message}",
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # Step 4: Data Adapt Agent (Phase 2 - Code Generation)
        # ---------------------------------------------------------------------
        logs.append("\n=== Step 4: Data Adapter Code Generation ===")
        
        # Phase 2 task
        adapt_task_phase2 = AgentTask(
            task_type="generate_adapter",
            model_name=model_name,
            data_dir=data_dir,
            y_column=y_column,
            prediction_horizon=prediction_horizon,
            user_overrides=user_overrides,
            paths={
                **model_paths,
                "model_code": model_code_path,
                "usage_guide": usage_guide_path,
                "feature_dictionary": feature_dict_path
            },
            context={
                "model_provided": "model_code" in provided_files,
                "phase": 2,
                "adapter_config": adapt_response.output_data.get("adapter_config")
            }
        )
        
        adapt_response_phase2 = self.data_adapt_agent.run(adapt_task_phase2)
        logs.extend(adapt_response_phase2.logs)
        
        if not adapt_response_phase2.success:
            return MCPResult(
                status="FAILED",
                message=f"Data Adapter code generation failed: {adapt_response_phase2.message}",
                logs=logs
            )
        
        # ---------------------------------------------------------------------
        # Step 5: Test Agent
        # ---------------------------------------------------------------------
        logs.append("\n=== Step 5: Testing ===")
        
        test_task = AgentTask(
            task_type="test",
            model_name=model_name,
            data_dir=data_dir,
            y_column=y_column,
            prediction_horizon=prediction_horizon,
            paths={
                **model_paths,
                "model_code": model_code_path,
                "usage_guide": usage_guide_path,
                "feature_dictionary": feature_dict_path,
                "data_adapter": model_paths["data_adapter"]
            },
            context={
                "model_provided": "model_code" in provided_files
            }
        )
        
        test_response = self.test_agent.run(test_task)
        logs.extend(test_response.logs)
        
        # Retry loop for adapter errors
        fix_attempts = 0
        while not test_response.success and fix_attempts < self.config.max_fix_iterations:
            # Analyze error and decide what to do
            error_analysis = self._analyze_test_error(
                test_response,
                "model_code" in provided_files
            )
            logs.extend(error_analysis["logs"])

            if error_analysis["action"] == "report_to_user":
                # Can't auto-fix, report to user
                return MCPResult(
                    status="FAILED",
                    message=error_analysis["message"],
                    issues=[{
                        "type": "test_failure",
                        "severity": "blocking",
                        "message": error_analysis["message"],
                        "error": test_response.error,
                        "suggestion": error_analysis["suggestion"]
                    }],
                    test_results=test_response.output_data,
                    logs=logs
                )

            elif error_analysis["action"] == "retry_data_adapt":
                # Try to fix the data adapter
                fix_attempts += 1
                logs.append(f"\n=== Auto-fix Attempt {fix_attempts}/{self.config.max_fix_iterations} ===")
                logs.append(f"Sending error to Data Adapt Agent for fix...")

                # Call Data Adapt Agent with error context
                fix_task = AgentTask(
                    task_type="fix_adapter",
                    model_name=model_name,
                    data_dir=data_dir,
                    y_column=y_column,
                    prediction_horizon=prediction_horizon,
                    context={
                        "error": test_response.error,
                        "adapter_config_path": model_paths["adapter_config"],
                        "data_adapter_path": model_paths["data_adapter"],
                        "usage_guide_path": usage_guide_path,
                        "feature_dict_path": feature_dict_path,
                        "fix_attempt": fix_attempts
                    }
                )

                fix_response = self.data_adapt_agent.fix_adapter(fix_task)
                logs.extend(fix_response.logs)

                if not fix_response.success:
                    logs.append(f"Fix attempt {fix_attempts} failed: {fix_response.error}")
                    continue

                logs.append(f"Data adapter regenerated, re-running tests...")

                # Re-run tests
                test_response = self.test_agent.run(test_task)
                logs.extend(test_response.logs)

            else:
                # Unknown action, break
                logs.append(f"Unknown action: {error_analysis['action']}, stopping")
                break

        # Check if we exhausted retries
        if not test_response.success:
            return MCPResult(
                status="FAILED",
                message=f"Test failed after {fix_attempts} fix attempts: {test_response.error}",
                issues=[{
                    "type": "test_failure",
                    "severity": "blocking",
                    "message": f"Could not fix adapter after {fix_attempts} attempts",
                    "error": test_response.error,
                    "suggestion": "Manual review of data_adapter.py required"
                }],
                test_results=test_response.output_data,
                logs=logs
            )

        # ---------------------------------------------------------------------
        # Success!
        # ---------------------------------------------------------------------
        logs.append("\n=== MCP Run Complete ===")

        return MCPResult(
            status="SUCCESS",
            message="Model pipeline created and tested successfully",
            generated_files={
                "model_code": model_code_path,
                "usage_guide": usage_guide_path,
                "adapter_config": model_paths["adapter_config"],
                "data_adapter": model_paths["data_adapter"],
                "reasoning": model_paths["reasoning"],
                "test_results": model_paths["test_results"]
            },
            test_results=test_response.output_data,
            logs=logs,
            session_id=session_id
        )

    def resume(
        self,
        session_id: str,
        user_overrides: Dict[str, Any]
    ) -> MCPResult:
        """
        Resume a paused workflow with user overrides.

        Args:
            session_id: Session ID from paused result
            user_overrides: User's decisions for blocking issues

        Returns:
            MCPResult with final status
        """
        logs = []
        logs.append(f"=== MCP Resume (Session: {session_id}) ===")

        # Get session state
        if session_id not in self._sessions:
            return MCPResult(
                status="FAILED",
                message=f"Session not found: {session_id}",
                logs=logs
            )

        session = self._sessions[session_id]

        # Merge overrides with existing config
        logs.append("Applying user overrides...")

        # Update adapter_config.json with overrides
        adapter_config_path = session["model_paths"]["adapter_config"]
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            # Deep merge overrides
            adapter_config = self._deep_merge(adapter_config, user_overrides)

            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f, indent=2)

            logs.append(f"Updated {adapter_config_path}")

        # Continue from Phase 2
        logs.append("\n=== Continuing: Data Adapter Code Generation ===")

        adapt_task_phase2 = AgentTask(
            task_type="generate_adapter",
            model_name=session["model_name"],
            data_dir=session["data_dir"],
            y_column=session["y_column"],
            prediction_horizon=session["prediction_horizon"],
            user_overrides=user_overrides,
            paths={
                **session["model_paths"],
                "model_code": session["model_code_path"],
                "usage_guide": session["usage_guide_path"],
                "feature_dictionary": session["feature_dict_path"]
            },
            context={
                "model_provided": "model_code" in session["provided_files"],
                "phase": 2,
                "from_resume": True
            }
        )

        adapt_response = self.data_adapt_agent.run(adapt_task_phase2)
        logs.extend(adapt_response.logs)

        if not adapt_response.success:
            return MCPResult(
                status="FAILED",
                message=f"Data Adapter code generation failed: {adapt_response.message}",
                logs=logs
            )

        # Test
        logs.append("\n=== Testing ===")

        test_task = AgentTask(
            task_type="test",
            model_name=session["model_name"],
            data_dir=session["data_dir"],
            y_column=session["y_column"],
            prediction_horizon=session["prediction_horizon"],
            paths={
                **session["model_paths"],
                "model_code": session["model_code_path"],
                "usage_guide": session["usage_guide_path"],
                "feature_dictionary": session["feature_dict_path"],
                "data_adapter": session["model_paths"]["data_adapter"]
            },
            context={
                "model_provided": "model_code" in session["provided_files"]
            }
        )

        test_response = self.test_agent.run(test_task)
        logs.extend(test_response.logs)

        if not test_response.success:
            error_analysis = self._analyze_test_error(
                test_response,
                "model_code" in session["provided_files"]
            )
            logs.extend(error_analysis["logs"])

            return MCPResult(
                status="FAILED",
                message=error_analysis["message"],
                issues=[{
                    "type": "test_failure",
                    "severity": "blocking",
                    "message": error_analysis["message"],
                    "error": test_response.error,
                    "suggestion": error_analysis["suggestion"]
                }],
                test_results=test_response.output_data,
                logs=logs
            )

        # Success!
        logs.append("\n=== MCP Resume Complete ===")

        # Clean up session
        del self._sessions[session_id]

        return MCPResult(
            status="SUCCESS",
            message="Model pipeline created and tested successfully",
            generated_files={
                "model_code": session["model_code_path"],
                "usage_guide": session["usage_guide_path"],
                "adapter_config": session["model_paths"]["adapter_config"],
                "data_adapter": session["model_paths"]["data_adapter"],
                "reasoning": session["model_paths"]["reasoning"],
                "test_results": session["model_paths"]["test_results"]
            },
            test_results=test_response.output_data,
            logs=logs
        )

    def _validate_inputs(
        self,
        model_name: str,
        data_dir: str,
        model_paths: Dict[str, str],
        feature_dict_path: str
    ) -> Dict[str, Any]:
        """
        Validate that required inputs exist.

        Returns:
            Dict with valid status, error message, logs, and provided files
        """
        logs = []

        # Check model directory exists
        model_dir = model_paths["model_dir"]
        if not os.path.exists(model_dir):
            logs.append(f"Model directory not found: {model_dir}")
            return {
                "valid": False,
                "error": f"Model directory not found: {model_dir}. Please create it first.",
                "logs": logs,
                "provided_files": {}
            }
        logs.append(f"✓ Model directory exists: {model_dir}")

        # Find provided files
        provided_files = find_provided_files(model_dir)
        logs.append(f"Found provided files: {list(provided_files.keys())}")

        # Check model_description.md exists (required unless model code is provided)
        if "description" not in provided_files and "model_code" not in provided_files:
            logs.append("Neither model_description.md nor model code found")
            return {
                "valid": False,
                "error": "model_description.md required (or provide model code with usage guide)",
                "logs": logs,
                "provided_files": provided_files
            }

        if "description" in provided_files:
            logs.append(f"✓ Model description found: {provided_files['description']}")

        if "model_code" in provided_files:
            logs.append(f"✓ Model code found: {provided_files['model_code']}")

        if "usage_guide" in provided_files:
            logs.append(f"✓ Usage guide found: {provided_files['usage_guide']}")

        # Check data directory exists
        if not os.path.exists(data_dir):
            logs.append(f"Data directory not found: {data_dir}")
            return {
                "valid": False,
                "error": f"Data directory not found: {data_dir}",
                "logs": logs,
                "provided_files": provided_files
            }
        logs.append(f"✓ Data directory exists: {data_dir}")

        # Check feature dictionary exists
        if not os.path.exists(feature_dict_path):
            logs.append(f"Feature dictionary not found: {feature_dict_path}")
            return {
                "valid": False,
                "error": f"Feature dictionary not found: {feature_dict_path}. Run Feature Agent first.",
                "logs": logs,
                "provided_files": provided_files
            }
        logs.append(f"✓ Feature dictionary found: {feature_dict_path}")

        return {
            "valid": True,
            "error": None,
            "logs": logs,
            "provided_files": provided_files
        }

    def _analyze_test_error(
        self,
        test_response: AgentResponse,
        model_provided: bool
    ) -> Dict[str, Any]:
        """
        Analyze test error and decide what to do.

        Args:
            test_response: Response from Test Agent
            model_provided: Whether model was provided by user

        Returns:
            Dict with action, message, suggestion, and logs
        """
        logs = []
        error = test_response.error or "Unknown error"
        error_lower = error.lower()

        logs.append(f"Analyzing test error: {error[:200]}...")

        # Check if traceback mentions data_adapter.py - strongest signal
        if "data_adapter.py" in error or "data_adapter" in error_lower:
            logs.append("Error traceback contains data_adapter.py - this is an adapter error")
            return {
                "action": "retry_data_adapt",
                "message": f"Data adapter error: {error[:200]}",
                "suggestion": "Regenerating data_adapter.py with error context",
                "logs": logs
            }

        # Patterns that indicate data adapter errors (not model errors)
        adapter_error_patterns = [
            "transform",
            "shape of passed values",  # pandas DataFrame shape mismatch
            "shape mismatch",
            "cannot reshape",
            "indexerror",
            "keyerror",
            "parquet",
            "load_features",
            "valueerror: cannot",
            "dataframe",
            "column",
            "broadcasting",
            "aggregation",
            "resample",
        ]

        # Patterns that indicate model code errors
        model_error_patterns = [
            "model import failed",
            "model.py",
            "train() failed",
            "predict() failed",
            "model.train",
            "model.predict",
        ]

        # Check for adapter errors
        is_adapter_error = any(pattern in error_lower for pattern in adapter_error_patterns)
        is_model_error = any(pattern in error_lower for pattern in model_error_patterns)

        # If it looks like an adapter error, try to fix it
        if is_adapter_error and not is_model_error:
            logs.append("Error appears to be in data adapter (shape/transform issue)")
            return {
                "action": "retry_data_adapt",
                "message": f"Data adapter error: {error[:200]}",
                "suggestion": "Regenerating data_adapter.py with error context",
                "logs": logs
            }

        # If model was provided by user, we can't fix model code
        if model_provided:
            logs.append("Model was provided by user - cannot auto-fix model code")

            # Even if it looks like a model error, check if adapter could be the cause
            if "shape" in error_lower or "dimension" in error_lower or "key" in error_lower:
                logs.append("Shape/dimension/key error - could be adapter output format issue")
                return {
                    "action": "retry_data_adapt",
                    "message": f"Possible adapter issue: {error[:200]}",
                    "suggestion": "Check adapter output format matches model input format",
                    "logs": logs
                }

            logs.append("Error appears to be in provided model")
            return {
                "action": "report_to_user",
                "message": f"Test failed with provided model: {error[:200]}",
                "suggestion": "Please check your model code and usage guide",
                "logs": logs
            }

        # Model was generated by us - we can try to fix
        logs.append("Model was generated - can attempt auto-fix")

        # For now, try data adapter fix first (most common source of errors)
        return {
            "action": "retry_data_adapt",
            "message": f"Test failed: {error[:200]}",
            "suggestion": "Attempting to regenerate data adapter",
            "logs": logs
        }
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result