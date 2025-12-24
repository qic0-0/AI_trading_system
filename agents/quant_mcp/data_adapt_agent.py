"""
Data Adapt Agent - Transforms features to model's expected format.

This is the most complex agent because it must:
1. Analyze source data format (from feature_dictionary.json)
2. Analyze target format (from usage_guide.md)
3. Reason about how to transform (aggregation, window shift, etc.)
4. Identify blocking issues that need user input
5. Generate transformation code (data_adapter.py)

Two-phase operation:
- Phase 1 (analyze_data): Analyze and identify issues, create adapter_config.json
- Phase 2 (generate_adapter): Generate data_adapter.py from config
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

from .base_mcp_agent import BaseMCPAgent
from .mcp_config import AgentTask, AgentResponse

logger = logging.getLogger(__name__)


class DataAdaptAgent(BaseMCPAgent):
    """
    Agent that transforms feature data to model's expected format.
    
    Key responsibilities:
    - Parse feature_dictionary.json to understand source data
    - Parse usage_guide.md to understand target format
    - Reason about appropriate transformations
    - Handle time window shifting (X at t → Y at t+horizon)
    - Map embeddings to time indices
    - Report blocking issues to user
    """
    
    def __init__(self, llm_client, config):
        super().__init__("DataAdaptAgent", llm_client, config)
    
    @property
    def system_prompt(self) -> str:
        return """You are a Data Adapt Agent specialized in transforming financial time series data.

Your task is to analyze source data and target model requirements, then create transformation code.

Key concepts you must handle:

1. RESOLUTION CONVERSION
   - Source: Hourly data (e.g., 7 bars per trading day)
   - Target: May be daily, hourly, or other
   - Methods: last, first, mean, sum, min, max, expand_to_columns

2. TIME WINDOW SHIFTING (CRITICAL for forecasting)
   - Raw features have same time index as Y
   - For prediction: X at time t-horizon predicts Y at time t
   - IMPORTANT: Shift X forward, NOT Y backward!
   - Y must keep its true time index for proper backtesting
   - Example: X from day 1 predicts Y from day 2 → row at day 2 has X[1] and Y[2]

3. Y VALUE EXTRACTION
   - User specifies which factor is Y
   - All other factors become X (input features)
   - Y may need to be expanded to multiple columns (hourly predictions)

4. EMBEDDING HANDLING
   - Embeddings have no time index (static or monthly)
   - Must broadcast to all timestamps
   - Each embedding dimension becomes a column

5. COLUMN NAMING
   - Follow model's expected naming convention
   - e.g., x_{feat}_s{series}, emb_{e}_s{s}_d{dim}, y_{h}_s{s}

When analyzing:
- Compare source vs target format carefully
- Identify ALL mismatches
- For each mismatch, decide: can solve automatically OR need user input
- Create detailed reasoning document

Be precise about time indices and avoid any look-ahead bias!"""

    def _register_tools(self):
        """Register Data Adapt Agent specific tools."""
        # Common tools already registered in base class
        pass

    def run(self, task: AgentTask) -> AgentResponse:
        """
        Run data adaptation (Phase 1 or Phase 2).

        Args:
            task: AgentTask with task_type="analyze_data" or "generate_adapter"

        Returns:
            AgentResponse with results
        """
        if task.task_type == "analyze_data":
            return self._run_analysis(task)
        elif task.task_type == "generate_adapter":
            return self._run_generation(task)
        else:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Unknown task type: {task.task_type}",
                logs=[]
            )

    def _run_analysis(self, task: AgentTask) -> AgentResponse:
        """
        Phase 1: Analyze source and target formats, identify issues.

        Creates:
        - adapter_config.json (transformation plan)
        - transformation_reasoning.md (explains decisions)
        - issues_report.json (if blocking issues found)
        """
        logs = []
        logs.append("=== Data Adapt Agent - Phase 1: Analysis ===")

        model_dir = task.paths.get("model_dir")
        feature_dict_path = task.paths.get("feature_dictionary")
        usage_guide_path = task.paths.get("usage_guide")
        adapter_config_path = task.paths.get("adapter_config")
        reasoning_path = task.paths.get("reasoning")
        issues_path = task.paths.get("issues")

        y_column = task.y_column
        prediction_horizon = task.prediction_horizon
        user_overrides = task.user_overrides or {}

        # ---------------------------------------------------------------------
        # Step 1: Read feature dictionary
        # ---------------------------------------------------------------------
        logs.append("Reading feature dictionary...")

        feature_dict_result = self.read_file(feature_dict_path)
        if not feature_dict_result["success"]:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Failed to read feature dictionary: {feature_dict_result['error']}",
                logs=logs
            )

        feature_dict = feature_dict_result.get("parsed", {})
        if not feature_dict:
            return self.create_response(
                success=False,
                status="ERROR",
                message="Failed to parse feature dictionary as JSON",
                logs=logs
            )

        logs.append(f"Found {len(feature_dict.get('tickers', []))} tickers")
        logs.append(f"Datasets: {list(feature_dict.get('datasets', {}).keys())}")

        # ---------------------------------------------------------------------
        # Step 2: Read usage guide
        # ---------------------------------------------------------------------
        logs.append("Reading usage guide...")

        usage_guide_result = self.read_file(usage_guide_path)
        if not usage_guide_result["success"]:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Failed to read usage guide: {usage_guide_result['error']}",
                logs=logs
            )

        usage_guide = usage_guide_result["content"]
        logs.append(f"Read {len(usage_guide)} chars from usage guide")

        # ---------------------------------------------------------------------
        # Step 3: Validate Y column
        # ---------------------------------------------------------------------
        logs.append(f"Validating Y column: {y_column}")

        independent_factors = feature_dict.get("datasets", {}).get("independent_factors", {})
        available_factors = list(independent_factors.get("factors", {}).keys())

        if y_column not in available_factors:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Y column '{y_column}' not found in independent factors. Available: {available_factors}",
                logs=logs
            )

        logs.append(f"✓ Y column '{y_column}' found in independent factors")

        # ---------------------------------------------------------------------
        # Step 4: Use LLM to analyze and create transformation plan
        # ---------------------------------------------------------------------
        logs.append("Analyzing transformation requirements...")

        analysis_prompt = self._create_analysis_prompt(
            feature_dict=feature_dict,
            usage_guide=usage_guide,
            y_column=y_column,
            prediction_horizon=prediction_horizon,
            user_overrides=user_overrides
        )

        analysis_response = self.think(analysis_prompt)

        # Extract JSON config from response
        adapter_config = self.extract_json_from_response(analysis_response)

        if not adapter_config:
            logs.append("Failed to extract JSON from LLM response, attempting structured extraction...")
            # Try to get LLM to format as JSON
            format_prompt = f"""Convert the following analysis into a JSON configuration:

{analysis_response}

Return ONLY valid JSON with this structure:
{{
  "source_format": {{ ... }},
  "target_format": {{ ... }},
  "y_config": {{ ... }},
  "x_config": {{ ... }},
  "transformation_plan": {{ ... }},
  "blocking_issues": [ ... ],
  "warnings": [ ... ]
}}"""

            format_response = self.think(format_prompt)
            adapter_config = self.extract_json_from_response(format_response)

        if not adapter_config:
            # Create minimal config if LLM fails
            logs.append("Using fallback config generation")
            adapter_config = self._create_fallback_config(
                feature_dict, y_column, prediction_horizon
            )

        # ---------------------------------------------------------------------
        # Step 4.5: Auto-detect data properties if needed
        # ---------------------------------------------------------------------
        needs_detection = (
            adapter_config.get("source_format", {}).get("bars_per_day") == "auto_detect" or
            adapter_config.get("y_config", {}).get("output_columns") == "auto_detect" or
            adapter_config.get("target_format", {}).get("output_steps") == "auto_detect"
        )

        if needs_detection:
            logs.append("Auto-detecting data properties from actual files...")
            tickers = feature_dict.get("tickers", [])
            detected_props = self._detect_data_properties(task.data_dir, tickers)

            if detected_props.get("detection_error"):
                logs.append(f"Detection warning: {detected_props['detection_error']}")
            else:
                logs.append(f"Detected: bars_per_day={detected_props.get('bars_per_day')}, "
                           f"resolution={detected_props.get('resolution')}, "
                           f"trading_hours={detected_props.get('trading_hours')}")

                # Resolve auto_detect values
                adapter_config = self._resolve_auto_detect_values(adapter_config, detected_props)
                logs.append("Resolved auto_detect values with detected properties")

        logs.append("Generated adapter configuration")

        # ---------------------------------------------------------------------
        # Step 5: Generate reasoning document
        # ---------------------------------------------------------------------
        logs.append("Generating transformation reasoning document...")

        reasoning_prompt = self._create_reasoning_prompt(
            feature_dict=feature_dict,
            usage_guide=usage_guide,
            y_column=y_column,
            prediction_horizon=prediction_horizon,
            adapter_config=adapter_config
        )

        reasoning_doc = self.think(reasoning_prompt)

        # Save reasoning
        self.write_file(reasoning_path, reasoning_doc)
        logs.append(f"Saved reasoning to {reasoning_path}")

        # ---------------------------------------------------------------------
        # Step 6: Check for blocking issues
        # ---------------------------------------------------------------------
        blocking_issues = adapter_config.get("blocking_issues", [])
        warnings = adapter_config.get("warnings", [])

        # Add user overrides to config
        if user_overrides:
            adapter_config["user_overrides"] = user_overrides
            # Apply overrides to resolve some blocking issues
            blocking_issues = self._apply_overrides_to_issues(
                blocking_issues, user_overrides
            )
            adapter_config["blocking_issues"] = blocking_issues

        # Save adapter config
        self.write_file(adapter_config_path, json.dumps(adapter_config, indent=2))
        logs.append(f"Saved adapter config to {adapter_config_path}")

        # Save issues report if any
        if blocking_issues or warnings:
            issues_report = {
                "blocking_issues": blocking_issues,
                "warnings": warnings
            }
            self.write_file(issues_path, json.dumps(issues_report, indent=2))
            logs.append(f"Saved issues report to {issues_path}")

        logs.append(f"Found {len(blocking_issues)} blocking issues, {len(warnings)} warnings")

        if blocking_issues:
            return self.create_response(
                success=True,  # Analysis succeeded, but has blocking issues
                status="BLOCKED",
                message=f"Analysis complete but found {len(blocking_issues)} blocking issue(s)",
                blocking_issues=blocking_issues,
                warnings=warnings,
                generated_files={
                    "adapter_config": adapter_config_path,
                    "reasoning": reasoning_path,
                    "issues": issues_path
                },
                output_data={"adapter_config": adapter_config},
                logs=logs
            )

        logs.append("=== Analysis Phase Complete ===")

        return self.create_response(
            success=True,
            status="COMPLETED",
            message="Analysis complete, no blocking issues",
            warnings=warnings,
            generated_files={
                "adapter_config": adapter_config_path,
                "reasoning": reasoning_path
            },
            output_data={"adapter_config": adapter_config},
            logs=logs
        )

    def _run_generation(self, task: AgentTask) -> AgentResponse:
        """
        Phase 2: Generate data_adapter.py from adapter_config.json.
        """
        logs = []
        logs.append("=== Data Adapt Agent - Phase 2: Code Generation ===")

        adapter_config_path = task.paths.get("adapter_config")
        data_adapter_path = task.paths.get("data_adapter")
        feature_dict_path = task.paths.get("feature_dictionary")
        usage_guide_path = task.paths.get("usage_guide")

        y_column = task.y_column
        prediction_horizon = task.prediction_horizon

        # Read adapter config
        config_result = self.read_file(adapter_config_path)
        if not config_result["success"]:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Failed to read adapter config: {config_result['error']}",
                logs=logs
            )

        adapter_config = config_result.get("parsed", {})

        # Check if from resume and has user overrides
        if task.context.get("from_resume") and task.user_overrides:
            # Merge user overrides
            adapter_config = self._merge_overrides(adapter_config, task.user_overrides)
            # Save updated config
            self.write_file(adapter_config_path, json.dumps(adapter_config, indent=2))
            logs.append("Applied user overrides to adapter config")

        # Read feature dictionary for context
        feature_dict_result = self.read_file(feature_dict_path)
        feature_dict = feature_dict_result.get("parsed", {})

        # Read usage guide for context
        usage_guide_result = self.read_file(usage_guide_path)
        usage_guide = usage_guide_result.get("content", "")

        # ---------------------------------------------------------------------
        # Generate data_adapter.py
        # ---------------------------------------------------------------------
        logs.append("Generating data adapter code...")

        code_prompt = self._create_adapter_code_prompt(
            adapter_config=adapter_config,
            feature_dict=feature_dict,
            usage_guide=usage_guide,
            y_column=y_column,
            prediction_horizon=prediction_horizon
        )

        code_response = self.think(code_prompt)
        adapter_code = self.extract_code_from_response(code_response)

        if not adapter_code:
            return self.create_response(
                success=False,
                status="ERROR",
                message="Failed to generate adapter code",
                logs=logs + [f"LLM response: {code_response[:500]}"]
            )

        logs.append(f"Generated {len(adapter_code)} chars of adapter code")

        # Verify syntax
        verify_result = self.execute_python(f"import ast; ast.parse('''{adapter_code}''')")
        if not verify_result["success"]:
            logs.append(f"Syntax error: {verify_result['error']}")
            # Try to fix
            fix_prompt = f"""Fix this Python code syntax error:

```python
{adapter_code}
```

Error: {verify_result['error']}

Return ONLY the fixed code."""

            fixed_response = self.think(fix_prompt)
            adapter_code = self.extract_code_from_response(fixed_response)

            verify_result = self.execute_python(f"import ast; ast.parse('''{adapter_code}''')")
            if not verify_result["success"]:
                return self.create_response(
                    success=False,
                    status="ERROR",
                    message=f"Could not fix syntax error: {verify_result['error']}",
                    logs=logs
                )

        logs.append("Syntax check passed")

        # Save adapter code
        self.write_file(data_adapter_path, adapter_code)
        logs.append(f"Saved data adapter to {data_adapter_path}")

        logs.append("=== Code Generation Complete ===")

        return self.create_response(
            success=True,
            status="COMPLETED",
            message="Data adapter code generated successfully",
            generated_files={
                "data_adapter": data_adapter_path
            },
            logs=logs
        )

    def fix_adapter(self, task: AgentTask) -> AgentResponse:
        """
        Fix data adapter code based on error from Test Agent.

        Args:
            task: AgentTask with error context:
                - context["error"]: Error message from test
                - context["adapter_config_path"]: Path to adapter config
                - context["data_adapter_path"]: Path to current adapter code
                - context["usage_guide_path"]: Path to usage guide
                - context["feature_dict_path"]: Path to feature dictionary
                - context["fix_attempt"]: Which attempt this is

        Returns:
            AgentResponse with fixed adapter code
        """
        logs = []
        logs.append("=== Data Adapt Agent - Fix Mode ===")

        error = task.context.get("error", "Unknown error")
        adapter_config_path = task.context.get("adapter_config_path")
        data_adapter_path = task.context.get("data_adapter_path")
        usage_guide_path = task.context.get("usage_guide_path")
        feature_dict_path = task.context.get("feature_dict_path")
        fix_attempt = task.context.get("fix_attempt", 1)

        logs.append(f"Fix attempt: {fix_attempt}")
        logs.append(f"Error to fix: {error[:200]}...")

        # Read current adapter code
        current_code = self.read_file(data_adapter_path) if data_adapter_path else ""
        adapter_config = json.loads(self.read_file(adapter_config_path)) if adapter_config_path else {}
        usage_guide = self.read_file(usage_guide_path) if usage_guide_path else ""
        feature_dict = json.loads(self.read_file(feature_dict_path)) if feature_dict_path else {}

        # Create fix prompt
        fix_prompt = self._create_fix_prompt(
            error=error,
            current_code=current_code,
            adapter_config=adapter_config,
            feature_dict=feature_dict,
            usage_guide=usage_guide,
            y_column=task.y_column,
            prediction_horizon=task.prediction_horizon
        )

        logs.append("Asking LLM to fix the code...")

        # Get fixed code from LLM
        fix_response = self.think(fix_prompt)
        fixed_code = self.extract_code_from_response(fix_response)

        if not fixed_code:
            logs.append("Failed to extract fixed code from LLM response")
            return self.create_response(
                success=False,
                status="ERROR",
                message="LLM did not return valid code",
                logs=logs
            )

        logs.append(f"Generated {len(fixed_code)} chars of fixed code")

        # Verify syntax
        verify_result = self.execute_python(f"import ast; ast.parse('''{fixed_code}''')")
        if not verify_result["success"]:
            logs.append(f"Syntax error in fixed code: {verify_result['error']}")
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Fixed code has syntax error: {verify_result['error']}",
                logs=logs
            )

        logs.append("Syntax check passed")

        # Save fixed code
        self.write_file(data_adapter_path, fixed_code)
        logs.append(f"Saved fixed adapter to {data_adapter_path}")

        return self.create_response(
            success=True,
            status="FIXED",
            message="Data adapter fixed successfully",
            generated_files={
                "data_adapter": data_adapter_path
            },
            logs=logs
        )

    def _create_fix_prompt(
        self,
        error: str,
        current_code: str,
        adapter_config: Dict,
        feature_dict: Dict,
        usage_guide: str,
        y_column: str,
        prediction_horizon: int
    ) -> str:
        """Create prompt for fixing data adapter code."""

        # Extract key values from config
        source_format = adapter_config.get("source_format", {})
        model_config = adapter_config.get("model_config_overrides", {})
        bars_per_day = source_format.get("bars_per_day", 7)
        output_hours = model_config.get("H", bars_per_day)

        return f"""Fix this Python data adapter code that has an error.

=== ERROR MESSAGE ===
{error}

=== CURRENT CODE (has bug) ===
```python
{current_code}
```

=== ADAPTER CONFIGURATION ===
{json.dumps(adapter_config, indent=2)[:2000]}

=== FEATURE DICTIONARY ===
{json.dumps(feature_dict, indent=2)[:1500]}

=== MODEL USAGE GUIDE ===
{usage_guide[:1500]}

=== KEY REQUIREMENTS ===

1. SELF-CONTAINED: Don't load any JSON config files at runtime
2. Use self.data_dir (the parameter), NOT literal strings
3. File paths:
   - Independent factors: os.path.join(self.data_dir, "features", "independent_factors", f"{{ticker}}.parquet")
   - Shared factors: os.path.join(self.data_dir, "features", "shared_factors.parquet")  
   - Embeddings: os.path.join(self.data_dir, "features", "embeddings", f"{{ticker}}.npy")

4. Output configuration:
   - BARS_PER_DAY = {bars_per_day}
   - OUTPUT_HOURS (H) = {output_hours}
   - Y_COLUMN = "{y_column}"
   - PREDICTION_HORIZON = {prediction_horizon}

5. Time Window Shift: Shift X forward, Y keeps true time index
```python
x_shifted = x_data.shift(+{prediction_horizon})
result = result.dropna()  # Drop rows where X is NaN
```

6. Column naming for output:
   - x_{{feat_idx}}_s{{series_idx}}
   - share_{{feat_idx}}
   - y_{{step}}_s{{series_idx}} (generate {output_hours} Y columns per series)
   - emb_0_s{{series_idx}}_d{{dim_idx}}

=== COMMON ISSUES ===

- Shape mismatch: Make sure DataFrame columns and indices align properly
- Using literal strings instead of self.data_dir
- Loading JSON files that don't exist
- Wrong number of Y columns (should be {output_hours} per series)
- Forgetting to handle NaN values after shift

=== FIX THE CODE ===

Analyze the error, identify the bug, and write the COMPLETE fixed code.
Return ONLY the Python code, no explanations."""

    def _create_analysis_prompt(
        self,
        feature_dict: Dict,
        usage_guide: str,
        y_column: str,
        prediction_horizon: int,
        user_overrides: Dict
    ) -> str:
        """Create prompt for analysis phase."""
        return f"""Analyze the source data format and target model requirements to create a transformation plan.

=== SOURCE DATA (Feature Dictionary) ===
{json.dumps(feature_dict, indent=2)}

=== TARGET FORMAT (Usage Guide) ===
{usage_guide}

=== USER CONFIGURATION ===
Y Column (target variable): {y_column}
Prediction Horizon: {prediction_horizon} time steps ahead
User Overrides: {json.dumps(user_overrides, indent=2) if user_overrides else "None"}

=== YOUR TASK ===

Analyze and create a JSON transformation configuration with:

1. SOURCE FORMAT ANALYSIS:
   - What resolution is the source data? (hourly, daily, etc.)
   - How many bars per day?
   - What columns/factors are available?
   - What is the embedding dimension?

2. TARGET FORMAT ANALYSIS:
   - What resolution does the model expect?
   - What column naming convention?
   - How many output hours/steps?

3. Y VALUE CONFIGURATION:
   - Which column is Y: "{y_column}"
   - How to handle Y (expand to columns for multi-step prediction?)
   - IMPORTANT: Y keeps its true time index! X is shifted forward.
   - Window shift direction: X shifted forward by {prediction_horizon} steps
   - Result: Row at time t has X[t-{prediction_horizon}] and Y[t]

4. X VALUE CONFIGURATION:
   - All factors except Y become X
   - For each factor, what aggregation method? (last, mean, sum, etc.)
   - How to handle embeddings (broadcast to all timestamps)

5. BLOCKING ISSUES vs AUTO-RESOLVABLE:
   
   **NOT blocking (auto-resolve these):**
   - Model has configurable parameters (like H for output hours) → Just set them to match data
   - Resolution mismatch when model accepts different resolutions → Configure model
   - Number of output steps is configurable → Set to match data (e.g., H=7 for 7 trading hours)
   
   **Actually blocking (need user input):**
   - Model REQUIRES specific format that data cannot provide
   - Missing required data that doesn't exist
   - Fundamental incompatibility with no workaround
   
   If a model parameter is configurable, AUTO-SET it based on the data. Include it in "model_config_overrides".

6. WARNINGS:
   - Non-blocking issues or suggestions

Return a JSON object with this structure:
{{
  "source_format": {{
    "resolution": "hourly",
    "bars_per_day": 7,
    "trading_hours": ["09:30", ...],
    "independent_factors": [...],
    "shared_factors": [...],
    "embedding_dim": 384
  }},
  "target_format": {{
    "input_resolution": "daily",
    "output_steps": 7,
    "column_convention": "x_{{feat}}_s{{series}}"
  }},
  "y_config": {{
    "column": "{y_column}",
    "prediction_horizon": {prediction_horizon},
    "shift_direction": "x_forward",
    "shift_steps": {prediction_horizon},
    "expand_to_columns": true,
    "output_columns": 7
  }},
  "x_config": {{
    "independent_factors": {{
      "columns": [...],
      "aggregations": {{"factor_name": "method", ...}}
    }},
    "shared_factors": {{
      "columns": [...],
      "aggregations": {{...}}
    }},
    "embeddings": {{
      "broadcast": true,
      "dimension": 384
    }}
  }},
  "model_config_overrides": {{
    "H": 7,
    "reason": "Source data has 7 trading hours per day, setting H=7 to match"
  }},
  "blocking_issues": [],
  "warnings": [...]
}}

Think carefully about time window shifting to avoid look-ahead bias!"""

    def _create_reasoning_prompt(
        self,
        feature_dict: Dict,
        usage_guide: str,
        y_column: str,
        prediction_horizon: int,
        adapter_config: Dict
    ) -> str:
        """Create prompt for generating reasoning document."""
        return f"""Create a detailed Markdown document explaining the data transformation reasoning.

=== CONTEXT ===
Feature Dictionary: {json.dumps(feature_dict, indent=2)[:2000]}...
Y Column: {y_column}
Prediction Horizon: {prediction_horizon}
Adapter Config: {json.dumps(adapter_config, indent=2)}

=== DOCUMENT STRUCTURE ===

# Data Transformation Reasoning

## 1. Source Data Analysis
- Resolution and format
- Available factors
- Embeddings

## 2. Target Model Requirements
- Expected input format
- Column naming
- Output format

## 3. Transformation Decisions

### Y Value Handling
- Why this Y column
- Window shift explanation (CRITICAL: explain how X(t) predicts Y(t+horizon))
- Output column format

### Independent Factors
| Factor | Aggregation | Reasoning |
|--------|-------------|-----------|
| ... | ... | ... |

### Shared Factors
| Factor | Aggregation | Reasoning |
|--------|-------------|-----------|
| ... | ... | ... |

### Embeddings
- How mapped to timestamps
- Dimension handling

## 4. Time Window Shifting
CRITICAL: Explain exactly how we avoid look-ahead bias while keeping Y's true time index:
- Y values keep their TRUE time index (when the return actually occurred)
- X features are shifted FORWARD by {prediction_horizon} steps
- Row at time t has: X[t-{prediction_horizon}] and Y[t]
- This means: to predict Y at time t, we use X from {prediction_horizon} step(s) ago
- Example with actual timestamps showing the alignment

## 5. Issues and Warnings
List any issues found and how they're handled.

Write a clear, detailed document that another developer could follow."""

    def _create_adapter_code_prompt(
        self,
        adapter_config: Dict,
        feature_dict: Dict,
        usage_guide: str,
        y_column: str,
        prediction_horizon: int
    ) -> str:
        """Create prompt for generating data adapter code."""

        # Extract key values from config to embed in code
        source_format = adapter_config.get("source_format", {})
        x_config = adapter_config.get("x_config", {})
        y_config = adapter_config.get("y_config", {})
        model_config = adapter_config.get("model_config_overrides", {})

        bars_per_day = source_format.get("bars_per_day", 7)
        output_hours = model_config.get("H", bars_per_day)

        return f"""Write a Python data adapter module based on this configuration.

=== ADAPTER CONFIGURATION (embed these values directly in code, do NOT load from JSON) ===
{json.dumps(adapter_config, indent=2)}

=== FEATURE DICTIONARY ===
{json.dumps(feature_dict, indent=2)[:3000]}

=== USAGE GUIDE (TARGET FORMAT) ===
{usage_guide[:2000]}

=== CRITICAL: DIRECTORY STRUCTURE ===
The data is organized as follows. The `data_dir` parameter is passed to __init__ at runtime 
(e.g., data_dir="./test_data"). Use self.data_dir to build paths:

```
<data_dir>/                          # e.g., "./test_data"
└── features/
    ├── feature_dictionary.json
    ├── independent_factors/
    │   ├── AAPL.parquet
    │   └── JNJ.parquet
    ├── shared_factors.parquet
    └── embeddings/
        ├── AAPL.npy
        └── JNJ.npy
```

=== REQUIREMENTS ===

**IMPORTANT: The DataAdapter class must be SELF-CONTAINED. 
Do NOT load adapter_config.json at runtime. 
Embed all configuration values directly in the code as class attributes or constants.**

Create a Python module with:

1. `class DataAdapter`:
   - `__init__(self, data_dir, tickers, y_column="{y_column}", prediction_horizon={prediction_horizon})`
     - Store: `self.data_dir = data_dir`  # USE THE PARAMETER, not a literal string!
     - Store: `self.tickers = tickers`
   - `load_features(self)` - Load all feature data from parquet/npy files
   - `transform(self)` - Apply all transformations, return DataFrame ready for model
   - `get_train_test_split(self, test_ratio=0.2)` - Split data properly
   - `get_model_config(self)` - Return dict with model config overrides (H={output_hours}, etc.)

2. EMBED CONFIGURATION IN CODE (do not load from JSON):
```python
class DataAdapter:
    # Configuration embedded directly in code
    BARS_PER_DAY = {bars_per_day}
    OUTPUT_HOURS = {output_hours}  # H for model
    Y_COLUMN = "{y_column}"
    PREDICTION_HORIZON = {prediction_horizon}
    
    # Aggregation methods for each factor
    AGGREGATIONS = {json.dumps(x_config.get("independent_factors", {}).get("aggregations", {}))}
    
    def __init__(self, data_dir, tickers, y_column=None, prediction_horizon=None):
        # IMPORTANT: Use the data_dir PARAMETER, not a hardcoded string!
        self.data_dir = data_dir  # This is passed in, e.g., "./test_data"
        self.tickers = tickers
        self.y_column = y_column or self.Y_COLUMN
        self.prediction_horizon = prediction_horizon or self.PREDICTION_HORIZON
```

3. EXACT FILE PATHS - use self.data_dir (the parameter, not a literal string!):
```python
# CORRECT - uses the data_dir parameter:
path = os.path.join(self.data_dir, "features", "independent_factors", f"{{ticker}}.parquet")

# WRONG - do NOT do this:
path = os.path.join("path_to_data_directory", ...)  # NO!
path = os.path.join("data_dir", ...)  # NO!
```
   - Independent factors: `os.path.join(self.data_dir, "features", "independent_factors", f"{{ticker}}.parquet")`
   - Shared factors: `os.path.join(self.data_dir, "features", "shared_factors.parquet")`
   - Embeddings: `os.path.join(self.data_dir, "features", "embeddings", f"{{ticker}}.npy")`

4. Key transformations to implement:
   - Load independent_factors/{{ticker}}.parquet for each ticker
   - Load shared_factors.parquet
   - Load embeddings/{{ticker}}.npy for each ticker
   - Apply aggregation methods (last, mean, sum, etc.) for resolution conversion
   - Shift X forward by prediction_horizon (Y keeps true time index)
   - Broadcast embeddings to all timestamps
   - Rename columns to match model's expected format

5. CRITICAL - Time Window Shift (for prediction horizon = {prediction_horizon}):
   
   **IMPORTANT: Shift X forward, NOT Y backward!**
   Y must keep its true time index for proper backtesting and evaluation.
   
```python
# CORRECT: Shift X forward so X[t-horizon] aligns with Y[t]
x_shifted = x_data.shift(+{prediction_horizon})  # X moves forward, creating lag

# Row at time t now has:
# - X features from time t-{prediction_horizon} (the past)
# - Y value from time t (the actual outcome at this time)

# Drop first rows where X is NaN (no historical data available)
result = result.dropna()
```

6. Column naming:
   - Follow the target format from usage guide
   - Independent features: x_{{feat_idx}}_s{{series_idx}}
   - Shared features: share_{{feat_idx}}
   - Embeddings: emb_0_s{{series_idx}}_d{{dim_idx}}
   - Y values: y_{{step}}_s{{series_idx}} (generate {output_hours} Y columns per series)

7. get_model_config() method:
```python
def get_model_config(self):
    return {{
        "H": {output_hours},
        "bars_per_day": {bars_per_day},
        "prediction_horizon": {prediction_horizon}
    }}
```

8. Include proper error handling and logging

Write ONLY the Python code. Start with imports. Do NOT load any JSON config files."""

    def _create_fallback_config(
        self,
        feature_dict: Dict,
        y_column: str,
        prediction_horizon: int
    ) -> Dict:
        """Create a basic fallback config if LLM fails."""
        datasets = feature_dict.get("datasets", {})
        independent = datasets.get("independent_factors", {})
        shared = datasets.get("shared_factors", {})
        embeddings = datasets.get("embeddings", {})

        independent_factors = list(independent.get("factors", {}).keys())
        shared_factors = list(shared.get("factors", {}).keys())

        # Remove Y from independent factors
        x_factors = [f for f in independent_factors if f != y_column]

        # Get frequency from feature dictionary if available
        frequency = independent.get("frequency", "unknown")

        # bars_per_day and output_steps should be detected from actual data
        # Mark as "auto_detect" so Data Adapt Agent will calculate from data
        return {
            "source_format": {
                "resolution": frequency,
                "bars_per_day": "auto_detect",  # Will be detected from actual data
                "independent_factors": independent_factors,
                "shared_factors": shared_factors,
                "embedding_dim": embeddings.get("dimension", None)
            },
            "target_format": {
                "input_resolution": "auto_detect",  # Depends on model requirements
                "output_steps": "auto_detect"  # Depends on model and data
            },
            "y_config": {
                "column": y_column,
                "prediction_horizon": prediction_horizon,
                "shift_direction": "x_forward",  # Shift X forward, Y keeps true time index
                "shift_steps": prediction_horizon,  # X is shifted forward by this many steps
                "expand_to_columns": "auto_detect",  # Depends on model
                "output_columns": "auto_detect"  # Will be detected from data
            },
            "x_config": {
                "independent_factors": {
                    "columns": x_factors,
                    "aggregations": {f: "last" for f in x_factors}  # Default to last, LLM should override
                },
                "shared_factors": {
                    "columns": shared_factors,
                    "aggregations": {f: "last" for f in shared_factors}
                },
                "embeddings": {
                    "broadcast": True,
                    "dimension": embeddings.get("dimension", None)
                }
            },
            "model_config_overrides": {
                "H": "auto_detect",  # Will be set to bars_per_day after detection
                "reason": "Auto-configure model output hours to match source data"
            },
            "blocking_issues": [
                {
                    "type": "auto_detect_required",
                    "severity": "blocking",
                    "message": "Could not fully analyze data format. Need to detect bars_per_day and output_steps from actual data.",
                    "fields": ["bars_per_day", "output_steps", "output_columns"],
                    "options": [
                        "Run data detection to calculate these values",
                        "Provide values manually in user_overrides"
                    ]
                }
            ],
            "warnings": [
                {
                    "type": "fallback_config",
                    "message": "Using fallback configuration. LLM analysis failed - values may need adjustment."
                }
            ]
        }

    def _apply_overrides_to_issues(
        self,
        blocking_issues: List[Dict],
        user_overrides: Dict
    ) -> List[Dict]:
        """Remove blocking issues that are resolved by user overrides."""
        remaining_issues = []

        for issue in blocking_issues:
            issue_type = issue.get("type", "")

            # Check if user override resolves this issue
            resolved = False

            if issue_type == "resolution_mismatch":
                if "y_config" in user_overrides:
                    y_override = user_overrides["y_config"]
                    if "output_columns" in y_override:
                        resolved = True

            if not resolved:
                remaining_issues.append(issue)

        return remaining_issues

    def _merge_overrides(self, config: Dict, overrides: Dict) -> Dict:
        """Deep merge overrides into config."""
        result = config.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_overrides(result[key], value)
            else:
                result[key] = value

        return result

    def _detect_data_properties(self, data_dir: str, tickers: List[str]) -> Dict[str, Any]:
        """
        Detect data properties from actual parquet files.

        Detects:
        - bars_per_day: How many data points per day
        - trading_hours: List of hours in the data
        - date_range: Start and end dates
        - resolution: Inferred resolution (hourly, daily, etc.)

        Args:
            data_dir: Path to data directory
            tickers: List of ticker symbols

        Returns:
            Dict with detected properties
        """
        import os

        result = {
            "bars_per_day": None,
            "trading_hours": [],
            "date_range": {"start": None, "end": None},
            "resolution": "unknown",
            "detection_error": None
        }

        # Try to load a sample parquet file
        features_dir = os.path.join(data_dir, "features", "independent_factors")

        if not os.path.exists(features_dir):
            result["detection_error"] = f"Features directory not found: {features_dir}"
            return result

        # Find a parquet file to analyze
        sample_file = None
        for ticker in tickers:
            ticker_file = os.path.join(features_dir, f"{ticker}.parquet")
            if os.path.exists(ticker_file):
                sample_file = ticker_file
                break

        if not sample_file:
            # Try any parquet file in the directory
            for f in os.listdir(features_dir):
                if f.endswith('.parquet'):
                    sample_file = os.path.join(features_dir, f)
                    break

        if not sample_file:
            result["detection_error"] = "No parquet files found in features directory"
            return result

        # Analyze the file
        detection_code = f'''
import pandas as pd
import numpy as np

df = pd.read_parquet("{sample_file}")

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    elif 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index)

# Get date only (no time)
dates = df.index.date

# Count bars per day
from collections import Counter
bars_count = Counter(dates)
bars_per_day_values = list(bars_count.values())

# Most common bars per day
if bars_per_day_values:
    bars_per_day = int(np.median(bars_per_day_values))
else:
    bars_per_day = 1

# Get trading hours
hours = sorted(df.index.hour.unique().tolist())

# Detect resolution
if len(hours) > 1:
    # Multiple hours per day = hourly or sub-hourly
    hour_diff = np.diff(sorted(hours))
    if len(hour_diff) > 0:
        typical_diff = int(np.median(hour_diff))
        if typical_diff == 1:
            resolution = "hourly"
        elif typical_diff < 1:
            resolution = "sub_hourly"
        else:
            resolution = f"{{typical_diff}}_hourly"
    else:
        resolution = "hourly"
elif bars_per_day == 1:
    resolution = "daily"
else:
    resolution = "unknown"

# Date range
date_start = str(df.index.min().date())
date_end = str(df.index.max().date())

result = {{
    "bars_per_day": bars_per_day,
    "trading_hours": hours,
    "date_range": {{"start": date_start, "end": date_end}},
    "resolution": resolution,
    "total_rows": len(df),
    "unique_days": len(set(dates))
}}
'''

        exec_result = self.execute_python(detection_code)

        if exec_result["success"] and exec_result.get("result"):
            result.update(exec_result["result"])
        else:
            result["detection_error"] = exec_result.get("error", "Unknown error during detection")

        return result

    def _resolve_auto_detect_values(
        self,
        adapter_config: Dict,
        detected_props: Dict
    ) -> Dict:
        """
        Replace 'auto_detect' values in config with detected values.

        Args:
            adapter_config: Config with potential 'auto_detect' values
            detected_props: Properties detected from actual data

        Returns:
            Updated config with resolved values
        """
        config = adapter_config.copy()

        # Resolve source_format
        if "source_format" in config:
            sf = config["source_format"]
            if sf.get("bars_per_day") == "auto_detect":
                sf["bars_per_day"] = detected_props.get("bars_per_day", 1)
            if sf.get("resolution") == "unknown" or sf.get("resolution") == "auto_detect":
                sf["resolution"] = detected_props.get("resolution", "unknown")
            if "trading_hours" not in sf or not sf["trading_hours"]:
                sf["trading_hours"] = detected_props.get("trading_hours", [])

        # Resolve target_format
        if "target_format" in config:
            tf = config["target_format"]
            if tf.get("output_steps") == "auto_detect":
                # Default to bars_per_day if not specified
                tf["output_steps"] = detected_props.get("bars_per_day", 1)

        # Resolve y_config
        if "y_config" in config:
            yc = config["y_config"]
            if yc.get("output_columns") == "auto_detect":
                yc["output_columns"] = detected_props.get("bars_per_day", 1)
            if yc.get("expand_to_columns") == "auto_detect":
                # Expand if bars_per_day > 1
                yc["expand_to_columns"] = detected_props.get("bars_per_day", 1) > 1

        # Resolve model_config_overrides (auto-set configurable model parameters)
        if "model_config_overrides" in config:
            mco = config["model_config_overrides"]
            bars_per_day = detected_props.get("bars_per_day", 1)

            # Auto-set H (output hours) to match source data
            if mco.get("H") == "auto_detect":
                mco["H"] = bars_per_day
                mco["reason"] = f"Auto-configured H={bars_per_day} to match source data ({bars_per_day} bars/day)"

        # Remove blocking issues related to auto_detect if resolved
        if "blocking_issues" in config:
            config["blocking_issues"] = [
                issue for issue in config["blocking_issues"]
                if issue.get("type") not in ["auto_detect_required", "resolution_mismatch"]
            ]
        
        # Add detection info to config
        config["detected_properties"] = detected_props
        
        return config