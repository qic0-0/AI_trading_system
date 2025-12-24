"""
Model Code Agent - Writes model code and usage guide from description.

This agent reads model_description.md and generates:
1. model.py - Complete model implementation with train() and predict() methods
2. usage_guide.md - Documentation for Data Adapt Agent to understand input/output format

The agent uses LLM to understand the model description and write appropriate code.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from .base_mcp_agent import BaseMCPAgent
from .mcp_config import AgentTask, AgentResponse

logger = logging.getLogger(__name__)


class ModelCodeAgent(BaseMCPAgent):
    """
    Agent that generates model code from description.
    
    Input: model_description.md
    Output: model.py, usage_guide.md
    """
    
    def __init__(self, llm_client, config):
        super().__init__("ModelCodeAgent", llm_client, config)
    
    @property
    def system_prompt(self) -> str:
        return """You are a Model Code Agent specialized in writing quantitative trading models.

Your task is to read a model description and generate:
1. Complete Python code implementing the model (model.py)
2. A usage guide documenting input/output format (usage_guide.md)

When writing model code:
- Create a clean, well-documented Python class
- Include train() and predict() methods
- Handle edge cases and errors gracefully
- Use standard libraries (numpy, pandas, sklearn, torch, hmmlearn, etc.)
- Save/load model state properly

When writing the usage guide:
- Clearly document the expected input DataFrame format
- Specify column naming conventions
- Document all parameters with types and defaults
- Show example usage code
- Describe output format

Be precise about data formats - the Data Adapt Agent will use this to transform data."""
    
    def _register_tools(self):
        """Register Model Code Agent specific tools."""
        # All common tools from base class are already registered
        pass
    
    def run(self, task: AgentTask) -> AgentResponse:
        """
        Generate model code and usage guide.
        
        Args:
            task: AgentTask with task_type="write_model"
            
        Returns:
            AgentResponse with generated files
        """
        logs = []
        logs.append(f"=== Model Code Agent Started ===")
        
        model_dir = task.paths.get("model_dir")
        description_path = task.paths.get("description")
        model_code_path = task.paths.get("model_code")
        usage_guide_path = task.paths.get("usage_guide")
        
        # ---------------------------------------------------------------------
        # Step 1: Read model description
        # ---------------------------------------------------------------------
        logs.append("Reading model description...")
        
        description_result = self.read_file(description_path)
        if not description_result["success"]:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Failed to read model description: {description_result['error']}",
                logs=logs
            )
        
        model_description = description_result["content"]
        logs.append(f"Read {len(model_description)} chars from description")
        
        # ---------------------------------------------------------------------
        # Step 2: Generate model code using LLM
        # ---------------------------------------------------------------------
        logs.append("Generating model code...")
        
        model_code_prompt = self._create_model_code_prompt(model_description)
        
        model_code_response = self.think(model_code_prompt)
        model_code = self.extract_code_from_response(model_code_response)
        
        if not model_code:
            return self.create_response(
                success=False,
                status="ERROR",
                message="Failed to generate model code",
                logs=logs + [f"LLM response: {model_code_response[:500]}"]
            )
        
        logs.append(f"Generated {len(model_code)} chars of model code")
        
        # Save model code
        write_result = self.write_file(model_code_path, model_code)
        if not write_result["success"]:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Failed to write model code: {write_result['error']}",
                logs=logs
            )
        
        logs.append(f"Saved model code to {model_code_path}")
        
        # ---------------------------------------------------------------------
        # Step 3: Generate usage guide using LLM
        # ---------------------------------------------------------------------
        logs.append("Generating usage guide...")
        
        usage_guide_prompt = self._create_usage_guide_prompt(model_description, model_code)
        
        usage_guide_response = self.think(usage_guide_prompt)
        usage_guide = self._extract_markdown_from_response(usage_guide_response)
        
        if not usage_guide:
            usage_guide = usage_guide_response  # Use raw response if extraction fails
        
        logs.append(f"Generated {len(usage_guide)} chars of usage guide")
        
        # Save usage guide
        write_result = self.write_file(usage_guide_path, usage_guide)
        if not write_result["success"]:
            return self.create_response(
                success=False,
                status="ERROR",
                message=f"Failed to write usage guide: {write_result['error']}",
                logs=logs
            )
        
        logs.append(f"Saved usage guide to {usage_guide_path}")
        
        # ---------------------------------------------------------------------
        # Step 4: Verify generated code (basic syntax check)
        # ---------------------------------------------------------------------
        logs.append("Verifying generated code...")
        
        verify_result = self.execute_python(f"import ast; ast.parse('''{model_code}''')")
        if not verify_result["success"]:
            logs.append(f"Syntax error in generated code: {verify_result['error']}")
            # Try to fix the code
            logs.append("Attempting to fix syntax error...")
            
            fix_prompt = f"""The following Python code has a syntax error:

```python
{model_code}
```

Error: {verify_result['error']}

Please fix the syntax error and return the corrected code.
Return ONLY the fixed Python code, no explanations."""
            
            fixed_response = self.think(fix_prompt)
            fixed_code = self.extract_code_from_response(fixed_response)
            
            if fixed_code:
                verify_result = self.execute_python(f"import ast; ast.parse('''{fixed_code}''')")
                if verify_result["success"]:
                    logs.append("Fixed syntax error successfully")
                    model_code = fixed_code
                    self.write_file(model_code_path, model_code)
                else:
                    return self.create_response(
                        success=False,
                        status="ERROR",
                        message=f"Could not fix syntax error: {verify_result['error']}",
                        logs=logs
                    )
            else:
                return self.create_response(
                    success=False,
                    status="ERROR",
                    message="Failed to fix syntax error",
                    logs=logs
                )
        else:
            logs.append("Syntax check passed")
        
        logs.append("=== Model Code Agent Complete ===")
        
        return self.create_response(
            success=True,
            status="COMPLETED",
            message="Model code and usage guide generated successfully",
            generated_files={
                "model_code": model_code_path,
                "usage_guide": usage_guide_path
            },
            logs=logs
        )
    
    def _create_model_code_prompt(self, model_description: str) -> str:
        """Create prompt for generating model code."""
        return f"""Based on the following model description, write a complete Python implementation.

=== MODEL DESCRIPTION ===
{model_description}
=== END DESCRIPTION ===

Requirements for the Python code:

1. Create a class called `Model` with these methods:
   - `__init__(self, **kwargs)`: Initialize with configurable parameters
   - `train(self, X, y=None, **kwargs)`: Train the model
   - `predict(self, X, **kwargs)`: Make predictions
   - `save(self, path)`: Save model to file
   - `load(self, path)`: Load model from file

2. Use standard imports at the top:
   - numpy as np
   - pandas as pd
   - Any model-specific imports (sklearn, torch, hmmlearn, etc.)

3. Include proper docstrings explaining:
   - What each method does
   - Expected input format (X should be DataFrame or numpy array)
   - Output format
   - Parameters

4. Handle edge cases:
   - Empty data
   - Missing values
   - Wrong data types

5. The train() method should return a dict with training metrics
6. The predict() method should return a dict with predictions

Write ONLY the Python code, no explanations. Start with imports."""
    
    def _create_usage_guide_prompt(self, model_description: str, model_code: str) -> str:
        """Create prompt for generating usage guide."""
        return f"""Based on the model description and implementation below, write a comprehensive usage guide in Markdown format.

=== MODEL DESCRIPTION ===
{model_description}
=== END DESCRIPTION ===

=== MODEL CODE ===
{model_code}
=== END CODE ===

The usage guide should include:

## 1. Overview
Brief description of what the model does.

## 2. Input Data Format
CRITICAL: Clearly specify the expected DataFrame/array format:
- Column names and their meanings
- Data types expected
- Resolution (hourly, daily, etc.)
- Required preprocessing

Example:
| Column | Type | Description |
|--------|------|-------------|
| feature_0 | float | First feature |
| ... | ... | ... |

## 3. Parameters
List all __init__ and method parameters with:
- Name
- Type
- Default value
- Description

## 4. Usage Example
```python
# Example code showing how to use the model
```

## 5. Output Format
Describe what train() and predict() return:
- Dictionary keys
- Value types
- Example output

## 6. Notes
Any important notes about data preparation, limitations, etc.

Write the complete Markdown document. Be precise about data formats - another agent will use this to prepare data."""
    
    def _extract_markdown_from_response(self, text: str) -> str:
        """Extract markdown content from LLM response."""
        import re
        
        # Try to extract from markdown code block
        md_block_pattern = r'```(?:markdown|md)?\n(.*?)\n```'
        matches = re.findall(md_block_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If response starts with #, assume it's all markdown
        if text.strip().startswith('#'):
            return text.strip()
        
        # Return as-is
        return text.strip()
