"""
MCP Configuration for Quant Model Agent System.

This module contains configuration specific to the MCP (Model-Code-Protocol) system
that orchestrates model code generation, data adaptation, and testing.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os


@dataclass
class MCPConfig:
    """Configuration for MCP Quant Agent System."""
    
    # =========================================================================
    # Directory Paths
    # =========================================================================
    
    # Base directory for models (each model gets a subfolder)
    models_dir: str = "models/"
    
    # Base directory for data (Feature Agent output location)
    data_dir: str = "data/"
    
    # =========================================================================
    # Retry and Timeout Settings
    # =========================================================================
    
    # Max retries for each agent before giving up
    max_retries: int = 3
    
    # Max iterations for error-fix cycles
    max_fix_iterations: int = 5
    
    # Timeout for LLM calls (seconds)
    llm_timeout: int = 120
    
    # Timeout for code execution (seconds)
    execution_timeout: int = 60
    
    # =========================================================================
    # Agent Behavior Settings
    # =========================================================================
    
    # Whether to save intermediate reasoning files
    save_reasoning: bool = True
    
    # Whether to pause on blocking issues (vs. auto-fail)
    pause_on_issues: bool = True
    
    # Whether to generate detailed logs
    verbose_logging: bool = True
    
    # =========================================================================
    # Model Code Agent Settings
    # =========================================================================
    
    # Default model type if not specified
    default_model_type: str = "hmm"
    
    # =========================================================================
    # Data Adapt Agent Settings
    # =========================================================================
    
    # Default aggregation methods for different feature types
    default_aggregations: Dict[str, str] = field(default_factory=lambda: {
        "rsi": "last",
        "macd": "last",
        "return": "sum",
        "volume": "sum",
        "volatility": "last",
        "price_range": "max",
        "default": "last"
    })
    
    # =========================================================================
    # Test Agent Settings
    # =========================================================================
    
    # Number of sample rows to use for testing
    test_sample_size: int = 100
    
    # Whether to run full integration test or just import test
    full_integration_test: bool = True


@dataclass
class MCPResult:
    """Result from MCP run or resume."""
    
    # Status: "SUCCESS", "PAUSED", "FAILED"
    status: str
    
    # Human-readable message
    message: str
    
    # List of blocking issues (if PAUSED)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # List of warnings (non-blocking)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Path to adapter_config.json (for user to edit)
    config_path: Optional[str] = None
    
    # Path to transformation_reasoning.md
    reasoning_path: Optional[str] = None
    
    # Paths to generated files
    generated_files: Dict[str, str] = field(default_factory=dict)
    
    # Test results (if testing completed)
    test_results: Optional[Dict[str, Any]] = None
    
    # Execution logs
    logs: List[str] = field(default_factory=list)
    
    # Session ID for resume capability
    session_id: Optional[str] = None
    
    # Partial work completed (for debugging)
    partial_work: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task definition passed between MCP Brain and sub-agents."""
    
    # Task type: "write_model", "analyze_data", "adapt_data", "test"
    task_type: str
    
    # Model name (folder under models/)
    model_name: str
    
    # Data directory
    data_dir: str
    
    # Y column name (user specified)
    y_column: Optional[str] = None
    
    # Prediction horizon
    prediction_horizon: int = 1
    
    # User overrides for data adaptation
    user_overrides: Optional[Dict[str, Any]] = None
    
    # Additional context from previous agents
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Paths to relevant files
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass 
class AgentResponse:
    """Response from a sub-agent back to MCP Brain."""
    
    # Whether task completed successfully
    success: bool
    
    # Status: "COMPLETED", "BLOCKED", "ERROR"
    status: str
    
    # Human-readable message
    message: str
    
    # Generated files (path -> description)
    generated_files: Dict[str, str] = field(default_factory=dict)
    
    # Blocking issues requiring user input
    blocking_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Warnings (non-blocking)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data to pass to next agent
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Execution logs
    logs: List[str] = field(default_factory=list)
    
    # Error details (if status == "ERROR")
    error: Optional[str] = None
    error_traceback: Optional[str] = None


# =============================================================================
# File Path Helpers
# =============================================================================

def get_model_dir(models_dir: str, model_name: str) -> str:
    """Get the directory path for a specific model."""
    return os.path.join(models_dir, model_name)


def get_features_dir(data_dir: str) -> str:
    """Get the features directory path."""
    return os.path.join(data_dir, "features")


def get_feature_dictionary_path(data_dir: str) -> str:
    """Get the feature dictionary file path."""
    return os.path.join(data_dir, "features", "feature_dictionary.json")


def get_model_paths(models_dir: str, model_name: str) -> Dict[str, str]:
    """
    Get all standard file paths for a model.
    
    Returns:
        Dict with keys:
        - model_dir: Base directory for this model
        - description: model_description.md
        - model_code: model.py (or user-provided .py file)
        - usage_guide: usage_guide.md (or user-provided)
        - adapter_config: adapter_config.json
        - data_adapter: data_adapter.py
        - reasoning: transformation_reasoning.md
        - issues: issues_report.json
        - test_results: test_results.json
    """
    model_dir = get_model_dir(models_dir, model_name)
    
    return {
        "model_dir": model_dir,
        "description": os.path.join(model_dir, "model_description.md"),
        "model_code": os.path.join(model_dir, "model.py"),
        "usage_guide": os.path.join(model_dir, "usage_guide.md"),
        "adapter_config": os.path.join(model_dir, "adapter_config.json"),
        "data_adapter": os.path.join(model_dir, "data_adapter.py"),
        "reasoning": os.path.join(model_dir, "transformation_reasoning.md"),
        "issues": os.path.join(model_dir, "issues_report.json"),
        "test_results": os.path.join(model_dir, "test_results.json"),
    }


def find_provided_files(model_dir: str) -> Dict[str, str]:
    """
    Find user-provided files in model directory.
    
    Looks for:
    - model_description.md (required)
    - Any .py file (model code)
    - Any *USAGE*.md or *usage*.md file (usage guide)
    
    Returns:
        Dict mapping file type to path (if found)
    """
    provided = {}
    
    if not os.path.exists(model_dir):
        return provided
    
    for filename in os.listdir(model_dir):
        filepath = os.path.join(model_dir, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        lower_name = filename.lower()
        
        # Model description
        if lower_name == "model_description.md":
            provided["description"] = filepath
        
        # Python model code (any .py file that's not data_adapter.py)
        elif filename.endswith(".py") and filename != "data_adapter.py":
            provided["model_code"] = filepath
        
        # Usage guide (contains "usage" in name)
        elif "usage" in lower_name and filename.endswith(".md"):
            provided["usage_guide"] = filepath
    
    return provided
