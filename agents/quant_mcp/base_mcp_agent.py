"""
Base MCP Agent class with shared tools.

All MCP sub-agents (Model Code Agent, Data Adapt Agent, Test Agent) inherit
from this base class which provides common tools for file I/O and code execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
import json
import logging
import traceback
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

from .mcp_config import AgentTask, AgentResponse

logger = logging.getLogger(__name__)


class BaseMCPAgent(ABC):
    """
    Abstract base class for all MCP sub-agents.
    
    Provides:
    - LLM client integration
    - Common tools (read_file, write_file, execute_python, etc.)
    - Tool registration for LLM function calling
    - ReAct-style thinking with tools
    """
    
    def __init__(self, name: str, llm_client, config):
        """
        Initialize base MCP agent.
        
        Args:
            name: Agent name (e.g., "ModelCodeAgent")
            llm_client: LLMClient instance for LLM calls
            config: MCPConfig instance
        """
        self.name = name
        self.llm_client = llm_client
        self.config = config
        self.tools = {}
        self.logger = logging.getLogger(name)
        
        # Register common tools
        self._register_common_tools()
        
        # Register agent-specific tools
        self._register_tools()
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        Must be implemented by each agent subclass.
        """
        pass
    
    @abstractmethod
    def _register_tools(self):
        """
        Register agent-specific tools.
        Each agent adds its own tools by calling self.register_tool()
        """
        pass
    
    @abstractmethod
    def run(self, task: AgentTask) -> AgentResponse:
        """
        Execute the agent's main task.
        
        Args:
            task: AgentTask with task details
            
        Returns:
            AgentResponse with results
        """
        pass
    
    # =========================================================================
    # Tool Registration
    # =========================================================================
    
    def register_tool(
        self,
        name: str,
        func: callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Register a tool for this agent.
        
        Args:
            name: Tool name
            func: Function to execute
            description: What the tool does
            parameters: JSON schema for parameters
        """
        self.tools[name] = {
            "function": func,
            "definition": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.logger.debug(f"Registered tool: {name}")
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get all tool definitions for LLM function calling."""
        return [tool["definition"] for tool in self.tools.values()]
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a registered tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        
        self.logger.info(f"Executing tool: {name}")
        return self.tools[name]["function"](**arguments)
    
    # =========================================================================
    # Common Tools Registration
    # =========================================================================
    
    def _register_common_tools(self):
        """Register tools common to all MCP agents."""
        
        # Read file tool
        self.register_tool(
            name="read_file",
            func=self.read_file,
            description="Read contents of a file (text, json, md, py, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        )
        
        # Write file tool
        self.register_tool(
            name="write_file",
            func=self.write_file,
            description="Write content to a file. Creates directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        )
        
        # Check path exists tool
        self.register_tool(
            name="check_path_exists",
            func=self.check_path_exists,
            description="Check if a file or directory exists",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check"
                    }
                },
                "required": ["path"]
            }
        )
        
        # List directory tool
        self.register_tool(
            name="list_directory",
            func=self.list_directory,
            description="List files and subdirectories in a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list"
                    }
                },
                "required": ["path"]
            }
        )
        
        # Execute Python tool
        self.register_tool(
            name="execute_python",
            func=self.execute_python,
            description="Execute Python code and return output, errors, and namespace",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variables to inject into execution namespace"
                    }
                },
                "required": ["code"]
            }
        )
        
        # Parse JSON tool
        self.register_tool(
            name="parse_json",
            func=self.parse_json,
            description="Parse JSON string into Python object",
            parameters={
                "type": "object",
                "properties": {
                    "json_string": {
                        "type": "string",
                        "description": "JSON string to parse"
                    }
                },
                "required": ["json_string"]
            }
        )
    
    # =========================================================================
    # Tool Implementations
    # =========================================================================
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read contents of a file.
        
        Args:
            path: Path to the file
            
        Returns:
            Dict with success status and content or error
        """
        try:
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine file type
            ext = os.path.splitext(path)[1].lower()
            
            result = {
                "success": True,
                "path": path,
                "content": content,
                "size": len(content),
                "extension": ext
            }
            
            # If JSON, also parse it
            if ext == '.json':
                try:
                    result["parsed"] = json.loads(content)
                except json.JSONDecodeError:
                    result["parse_error"] = "Invalid JSON"
            
            self.logger.info(f"Read file: {path} ({len(content)} chars)")
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading file {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            path: Path to the file
            content: Content to write
            
        Returns:
            Dict with success status
        """
        try:
            # Create directory if needed
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Wrote file: {path} ({len(content)} chars)")
            return {
                "success": True,
                "path": path,
                "size": len(content)
            }
            
        except Exception as e:
            self.logger.error(f"Error writing file {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_path_exists(self, path: str) -> Dict[str, Any]:
        """
        Check if a path exists.
        
        Args:
            path: Path to check
            
        Returns:
            Dict with exists status and type
        """
        exists = os.path.exists(path)
        result = {
            "path": path,
            "exists": exists
        }
        
        if exists:
            result["is_file"] = os.path.isfile(path)
            result["is_directory"] = os.path.isdir(path)
            if result["is_file"]:
                result["size"] = os.path.getsize(path)
        
        return result
    
    def list_directory(self, path: str) -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path
            
        Returns:
            Dict with files and directories
        """
        try:
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"Directory not found: {path}"
                }
            
            if not os.path.isdir(path):
                return {
                    "success": False,
                    "error": f"Not a directory: {path}"
                }
            
            items = os.listdir(path)
            files = []
            directories = []
            
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    files.append({
                        "name": item,
                        "size": os.path.getsize(item_path)
                    })
                elif os.path.isdir(item_path):
                    directories.append(item)
            
            return {
                "success": True,
                "path": path,
                "files": files,
                "directories": directories,
                "total_items": len(items)
            }
            
        except Exception as e:
            self.logger.error(f"Error listing directory {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_python(
        self,
        code: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code with provided variables.
        
        Args:
            code: Python code to execute
            variables: Variables to inject into namespace
            
        Returns:
            Dict with success, output, error, and namespace
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Build execution namespace with common imports
            exec_namespace = {
                "__builtins__": __builtins__,
            }
            
            # Add common imports
            try:
                import pandas as pd
                exec_namespace["pd"] = pd
            except ImportError:
                pass
            
            try:
                import numpy as np
                exec_namespace["np"] = np
            except ImportError:
                pass
            
            try:
                import json
                exec_namespace["json"] = json
            except ImportError:
                pass
            
            try:
                import joblib
                exec_namespace["joblib"] = joblib
            except ImportError:
                pass
            
            try:
                from datetime import datetime, timedelta
                exec_namespace["datetime"] = datetime
                exec_namespace["timedelta"] = timedelta
            except ImportError:
                pass
            
            try:
                import torch
                exec_namespace["torch"] = torch
                import torch.nn as nn
                exec_namespace["nn"] = nn
            except ImportError:
                pass
            
            try:
                from hmmlearn import hmm
                exec_namespace["hmm"] = hmm
            except ImportError:
                pass
            
            try:
                import xgboost as xgb
                exec_namespace["xgb"] = xgb
            except ImportError:
                pass
            
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                exec_namespace["StandardScaler"] = StandardScaler
                exec_namespace["train_test_split"] = train_test_split
            except ImportError:
                pass
            
            # Inject provided variables
            if variables:
                exec_namespace.update(variables)
            
            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_namespace)
            
            # Get result if defined
            result_var = exec_namespace.get("result", None)
            
            return {
                "success": True,
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": None,
                "result": result_var,
                "namespace_keys": [k for k in exec_namespace.keys() 
                                   if not k.startswith("_")]
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "result": None,
                "namespace_keys": []
            }
    
    def parse_json(self, json_string: str) -> Dict[str, Any]:
        """
        Parse JSON string.
        
        Args:
            json_string: JSON string to parse
            
        Returns:
            Dict with parsed data or error
        """
        try:
            parsed = json.loads(json_string)
            return {
                "success": True,
                "data": parsed
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON parse error: {e}"
            }
    
    # =========================================================================
    # LLM Interaction Methods
    # =========================================================================
    
    def think(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Use LLM to reason about a problem (without tools).
        
        Args:
            prompt: The question or task
            context: Optional additional context
            
        Returns:
            LLM response text
        """
        # Import here to avoid circular imports
        from llm.llm_client import Message
        
        messages = [
            Message(role="system", content=self.system_prompt)
        ]
        
        # Add context if provided
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append(Message(role="user", content=f"Context:\n{context_str}"))
        
        messages.append(Message(role="user", content=prompt))
        
        self.logger.debug(f"Thinking about: {prompt[:100]}...")
        response = self.llm_client.chat(messages)
        
        return response.content
    
    def think_with_tools(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Use LLM with tools to complete a task (ReAct pattern).
        
        Args:
            prompt: The task to complete
            context: Optional additional context
            max_iterations: Max tool execution iterations
            
        Returns:
            Dict with response and execution logs
        """
        from llm.llm_client import Message
        
        logs = []
        
        # Build initial messages
        messages = [
            Message(role="system", content=self.system_prompt)
        ]
        
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append(Message(role="user", content=f"Context:\n{context_str}"))
            logs.append("Added context to prompt")
        
        messages.append(Message(role="user", content=prompt))
        
        # Get tool definitions
        tools = self.get_tool_definitions()
        
        if not tools:
            response = self.llm_client.chat(messages)
            return {
                "success": True,
                "response": response.content,
                "logs": logs
            }
        
        self.logger.debug(f"Thinking with {len(tools)} tools...")
        logs.append(f"Starting ReAct loop with {len(tools)} tools")
        
        try:
            response = self.llm_client.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_executor=self.execute_tool
            )
            
            logs.append("ReAct loop completed")
            
            return {
                "success": True,
                "response": response.content,
                "logs": logs
            }
            
        except Exception as e:
            self.logger.error(f"Error in think_with_tools: {e}")
            return {
                "success": False,
                "response": None,
                "error": str(e),
                "logs": logs
            }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def create_response(
        self,
        success: bool,
        status: str,
        message: str,
        **kwargs
    ) -> AgentResponse:
        """
        Create a standardized AgentResponse.
        
        Args:
            success: Whether task completed successfully
            status: "COMPLETED", "BLOCKED", or "ERROR"
            message: Human-readable message
            **kwargs: Additional fields for AgentResponse
            
        Returns:
            AgentResponse instance
        """
        return AgentResponse(
            success=success,
            status=status,
            message=message,
            **kwargs
        )
    
    def extract_code_from_response(self, text: str) -> str:
        """
        Extract Python code from LLM response.
        
        Handles responses with or without markdown code blocks.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted Python code
        """
        import re
        
        # Try to extract code from markdown blocks
        code_block_pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # No code blocks found, assume entire text is code
        return text.strip()
    
    def extract_json_from_response(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Parsed JSON dict or None
        """
        import re
        
        # Try to extract JSON from markdown blocks
        json_block_pattern = r'```(?:json)?\n(.*?)\n```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to parse entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        
        return None
