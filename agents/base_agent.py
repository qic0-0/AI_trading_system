"""
Base Agent class for the Quant Trading Agent System.

All agents inherit from this base class which provides:
- LLM client integration
- Tool registration and execution
- Logging
- Common utilities
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging


@dataclass
class AgentOutput:
    """Standard output format for all agents."""
    success: bool
    data: Any
    message: str
    logs: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Each agent has:
    - A name and description
    - A system prompt defining its role
    - A set of tools it can use
    - An LLM client for reasoning
    """
    
    def __init__(self, name: str, llm_client, config):
        """
        Initialize base agent.
        
        Args:
            name: Agent name (e.g., "DataAgent")
            llm_client: LLMClient instance for LLM calls
            config: SystemConfig instance
        """
        self.name = name
        self.llm_client = llm_client
        self.config = config
        self.tools = {}
        self.logger = logging.getLogger(name)
        
        # Register tools specific to this agent
        self._register_tools()
        
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        
        This defines the agent's role, capabilities, and behavior.
        Must be implemented by each agent subclass.
        """
        pass
    
    @abstractmethod
    def _register_tools(self):
        """
        Register tools available to this agent.
        
        Each agent registers its specific tools by calling:
        self.register_tool(name, function, description, parameters)
        """
        pass
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent's main task.
        
        Args:
            input_data: Input data required by this agent
            
        Returns:
            AgentOutput with results
        """
        pass
    
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
        self.logger.info(f"Registered tool: {name}")
    
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
        
        self.logger.info(f"Executing tool: {name} with args: {arguments}")
        return self.tools[name]["function"](**arguments)
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get all tool definitions for LLM function calling.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        return [tool["definition"] for tool in self.tools.values()]

    def think(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Use LLM to reason about a problem.

        Args:
            prompt: The question or task
            context: Optional additional context

        Returns:
            LLM response text
        """
        from llm.llm_client import Message

        # Build messages
        messages = [
            Message(role="system", content=self.system_prompt)
        ]

        # Add context if provided
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append(Message(role="user", content=f"Context:\n{context_str}"))

        # Add the main prompt
        messages.append(Message(role="user", content=prompt))

        # Call LLM
        self.logger.debug(f"Thinking about: {prompt[:100]}...")
        response = self.llm_client.chat(messages)

        return response.content

    def think_with_tools(
            self,
            prompt: str,
            context: Optional[Dict] = None
    ) -> AgentOutput:
        """
        Use LLM with tools to complete a task.

        Implements the ReAct pattern where the LLM can
        reason and call tools iteratively.

        Args:
            prompt: The task to complete
            context: Optional additional context

        Returns:
            AgentOutput with results
        """
        from llm.llm_client import Message

        logs = []

        # Build initial messages
        messages = [
            Message(role="system", content=self.system_prompt)
        ]

        # Add context if provided
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append(Message(role="user", content=f"Context:\n{context_str}"))
            logs.append("Added context to prompt")

        # Add the main prompt
        messages.append(Message(role="user", content=prompt))

        # Get tool definitions
        tools = self.get_tool_definitions()

        if not tools:
            # No tools, just use regular thinking
            response = self.llm_client.chat(messages)
            return AgentOutput(
                success=True,
                data={"response": response.content},
                message="Completed without tools",
                logs=logs
            )

        # Use LLM with tools (ReAct pattern)
        self.logger.debug(f"Thinking with {len(tools)} tools about: {prompt[:100]}...")
        logs.append(f"Starting ReAct loop with {len(tools)} tools")

        try:
            response = self.llm_client.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_executor=self.execute_tool
            )

            logs.append("ReAct loop completed")

            return AgentOutput(
                success=True,
                data={"response": response.content},
                message="Completed with tools",
                logs=logs
            )

        except Exception as e:
            self.logger.error(f"Error in think_with_tools: {e}")
            return AgentOutput(
                success=False,
                data={},
                message=f"Error: {str(e)}",
                logs=logs
            )
