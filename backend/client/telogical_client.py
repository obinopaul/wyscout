import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from backend.client.client import AgentClient, AgentClientError
from backend.schema import (
    ChatMessage,
    StreamInput,
    UserInput,
)


class TelogicalClient(AgentClient):
    """
    Extended client for interacting with Telogical Chatbot agents.
    
    Provides additional functionality for working with multi-agent workflows:
    - Reasoning visualization
    - Tool execution tracking
    - Customized streaming options
    - Detailed metadata handling
    """

    def __init__(
        self,
        base_url: str = None,
        agent: str = "telogical-assistant",  # Default to the telogical assistant
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the Telogical client.
        
        Args:
            base_url: Base URL for the Telogical API. If None, uses TELOGICAL_API_URL environment variable or defaults to http://0.0.0.0
            agent: The agent type to use (default: telogical-assistant)
            timeout: Request timeout in seconds
            get_info: Whether to get server info on initialization
        """
        if base_url is None:
            base_url = os.getenv("TELOGICAL_API_URL", "http://0.0.0.0")

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        super().__init__(base_url, agent, timeout, get_info)
        self.trace_enabled = True  # Whether to collect trace information
        
    def chat(
        self,
        messages: List[ChatMessage],
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatMessage:
        """
        Send a list of messages to the agent and get a response.
        
        Args:
            messages (List[ChatMessage]): List of messages in the conversation
            agent_type (str, optional): The type of agent to use. If None, uses the default.
            model (str, optional): The model to use for this request
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation
            
        Returns:
            ChatMessage: The response from the agent
        """
        if agent_type and agent_type != self.agent:
            self.update_agent(agent_type)
        
        if not messages:
            raise ValueError("At least one message is required")
        
        # Use the last user message as the input
        last_user_message = next((msg for msg in reversed(messages) if msg.type == "human"), None)
        if not last_user_message:
            raise ValueError("No user message found in the provided messages")
        
        # Create agent config to pass message history context
        agent_config = {
            "message_history": [msg.model_dump() for msg in messages[:-1]] if len(messages) > 1 else []
        }
        
        # Call the agent
        return self.invoke(
            message=last_user_message.content,
            model=model,
            thread_id=thread_id,
            user_id=user_id,
            agent_config=agent_config,
        )
    
    async def chat_async(
        self,
        messages: List[ChatMessage],
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatMessage:
        """
        Async version of chat method.
        
        Args:
            messages (List[ChatMessage]): List of messages in the conversation
            agent_type (str, optional): The type of agent to use. If None, uses the default.
            model (str, optional): The model to use for this request
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation
            
        Returns:
            ChatMessage: The response from the agent
        """
        if agent_type and agent_type != self.agent:
            self.update_agent(agent_type)
        
        if not messages:
            raise ValueError("At least one message is required")
        
        # Use the last user message as the input
        last_user_message = next((msg for msg in reversed(messages) if msg.type == "human"), None)
        if not last_user_message:
            raise ValueError("No user message found in the provided messages")
        
        # Create agent config to pass message history context
        agent_config = {
            "message_history": [msg.model_dump() for msg in messages[:-1]] if len(messages) > 1 else []
        }
        
        # Call the agent
        return await self.ainvoke(
            message=last_user_message.content,
            model=model,
            thread_id=thread_id,
            user_id=user_id,
            agent_config=agent_config,
        )
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stream_tokens: bool = True,
        show_reasoning: bool = True,
    ) -> AsyncGenerator[Union[ChatMessage, Dict[str, Any], str], None]:
        """
        Stream the agent's response with optional reasoning visualization.
        
        Args:
            messages (List[ChatMessage]): List of messages in the conversation
            agent_type (str, optional): The type of agent to use. If None, uses the default.
            model (str, optional): The model to use for this request
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation
            stream_tokens (bool): Whether to stream tokens as they're generated
            show_reasoning (bool): Whether to include reasoning steps in output
            
        Returns:
            AsyncGenerator: Yields messages, reasoning steps, and tokens
        """
        if agent_type and agent_type != self.agent:
            self.update_agent(agent_type)
        
        if not messages:
            raise ValueError("At least one message is required")
        
        # Use the last user message as the input
        last_user_message = next((msg for msg in reversed(messages) if msg.type == "human"), None)
        if not last_user_message:
            raise ValueError("No user message found in the provided messages")
        
        # Create agent config to pass message history context and streaming preferences
        agent_config = {
            "message_history": [msg.model_dump() for msg in messages[:-1]] if len(messages) > 1 else [],
            "show_reasoning": show_reasoning,
        }
        
        # Stream from the agent
        async for item in self.astream(
            message=last_user_message.content,
            model=model,
            thread_id=thread_id,
            user_id=user_id,
            agent_config=agent_config,
            stream_tokens=stream_tokens,
        ):
            # Process custom data information if available
            if isinstance(item, ChatMessage) and show_reasoning:
                # Check if there's trace info in custom_data
                if "trace" in item.custom_data:
                    trace_data = item.custom_data["trace"]
                    
                    # Extract reasoning if available
                    if "reasoning" in trace_data:
                        yield {
                            "type": "reasoning",
                            "content": trace_data["reasoning"]
                        }
                    
                    # Extract tool calls if available
                    if "tool_calls" in trace_data:
                        for tool_call in trace_data["tool_calls"]:
                            yield {
                                "type": "tool",
                                "name": tool_call.get("name", "unknown_tool"),
                                "input": tool_call.get("input", ""),
                                "output": tool_call.get("output", "")
                            }
            
            # Always yield the original item
            yield item

    def _extract_trace_info(self, message: ChatMessage) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Extract reasoning and tool call information from a message's custom data.
        
        Args:
            message (ChatMessage): The message to extract trace info from
            
        Returns:
            Tuple[Optional[str], List[Dict[str, Any]]]: Reasoning text and tool calls
        """
        reasoning = None
        tool_calls = []
        
        if "trace" in message.custom_data:
            trace = message.custom_data["trace"]
            reasoning = trace.get("reasoning")
            
            if "tool_calls" in trace:
                tool_calls = trace["tool_calls"]
                
        return reasoning, tool_calls
    
    async def get_reasoning_steps(
        self,
        message: str,
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed reasoning steps for an agent response without streaming.
        
        This method is useful when you want to get the full reasoning process
        at once rather than streaming it.
        
        Args:
            message (str): Message to send to the agent
            agent_type (str, optional): Type of agent to use
            model (str, optional): Model to use
            thread_id (str, optional): Thread ID for conversation
            user_id (str, optional): User ID for conversation
            
        Returns:
            Dict: Contains 'response', 'reasoning', and 'tool_calls' keys
        """
        if agent_type and agent_type != self.agent:
            self.update_agent(agent_type)
            
        # Create agent config to force trace collection
        agent_config = {
            "collect_trace": True
        }
        
        # Get response with trace information
        response = await self.ainvoke(
            message=message,
            model=model,
            thread_id=thread_id,
            user_id=user_id,
            agent_config=agent_config
        )
        
        # Extract reasoning and tool calls
        reasoning, tool_calls = self._extract_trace_info(response)
        
        return {
            "response": response,
            "reasoning": reasoning,
            "tool_calls": tool_calls
        }