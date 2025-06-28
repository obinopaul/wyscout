#!/usr/bin/env python3
"""
TelogicalClient Demo

This script demonstrates how to use the TelogicalClient to:
1. Send messages to the Telogical agent
2. Visualize the agent's reasoning process
3. Stream responses with token-by-token updates
4. Extract and display tool calls

Usage:
    python telogical_client_demo.py [--base-url URL]
    
Environment Variables:
    TELOGICAL_API_URL: Base URL for the Telogical API (default: http://localhost:8081)
"""

import asyncio
import argparse
import os
import sys
from typing import Dict, List, Any

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.client.telogical_client import TelogicalClient
from src.schema.schema import ChatMessage


def print_tool_call(tool_call: Dict[str, Any]) -> None:
    """Print a formatted tool call"""
    print(f"\n🛠️  TOOL: {tool_call.get('name', 'unknown')}")
    print(f"   Input: {tool_call.get('input', '')}")
    print(f"   Output: {tool_call.get('output', '')}")


def print_reasoning(reasoning: str) -> None:
    """Print formatted reasoning information"""
    print("\n🤔 REASONING:")
    for line in reasoning.strip().split("\n"):
        print(f"   {line}")


async def demo_streaming() -> None:
    """Demonstrate streaming with reasoning visualization"""
    print("\n=== TelogicalClient Streaming Demo ===\n")
    
    # Create client
    # Use environment variable or default
    base_url = os.getenv("TELOGICAL_API_URL", "http://localhost:8081")
    client = TelogicalClient(base_url=base_url)
    print(f"Connected to service at {client.base_url}")
    print(f"Using agent: {client.agent}")
    
    # Create message
    message = ChatMessage(
        type="human",
        content="What internet plans are available from AT&T in Dallas, TX?"
    )
    
    # Stream response with reasoning
    print("\nSending message and streaming response with reasoning...\n")
    print(f"USER: {message.content}\n")
    print("ASSISTANT: ", end="", flush=True)
    
    # Track the final content
    final_content = ""
    token_stream_started = False
    
    async for item in client.chat_stream(
        messages=[message],
        show_reasoning=True
    ):
        if isinstance(item, dict):
            # Handle reasoning or tool call info
            if item["type"] == "reasoning":
                print_reasoning(item["content"])
            elif item["type"] == "tool":
                print_tool_call(item)
        elif isinstance(item, str):
            # Handle token
            if not token_stream_started:
                token_stream_started = True
            print(item, end="", flush=True)
            final_content += item
        elif isinstance(item, ChatMessage) and item.type == "ai":
            # Handle complete message (only print if we haven't started streaming tokens)
            if not token_stream_started:
                print(item.content)
                final_content = item.content
    
    print("\n\nStream complete!")


async def demo_reasoning_extraction() -> None:
    """Demonstrate extraction of reasoning steps without streaming"""
    print("\n=== TelogicalClient Reasoning Extraction Demo ===\n")
    
    # Create client
    # Use environment variable or default
    base_url = os.getenv("TELOGICAL_API_URL", "http://localhost:8081")
    client = TelogicalClient(base_url=base_url)
    
    # Get response with reasoning steps
    query = "What are Verizon's fiber internet options in New York City?"
    print(f"USER: {query}\n")
    
    print("Getting response with reasoning steps...\n")
    result = await client.get_reasoning_steps(
        message=query,
        agent_type="telogical-assistant"
    )
    
    # Print response
    print(f"ASSISTANT: {result['response'].content}\n")
    
    # Print reasoning if available
    if result["reasoning"]:
        print_reasoning(result["reasoning"])
    
    # Print tool calls if available
    if result["tool_calls"]:
        print("\nTOOL CALLS:")
        for tool_call in result["tool_calls"]:
            print_tool_call(tool_call)


async def demo_multi_turn_conversation() -> None:
    """Demonstrate a multi-turn conversation with history tracking"""
    print("\n=== TelogicalClient Multi-turn Conversation Demo ===\n")
    
    # Create client
    # Use environment variable or default
    base_url = os.getenv("TELOGICAL_API_URL", "http://localhost:8081")
    client = TelogicalClient(base_url=base_url)
    
    # Create thread ID
    thread_id = "demo-conversation-1"
    messages: List[ChatMessage] = []
    
    # First question
    user_message1 = ChatMessage(
        type="human",
        content="What internet providers are in Chicago?"
    )
    messages.append(user_message1)
    
    print(f"USER: {user_message1.content}\n")
    response1 = await client.chat_async(
        messages=messages,
        thread_id=thread_id
    )
    messages.append(response1)
    
    print(f"ASSISTANT: {response1.content}\n")
    
    # Follow-up question
    user_message2 = ChatMessage(
        type="human",
        content="Which one offers the fastest speeds?"
    )
    messages.append(user_message2)
    
    print(f"USER: {user_message2.content}\n")
    response2 = await client.chat_async(
        messages=messages,
        thread_id=thread_id
    )
    messages.append(response2)
    
    print(f"ASSISTANT: {response2.content}\n")


async def main():
    parser = argparse.ArgumentParser(description="TelogicalClient Demo")
    parser.add_argument(
        "--demo", 
        choices=["stream", "reasoning", "conversation", "all"],
        default="all",
        help="Which demo to run"
    )
    args = parser.parse_args()
    
    if args.demo in ["stream", "all"]:
        await demo_streaming()
    
    if args.demo in ["reasoning", "all"]:
        await demo_reasoning_extraction()
    
    if args.demo in ["conversation", "all"]:
        await demo_multi_turn_conversation()


if __name__ == "__main__":
    asyncio.run(main())