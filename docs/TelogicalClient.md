# TelogicalClient

The `TelogicalClient` is an extended client for interacting with the Telogical Chatbot framework, specifically designed for working with multi-agent workflows and visualization of agent reasoning.

## Features

- **Reasoning Visualization**: Display the agent's reasoning process
- **Tool Execution Tracking**: Track and visualize tool calls
- **Customized Streaming**: Control what information is streamed
- **Message History Management**: Simplified handling of conversation history

## Installation

The TelogicalClient is part of the Telogical Chatbot framework. No separate installation is required.

## Basic Usage

```python
from src.client.telogical_client import TelogicalClient
from src.schema.schema import ChatMessage

# Create a client instance
client = TelogicalClient(base_url="http://localhost:8000")

# Create a message
message = ChatMessage(
    type="human",
    content="What internet packages does AT&T offer in Dallas?"
)

# Get a response
response = await client.chat_async(messages=[message])
print(response.content)
```

## Visualizing Agent Reasoning

The TelogicalClient can show the agent's reasoning process and tool usage. The Telogical agents now include detailed trace information in their responses, showing contextual analysis, database access, and query execution details:

```python
# Stream with reasoning visualization
async for item in client.chat_stream(
    messages=[message], 
    show_reasoning=True
):
    if isinstance(item, dict):
        # Handle reasoning or tool call info
        if item["type"] == "reasoning":
            print(f"\nREASONING: {item['content']}")
        elif item["type"] == "tool":
            print(f"\nTOOL ({item['name']}): {item['input']} -> {item['output']}")
    elif isinstance(item, str):
        # Handle token
        print(item, end="", flush=True)
    elif isinstance(item, ChatMessage):
        # Handle complete message
        print(item.content)
```

The trace information includes:

1. **Contextual Analysis**: How the agent interpreted the user's query
2. **Database Access**: Whether the agent accessed the telecom database schema
3. **Tool Calls**: Detailed information about GraphQL queries and other tools used
4. **Reasoning Steps**: The agent's step-by-step thought process

## Multi-turn Conversations

```python
# Create a thread ID to maintain conversation state
thread_id = "conversation-123"

# First message
message1 = ChatMessage(
    type="human",
    content="What internet providers are available in Chicago?"
)
response1 = await client.chat_async(messages=[message1], thread_id=thread_id)

# Follow-up message (maintains conversation context)
message2 = ChatMessage(
    type="human",
    content="Which one has the fastest speeds?"
)
messages = [message1, response1, message2]
response2 = await client.chat_async(messages=messages, thread_id=thread_id)
```

## Extracting Reasoning Steps

```python
# Get detailed reasoning without streaming
result = await client.get_reasoning_steps(
    message="What are Verizon's fiber internet options in New York City?"
)

# Access components
response = result["response"]
reasoning = result["reasoning"]
tool_calls = result["tool_calls"]
```

## API Reference

### TelogicalClient

**Constructor**

```python
TelogicalClient(
    base_url: str = "http://0.0.0.0",
    agent: str = "telogical-assistant",
    timeout: float | None = None,
    get_info: bool = True
)
```

**Methods**

#### `chat`

```python
def chat(
    self,
    messages: List[ChatMessage],
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ChatMessage
```

Synchronously send a list of messages to the agent and get a response.

#### `chat_async`

```python
async def chat_async(
    self,
    messages: List[ChatMessage],
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ChatMessage
```

Asynchronously send a list of messages to the agent and get a response.

#### `chat_stream`

```python
async def chat_stream(
    self,
    messages: List[ChatMessage],
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    stream_tokens: bool = True,
    show_reasoning: bool = True
) -> AsyncGenerator[Union[ChatMessage, Dict[str, Any], str], None]
```

Stream the agent's response with optional reasoning visualization.

#### `get_reasoning_steps`

```python
async def get_reasoning_steps(
    self,
    message: str,
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]
```

Get detailed reasoning steps for an agent response without streaming.

## Examples

See the `examples/telogical_client_demo.py` script for complete usage examples.