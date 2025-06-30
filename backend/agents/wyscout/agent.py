# Must be the very first import
from __future__ import annotations

import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
import operator
import datetime
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()
import datetime

# Get the current date
current_date = datetime.date.today()

# ───────────────────────── LLM / LangGraph imports ──────────────────────────
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph_swarm.swarm import SwarmState
from functools import cache # Used for Python 3.9+
from backend.agents.wyscout.tools import *
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from backend.agents.wyscout.prompts import REFLECTION_PROMPT, MAIN_PROMPT, CONTEXTUALIZER_SYSTEM_PROMPT
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Import LLM and Memory modules from the framework
from backend.core.llm import get_telogical_primary_llm, get_telogical_secondary_llm
from backend.memory.postgres import get_telogical_postgres_saver

# Module-level variables to hold the compiled graph instances
_compiled_telogical_swarm: Optional[Any] = None
_compiled_telogical_swarm_refined: Optional[Any] = None
_compiled_introspection_agent: Optional[Any] = None
# Module-level variables to hold the agent instances (cached by initialize_agents)
_reflection_agent: Optional[Any] = None
_main_agent: Optional[Any] = None


async def get_saver() -> AsyncPostgresSaver:
    """Initializes and returns the asynchronous Postgres saver."""
    return await get_telogical_postgres_saver()

# Optional long-term store for embeddings (if used with a vector store retriever)
# store = InMemoryStore(
# index={
# "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
# "dims": 1536,
#     }
# )

# Tools for MainAgent
main_agent_tools = [
    wyscout_advanced_stats,
    wyscout_area_list,

]

# Tools for ReflectionAgent
reflection_agent_tools = [
    wyscout_advanced_stats,
    wyscout_area_list,
]

async def initialize_agents():
    """Initialize the agents only once."""
    global _reflection_agent, _main_agent
    if _reflection_agent is None:
        # Get the primary LLM from the framework
        primary_llm = get_telogical_primary_llm()
        _reflection_agent = create_react_agent(
            model=primary_llm,
            tools=reflection_agent_tools,
            name="ReflectionAgent",
            prompt=REFLECTION_PROMPT,
        )
    if _main_agent is None:
        # Get the primary LLM from the framework
        primary_llm = get_telogical_primary_llm()
        _main_agent = create_react_agent(
            model=primary_llm,
            tools=main_agent_tools,
            name="MainAgent",
            prompt=MAIN_PROMPT,
        )
    return _reflection_agent, _main_agent

# ------------------------------------ Agent 1 Graph Factory (Swarm) ------------------------------------
async def create_swarm_workflow(checkpointer):
    reflection_agent, main_agent = await initialize_agents()
    workflow = create_swarm(
        agents=[reflection_agent, main_agent],
        default_active_agent="MainAgent",
    )
    return workflow.compile(checkpointer=checkpointer)

# ------------------------------------ Agent 2 (Refined Workflow) ------------------------------------

class RefinedAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    app_output: str
    # refined_output: str
    agent_tool_outputs: Annotated[Optional[List[str]], None]
    graphql_schema: Annotated[Optional[str], None]
    internal_context_insights: Annotated[Optional[str], None] # NEW FIELD
    requires_schema_flag: Annotated[Optional[bool], None]    # For the boolean decision


class QueryContextAnalysis(BaseModel):
    contextual_insights: str = Field(
        description="3-5 concise bullet points that clarify the LATEST USER QUERY, derived from the query itself and chat history. Each bullet point starts with '* '. Do not include the original query text itself within these bullet points, only the clarifying points."
    )
    requires_database_access: bool = Field(
        description="True if the LATEST USER QUERY implies a need to consult Telogical's telecommunications database (and thus its GraphQL schema) for a factual answer. False for general conversation, greetings, questions about the AI's identity, or queries that can be answered from general knowledge without specific data lookup."
    )
    
# Global dictionary to track schema injection frequency per user/session
schema_injection_tracker: Dict[str, Dict[str, int]] = {}

MAX_TURNS_BETWEEN_SCHEMA_INJECTION = 20
MIN_TURNS_BEFORE_REINJECT_ON_KEYWORD = 3
SCHEMA_TRIGGER_KEYWORDS = [
    "schema", "table", "query", "select", "insert", "update", "delete",
    "database", "graphql", "field", "type", "id", "name", "price",
    "order", "user", "product"
]

async def async_graphql_schema() -> Dict[str, Any]:
    """Async wrapper to run the synchronous graphql_schema_tool_2."""
    loop = asyncio.get_running_loop()
    # Assuming graphql_schema_tool_2.invoke({}) is the correct way to call it for full schema
    result = await loop.run_in_executor(None, graphql_schema_tool_2.invoke, {})
    return result if isinstance(result, dict) else {"documentation": str(result)}


def _convert_to_base_message(message_data: Any) -> Optional[BaseMessage]:
    """Converts various input types to a Langchain BaseMessage object."""
    if isinstance(message_data, BaseMessage):
        return message_data
    elif isinstance(message_data, dict):
        content = message_data.get('content')
        msg_type = message_data.get('type') or message_data.get('role')

        if content is None and msg_type not in ['ai', 'assistant']: # AI messages can have empty content with tool calls
            return None # Skip messages without content unless it's an AI message with tool calls

        if msg_type in ['human', 'user']:
            return HumanMessage(content=content if content is not None else "")
        elif msg_type in ['ai', 'assistant']:
            tool_calls = message_data.get('tool_calls', [])
            return AIMessage(content=content if content is not None else "", tool_calls=tool_calls)
        elif msg_type in ['tool', 'tool_message']:
            tool_call_id = message_data.get('tool_call_id')
            name = message_data.get('name') or message_data.get('tool_name')
            if tool_call_id and name:
                return ToolMessage(content=content if content is not None else "", tool_call_id=tool_call_id, name=name)
            else: # Fallback for incomplete tool message dict
                return HumanMessage(content=str(message_data))
        elif msg_type == 'system':
            return SystemMessage(content=content if content is not None else "")
        else: # Fallback for unrecognized dict structure
            return HumanMessage(content=str(message_data))
    elif message_data is not None: # Fallback for other types
        return HumanMessage(content=str(message_data))
    return None


def _extract_string_content_from_message(message: Optional[BaseMessage]) -> str:
    """Extracts and stringifies content from a BaseMessage, prioritizing HumanMessage."""
    if not isinstance(message, HumanMessage):
        return ""

    content = message.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle multi-part content: extract and join text parts
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item.get("text", ""))
        processed_text = " ".join(text_parts).strip()
        return processed_text if processed_text else str(content) # Fallback to stringifying the list if no text parts
    elif content is not None:
        return str(content)
    return ""



async def contextualize_query_node(state: RefinedAgentState, config: RunnableConfig) -> Dict[str, Any]:
    # print("--- Entering Contextualize Query Node ---")

    processed_history_messages: List[BaseMessage] = []
    for msg_data in state.get("messages", []):
        converted_msg = _convert_to_base_message(msg_data)
        if converted_msg:
            processed_history_messages.append(converted_msg)

    if not processed_history_messages:
        return {"internal_context_insights": None, "requires_schema_flag": False} # Default

    latest_user_query_content: Optional[str] = None
    current_message_to_analyze = processed_history_messages[-1]
    if isinstance(current_message_to_analyze, HumanMessage):
        latest_user_query_content = _extract_string_content_from_message(current_message_to_analyze)
    
    if not latest_user_query_content or not latest_user_query_content.strip():
        return {"internal_context_insights": None, "requires_schema_flag": False}

    history_for_prompt_messages: List[BaseMessage] = []
    if len(processed_history_messages) > 1:
        history_for_prompt_messages = processed_history_messages[:-1]

    MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZER = 4
    if len(history_for_prompt_messages) > MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZER:
        history_for_prompt_messages = history_for_prompt_messages[-MAX_HISTORY_MESSAGES_FOR_CONTEXTUALIZER:]

    chat_history_str = "\n".join(
        [f"{msg.type.upper()}: {str(msg.content).strip()}" for msg in history_for_prompt_messages]
    ).strip()
    if not chat_history_str:
        chat_history_str = "No prior conversational history provided for this turn."
    
    prompt_input_dict = {
        "chat_history": chat_history_str,
        "latest_user_query": latest_user_query_content
    }
    
    # Get the secondary LLM from the framework
    secondary_llm = get_telogical_secondary_llm()
    # Use secondary_llm with structured output for QueryContextAnalysis
    structured_llm_contextualizer = secondary_llm.with_structured_output(QueryContextAnalysis)
    contextualizer_prompt_template = ChatPromptTemplate.from_template(CONTEXTUALIZER_SYSTEM_PROMPT)
    context_llm_chain = contextualizer_prompt_template | structured_llm_contextualizer

    formatted_insights_for_state: Optional[str] = None
    schema_needed_flag: bool = True # Default to False

    try:
        analysis_result: QueryContextAnalysis = await context_llm_chain.ainvoke(prompt_input_dict)
        # print(f"Contextualize Insights:\n{analysis_result.contextual_insights}") # Optional debug
        # print(f"Schema Needed Flag: {analysis_result.requires_database_access}") # Optional debug
        
        if analysis_result.contextual_insights and analysis_result.contextual_insights.strip():
            # Format the insights string to include the original query for clarity when it becomes a SystemMessage later
            formatted_insights_for_state = (
                f"Contextual Insights for the query: \"{latest_user_query_content}\"\n"
                f"(Derived considering chat history, if any):\n{analysis_result.contextual_insights.strip()}"
            )
        else: # Handle case where LLM might return empty insights string
            formatted_insights_for_state = (
                 f"Contextual Insights for the query: \"{latest_user_query_content}\"\n"
                 f"(No specific bullet points generated by contextualizer)"
            )
        schema_needed_flag = analysis_result.requires_database_access
        # print(f"Contextualize Node: Insights: {formatted_insights_for_state}, Requires Schema: {schema_needed_flag}")

    except Exception as e:
        print(f"Error in contextualizer LLM call or parsing structured output: {e}")
        # Provide a fallback insight message indicating failure
        formatted_insights_for_state = (
            f"Contextualization failed for query: \"{latest_user_query_content}\". Error: {e}"
        )
        schema_needed_flag = False # Default to False on error to be safe

    return {
        "internal_context_insights": formatted_insights_for_state,
        "requires_schema_flag": schema_needed_flag
    }


SCHEMA_APPENDIX_DELIMITER_START = "\n\n--- BEGIN ATTACHED GRAPHQL SCHEMA (FOR AI REFERENCE) ---\n"
SCHEMA_APPENDIX_DELIMITER_END = "\n--- END ATTACHED GRAPHQL SCHEMA ---"


# (Other imports, constants like SCHEMA_APPENDIX_DELIMITER_START/END,
# helper functions _convert_to_base_message, _extract_string_content_from_message,
# async_graphql_schema, etc., are assumed to be defined as before)

# (Ensure all necessary imports, RefinedAgentState, Pydantic models for contextualizer,
# helper functions like _convert_to_base_message, _extract_string_content_from_message,
# async_graphql_schema, and global constants like SCHEMA_TRIGGER_KEYWORDS,
# MIN_TURNS_BEFORE_REINJECT_ON_KEYWORD, MAX_TURNS_BETWEEN_SCHEMA_INJECTION,
# schema_injection_tracker, SCHEMA_APPENDIX_DELIMITER_START, SCHEMA_APPENDIX_DELIMITER_END
# are defined as in your complete code that you provided last)

async def run_app_agent_refined(state: RefinedAgentState, config: RunnableConfig) -> Dict[str, Any]:
    session_id = str(config.get("configurable", {}).get("thread_id", "default_session"))
    if session_id not in schema_injection_tracker: # schema_injection_tracker should be defined globally
        schema_injection_tracker[session_id] = {"turn_count": 0, "last_injection_turn": 0}
    schema_injection_tracker[session_id]["turn_count"] += 1
    current_turn = schema_injection_tracker[session_id]["turn_count"]

    additional_messages_for_state: List[BaseMessage] = []
    
    # --- 1. Process Incoming Messages from main graph state ---
    processed_input_messages: List[BaseMessage] = []
    for msg_data in state.get("messages", []):
        converted_msg = _convert_to_base_message(msg_data) # Ensure helper is defined
        if converted_msg:
            processed_input_messages.append(converted_msg)

    # --- 2. Get information from state (set by contextualize_query_node and previous runs) ---
    schema_to_use_for_this_run = state.get("graphql_schema")
    context_insights_str = state.get("internal_context_insights")
    # query_needs_schema_flag is retrieved but its use for appending schema to HumanMessage is now superseded by current_turn check
    # query_needs_schema_flag = state.get("requires_schema_flag", True) # As per your snippet's default

    # --- 3. Decide if the schema string in state needs to be updated from source ---
    # (This Fetch Logic remains UNCHANGED from your provided code - it updates schema_to_use_for_this_run)
    latest_user_message_content = ""
    if processed_input_messages and isinstance(processed_input_messages[-1], HumanMessage):
        latest_user_message_content = _extract_string_content_from_message(processed_input_messages[-1])

    should_fetch_new_schema_from_source = False
    if not schema_to_use_for_this_run:
        should_fetch_new_schema_from_source = True
    elif latest_user_message_content and any(k.lower() in latest_user_message_content.lower() for k in SCHEMA_TRIGGER_KEYWORDS):
        if (current_turn - schema_injection_tracker[session_id].get("last_injection_turn",0)) >= MIN_TURNS_BEFORE_REINJECT_ON_KEYWORD:
            should_fetch_new_schema_from_source = True
    elif (current_turn - schema_injection_tracker[session_id].get("last_injection_turn",0)) >= MAX_TURNS_BETWEEN_SCHEMA_INJECTION:
        should_fetch_new_schema_from_source = True

    if should_fetch_new_schema_from_source:
        try:
            schema_data = await async_graphql_schema()
            fetched_schema_str = schema_data.get("documentation")
            if fetched_schema_str and fetched_schema_str.strip():
                schema_to_use_for_this_run = fetched_schema_str
                schema_injection_tracker[session_id]["last_injection_turn"] = current_turn
            elif not schema_to_use_for_this_run:
                schema_to_use_for_this_run = None
        except Exception as e:
            print(f"Warning: Session {session_id}, Turn {current_turn}: Error fetching/updating GraphQL schema: {e}")
            if not schema_to_use_for_this_run:
                schema_to_use_for_this_run = None
    # --- End Schema Fetching Logic ---

    # --- New: Inject schema as a SystemMessage into RefinedAgentState.messages (to be returned by this node) ---
    if schema_to_use_for_this_run and schema_to_use_for_this_run.strip():
        # Conditions: Schema is available AND (it's the first turn OR schema was just freshly fetched/updated this turn)
        if current_turn == 1 or should_fetch_new_schema_from_source:
            schema_system_content = (
                f"Context: The following GraphQL schema was identified as relevant for the current turn "
                f"and is available to the agent system if needed for query formulation or understanding capabilities:\n"
                f"{SCHEMA_APPENDIX_DELIMITER_START}"
                f"{schema_to_use_for_this_run}"
                f"{SCHEMA_APPENDIX_DELIMITER_END}"
            )
            schema_system_message = SystemMessage(content=schema_system_content)
            additional_messages_for_state.append(schema_system_message)
            # print(f"DEBUG: Added schema SystemMessage to additional_messages_for_state for RefinedAgentState.")


    # --- 4. Prepare messages_for_swarm ---
    messages_for_swarm: List[BaseMessage] = []
    
    # 4a. Add Contextualizer's insights as a SystemMessage if available
    if context_insights_str:
        messages_for_swarm.append(SystemMessage(content=context_insights_str))
        # print(f"DEBUG: Prepended contextual insights to messages_for_swarm.")

    # 4b. Prepare a mutable copy of the main conversation history
    current_history_for_swarm = list(processed_input_messages)

    # 4c. Append schema to the last HumanMessage ONLY IF it's the FIRST TURN of the session.
    # The `query_needs_schema_flag` is NOT used for this decision anymore.
    # print(f"DEBUG: Current turn for schema append check: {current_turn}")
    if current_turn == 1 and schema_to_use_for_this_run and schema_to_use_for_this_run.strip():
        last_human_message_index = -1
        # Ensure there's at least one message and it's human, to avoid errors if history is unexpectedly empty
        if current_history_for_swarm: 
            for i in range(len(current_history_for_swarm) - 1, -1, -1):
                if isinstance(current_history_for_swarm[i], HumanMessage):
                    last_human_message_index = i
                    break
        
        if last_human_message_index != -1:
            original_human_message = current_history_for_swarm[last_human_message_index]
            original_human_content = _extract_string_content_from_message(original_human_message)
            
            new_human_content = (
                f"{original_human_content}"
                f"{SCHEMA_APPENDIX_DELIMITER_START}" # Ensure these delimiters are defined globally
                f"{schema_to_use_for_this_run}"
                f"{SCHEMA_APPENDIX_DELIMITER_END}"
            )
            
            modified_human_message = HumanMessage(content=new_human_content)
            if hasattr(original_human_message, 'id') and original_human_message.id:
                 modified_human_message.id = original_human_message.id
            # Add other attributes if needed (e.g., name)

            current_history_for_swarm[last_human_message_index] = modified_human_message
            # print(f"DEBUG: Schema appended to HumanMessage because current_turn == 1.")
    
    # 4d. Add the (potentially modified on first turn) history to messages_for_swarm
    messages_for_swarm.extend(current_history_for_swarm)
    
    # print(f"DEBUG: Messages for swarm prepared. Total messages: {len(messages_for_swarm)}") # Corrected print statement from your snippet
    # --- 5. Invoke the Swarm Agent ---
    compiled_graph = await dynamic_swarm()
    swarm_input_state = {"messages": messages_for_swarm}
    # Pass the current schema_to_use_for_this_run into the swarm's initial state if needed by its agents directly.
    # This is optional if the schema is only expected via the HumanMessage injection.
    # if schema_to_use_for_this_run:
    #     swarm_input_state["current_graphql_schema"] = schema_to_use_for_this_run

    result = await compiled_graph.ainvoke(swarm_input_state, config=config)

    # --- 6. Extract Output from Swarm ---
    final_messages_from_swarm = result.get("messages", [])
    current_agent_tool_outputs: List[str] = []
    app_output_content = ""
    for msg_from_swarm in final_messages_from_swarm:
        if isinstance(msg_from_swarm, ToolMessage):
            current_agent_tool_outputs.append(str(msg_from_swarm.content or ""))
        if isinstance(msg_from_swarm, AIMessage):
             app_output_content = str(msg_from_swarm.content or "")
             
    # if not app_output_content and final_messages_from_swarm: # Fallback
    #     last_msg_item = final_messages_from_swarm[-1]
    #     app_output_content = str(last_msg_item.content or "") if hasattr(last_msg_item, 'content') else str(last_msg_item)
        
    if not app_output_content and final_messages_from_swarm: # Fallback if AIMessage wasn't last or was empty
        # Try to get content from the last message if it's an AIMessage
        if isinstance(final_messages_from_swarm[-1], AIMessage):
             app_output_content = str(final_messages_from_swarm[-1].content or "")
        else: # If last message is not AI, or if still no content, stringify the last message as a last resort.
             last_msg_item = final_messages_from_swarm[-1]
             app_output_content = str(last_msg_item.content or "") if hasattr(last_msg_item, 'content') else str(last_msg_item)


    # --- New: (Commented Out) Logic to Inject Swarm Internal Messages into RefinedAgentState ---
    # --- This would add all messages from the swarm's execution to the main graph's history ---
    # # relevant_swarm_output_messages: List[BaseMessage] = []
    # # if final_messages_from_swarm:
    # #     # This SystemMessage helps delineate the messages from the swarm in the main history.
    # #     announcement_start_message = SystemMessage(
    # #         content="--- BEGIN: Messages from internal agent (Swarm) execution ---"
    # #     )
    # #     relevant_swarm_output_messages.append(announcement_start_message)
    # #
    # #     # Add all messages from the swarm's execution.
    # #     # These include the initial human message (possibly with schema),
    # #     # tool calls, tool responses, and the swarm's final AIMessage.
    # #     relevant_swarm_output_messages.extend(final_messages_from_swarm)
    # #
    # #     announcement_end_message = SystemMessage(
    # #         content="--- END: Messages from internal agent (Swarm) execution ---"
    # #     )
    # #     relevant_swarm_output_messages.append(announcement_end_message)
    # #
    # #     additional_messages_for_state.extend(relevant_swarm_output_messages)
    # #     # print(f"DEBUG: (Commented out) Added {len(relevant_swarm_output_messages)} swarm messages to additional_messages_for_state.")
    # --- End of (Commented Out) Logic ---


    # --- 7. Collect reasoning steps and tool outputs for tracing ---
    # Extract reasoning steps from context insights
    reasoning_steps = []
    
    # Add contextual analysis step if available
    if state.get("internal_context_insights"):
        reasoning_steps.append(f"Contextual analysis: {state.get('internal_context_insights')}")
        
    # Add schema usage step if applicable
    if schema_to_use_for_this_run:
        reasoning_steps.append("Accessed database schema for telecommunications data")
        
    # Format agent tool outputs for tracing
    formatted_tool_outputs = []
    for i, tool_output in enumerate(current_agent_tool_outputs):
        formatted_tool_outputs.append({
            "name": f"GraphQLTool-{i+1}",
            "input": "Query execution",
            "output": tool_output
        })
            
    # Create structured tool output objects
    detailed_tool_outputs = formatted_tool_outputs
    
    # Combine reasoning steps into a single narrative
    reasoning_narrative = "\n".join(reasoning_steps)
    if not reasoning_narrative:
        reasoning_narrative = "Agent processed user query through multi-step workflow"
    
    # --- 8. Return results ---
    # The 'messages' key here will ensure LangGraph appends additional_messages_for_state
    # to the RefinedAgentState.messages via the operator.add mechanism.
    return {
        "app_output": app_output_content,
        "agent_tool_outputs": detailed_tool_outputs,  # Use the formatted tool outputs
        "graphql_schema": schema_to_use_for_this_run,      # Persist the schema string
        "internal_context_insights": reasoning_narrative,  # Store reasoning for refine_output_refined
        "requires_schema_flag": None,     # Clear for next cycle
        "messages": additional_messages_for_state           # Add collected messages to state
    }

# async def run_app_agent_refined(state: RefinedAgentState, config: RunnableConfig) -> Dict[str, Any]:
#     """Runs the primary application agent with dynamic schema injection and robust message handling."""
#     session_id = str(config.get("configurable", {}).get("thread_id", "default_session"))

#     if session_id not in schema_injection_tracker:
#         schema_injection_tracker[session_id] = {"turn_count": 0, "last_injection_turn": 0}

#     schema_injection_tracker[session_id]["turn_count"] += 1
#     current_turn = schema_injection_tracker[session_id]["turn_count"]

#     # --- Manage GraphQL Schema in State ---
#     current_graphql_schema_str = state.get("graphql_schema")
#     if not current_graphql_schema_str:
#         try:
#             schema_data = await async_graphql_schema()
#             fetched_schema_str = schema_data.get("documentation")
#             if fetched_schema_str:
#                 current_graphql_schema_str = fetched_schema_str
#         except Exception as e:
#             print(f"Warning: Error fetching GraphQL schema: {e}")
#             current_graphql_schema_str = None # Ensure it's None on error
            
#     # print(f"Current GraphQL Schema: {current_graphql_schema_str}")
#     # --- Process Incoming Messages ---
#     processed_messages: List[BaseMessage] = []
#     for msg_data in state.get("messages", []):
#         converted_msg = _convert_to_base_message(msg_data)
#         if converted_msg:
#             processed_messages.append(converted_msg)

#     # --- Determine if schema should be fetched and injected ---
#     latest_user_message_content = ""
#     last_human_message_obj: Optional[HumanMessage] = None

#     for msg in reversed(processed_messages):
#         if isinstance(msg, HumanMessage):
#             last_human_message_obj = msg
#             latest_user_message_content = _extract_string_content_from_message(msg)
#             break
#     # print(f"Latest User Message Content: {latest_user_message_content}")
#     fetch_and_inject_schema_this_turn = False
#     if current_turn == 1:
#         fetch_and_inject_schema_this_turn = True
#     else:
#         if any(keyword.lower() in latest_user_message_content.lower() for keyword in SCHEMA_TRIGGER_KEYWORDS):
#             if (current_turn - schema_injection_tracker[session_id]["last_injection_turn"]) >= MIN_TURNS_BEFORE_REINJECT_ON_KEYWORD:
#                 fetch_and_inject_schema_this_turn = True
        
#         if not fetch_and_inject_schema_this_turn:
#             turns_since_last_injection = current_turn - schema_injection_tracker[session_id]["last_injection_turn"]
#             if turns_since_last_injection >= MAX_TURNS_BETWEEN_SCHEMA_INJECTION:
#                 fetch_and_inject_schema_this_turn = True

#     schema_context_to_inject = ""
#     if fetch_and_inject_schema_this_turn:
#         try:
#             schema_data = await async_graphql_schema() # Fetch fresh schema
#             fetched_schema_str = schema_data.get("documentation")
#             if fetched_schema_str:
#                 schema_context_to_inject = f"\n\n-- RELEVANT DATABASE SCHEMA --\n{fetched_schema_str}\n-- END SCHEMA --\n\n"
#                 current_graphql_schema_str = fetched_schema_str # Update schema in state for next turn
#                 schema_injection_tracker[session_id]["last_injection_turn"] = current_turn
#         except Exception as e:
#             print(f"Warning: Error fetching dynamic schema for injection: {e}")

#     messages_for_swarm = list(processed_messages) # Create a mutable copy
#     if schema_context_to_inject and last_human_message_obj:
#         # Find the instance of the last human message in the copied list and append to its content
#         try:
#             # We need to modify the actual object in the list that will be passed to the swarm
#             index_of_last_human = -1
#             for i, m in reversed(list(enumerate(messages_for_swarm))):
#                  if m is last_human_message_obj: # Check for object identity
#                     index_of_last_human = i
#                     break
            
#             if index_of_last_human != -1:
#                 original_content = _extract_string_content_from_message(messages_for_swarm[index_of_last_human])
#                 messages_for_swarm[index_of_last_human] = HumanMessage(
#                     content=original_content + schema_context_to_inject
#                 )
#             else: # Fallback if object identity check fails (should not happen if processed_messages contains it)
#                  # This indicates last_human_message_obj was not from processed_messages or list was modified
#                 #  print("Warning: Could not find the exact last human message object to append schema. Appending to a new message.")
#                  if messages_for_swarm and isinstance(messages_for_swarm[-1], HumanMessage):
#                     messages_for_swarm[-1].content += schema_context_to_inject
#                  else: # less ideal, adds schema as a new message or to a non-human message
#                     messages_for_swarm.append(HumanMessage(content=schema_context_to_inject))

#         except Exception as e:
#             print(f"Error appending schema to HumanMessage content: {e}")


#     # --- Invoke the Swarm Agent ---
#     compiled_graph = await dynamic_swarm()
#     swarm_input_state = {"messages": messages_for_swarm}
#     if current_graphql_schema_str: # Make schema available if sub-graph needs it directly in its state
#         swarm_input_state["current_graphql_schema"] = current_graphql_schema_str
        
#     result = await compiled_graph.ainvoke(swarm_input_state, config=config)

#     # --- Extract Output and Tool Messages ---
#     final_messages_from_swarm = result.get("messages", [])
#     current_agent_tool_outputs: List[str] = []
#     app_output_content = ""

#     for msg in final_messages_from_swarm: # These should be BaseMessage objects from a well-behaved swarm
#         if isinstance(msg, ToolMessage):
#             current_agent_tool_outputs.append(str(msg.content) if msg.content is not None else "")
#         # The last AIMessage is usually the one to refine
#         if isinstance(msg, AIMessage): # Or any non-Tool, non-System message that represents agent's response
#              app_output_content = str(msg.content) if msg.content is not None else ""


#     if not app_output_content and final_messages_from_swarm: # Fallback: get content from the very last message
#         last_msg_item = final_messages_from_swarm[-1]
#         if isinstance(last_msg_item, BaseMessage):
#             app_output_content = str(last_msg_item.content) if last_msg_item.content is not None else ""
#         else: # Should not happen if swarm returns BaseMessages
#             app_output_content = str(last_msg_item)

#     # print(f"Final App Output Content: {app_output_content}")
#     # print(f"Current Agent Tool Outputs: {current_agent_tool_outputs}")
#     # print(f"Current GraphQL Schema: {current_graphql_schema_str}")
    
#     return {
#         "app_output": app_output_content,
#         "agent_tool_outputs": current_agent_tool_outputs,
#         "graphql_schema": current_graphql_schema_str # Persist schema if it was updated
#         # Note: This node does not directly update RefinedAgentState.messages
#         # The graph's `operator.add` handles messages returned by `refine_output`
#     }

# ------------------------------------ Refined Agent Workflow Components ------------------------------------
class RefinedOutput(BaseModel):
    refined_text: str = Field(description="The final, polished text after removing AI reasoning, workflow narratives, and ensuring it aligns with Telogical's voice. This field should contain only the core information intended for the user, with original formatting and detail preserved.")

async def refine_output_refined(state: RefinedAgentState) -> Dict[str, Any]:
    app_output_to_refine = state["app_output"]
    agent_tool_outputs: List[str] = state.get("agent_tool_outputs") or []
    
    LATEST_MESSAGE_TAG = "[LATEST_MESSAGE] "
    last_human_query_content = "No specific user query found for context."

    # Extract last human message from the accumulated history in RefinedAgentState
    # This history includes messages from before run_app_agent_refined was called
    temp_messages: List[BaseMessage] = []
    for msg_data in state.get("messages", []): # state.messages is the graph-level history
        converted = _convert_to_base_message(msg_data)
        if converted:
            temp_messages.append(converted)
            
    for msg in reversed(temp_messages):
        if isinstance(msg, HumanMessage):
            content_str = _extract_string_content_from_message(msg)
            if content_str.startswith(LATEST_MESSAGE_TAG):
                last_human_query_content = content_str[len(LATEST_MESSAGE_TAG):]
            else:
                last_human_query_content = content_str
            break

    formatted_tool_outputs = "No tool outputs were recorded or applicable for the previous agent step."
    if agent_tool_outputs:
        formatted_tool_outputs = "Tool Outputs from Previous Agent Step:\n" + "\n".join(
            f"{i+1}. {output_content}" for i, output_content in enumerate(agent_tool_outputs)
        )
    elif isinstance(agent_tool_outputs, list) and not agent_tool_outputs:
        formatted_tool_outputs = "The previous agent step recorded that no tools were used or no outputs were generated from tools."

    # print(f"Last Human Query Content: {last_human_query_content}")
    # print(f"Formatted Tool Outputs: {formatted_tool_outputs}")
    
    # (Refiner prompt remains the same as it was extensive and specific)
    # ... [Your existing long refiner_prompt SystemMessage content] ...
    # For brevity in this response, I'm not repeating the giant prompt string.
    # Assume REFINE_SYSTEM_PROMPT_CONTENT contains that long string.
    
    REFINE_SYSTEM_PROMPT_CONTENT = """ 
    You are the **Refinement Specialist** for Telogical Systems' AI assistant. Your designated role is to meticulously process the output generated by the primary AI analysis component (`Agent Output to Refine`), ensuring that every user-facing response is impeccably polished, professional, and authentically represents the voice and high standards of Telogical Systems LLC. Consider yourself the final quality assurance step, transforming detailed analytical output into a refined, client-ready communication that directly reflects Telogical's expertise in providing exhaustive and precise data.
    
    Your primary task is to **refine and enhance** the provided `Agent Output to Refine` using the `Original Query` for context and the `Supporting Tool Outputs` as the ground truth for data. This means you must **remove specific unwanted elements** (AI thinking processes) while **preserving and often augmenting the integrity, detail, and factual substance of the information**. **Crucially, all refinements must ensure the final output remains a direct, relevant, and coherent answer to the `Original Query` that prompted the agent's output.** The goal is to produce responses that present information as Telogical's official product,  mirroring the thoroughness of a senior data analyst and appropriately addressing the user's `Original Query`, not as an AI searching a database or narrating its thought process.

    **Key Refinement Guidelines:**

    1.  **Client Priority: Preserve All Core Information, Detail, and Original Formatting Faithfully:**
        * Your paramount goal is to present the original information with **absolute accuracy and maximum completeness as supported by the `Supporting Tool Outputs`**. Telogical's clients prioritize rich, detailed, and meticulously structured data over brevity or summarization. They expect the full depth of data available, akin to what a senior data analyst would provide from raw data sources.
        * **ABSOLUTELY DO NOT** summarize, paraphrase, re-interpret (beyond correcting errors against tool outputs), condense, or otherwise alter the core data, factual details, or the substance of the information provided in the agent's output if they accurately reflect the `Supporting Tool Outputs`. Your task is **not** to "improve" or "re-word" the factual content if it is already clear and accurate against the source data; your role is to ensure it is presented cleanly as if from Telogical Systems directly. Any attempt to "make it more product-friendly" by reducing detail is counter to the client's needs.
        * **MAINTAIN THE ORIGINAL FORMATTING AND LEVEL OF DETAIL WITH EXTREME FIDELITY, ESPECIALLY FOR TABLES, LISTS, AND ALL FORMS OF STRUCTURED DATA.** If the `Agent Output to Refine` includes tables, lists, code blocks, specific paragraph structures, or any other structured formatting containing factual information, these structures **MUST be replicated exactly as they were**, minus only the AI thought process elements. For example, if a table is provided, reproduce the table in its entirety and its original structure; **do not** convert it into a prose summary or alter its presentation in a way that reduces detail.
        * Think of yourself as meticulously cleaning the "frame" (AI thought process, conversational fluff) around a picture (the core data and its original presentation structure). The picture itself—its detail, substance, and all essential formatting (like table structures, itemized lists, and their full content)—must remain entirely untouched and unreduced. Your refinement must not lead to *any* loss of information or any change in how detailed or structured information was originally conveyed. This commitment to comprehensive detail and faithful reproduction of format is a key client expectation.

    2.  **Eliminate AI Thinking Processes and Workflow Narratives:**
        * **REMOVE** all traces of the AI's internal reasoning or how it arrived at the information. This includes:
            * Step-by-step explanations of its thought process.
            * Internal deliberations or self-correction narratives.
            * Descriptions of search methodologies, queries used, or data retrieval steps.
            * Any language suggesting the AI is actively thinking, searching, or processing (e.g., "let me think about this," "I need to consider...," "first I'll look at...").

    3.  **Adopt Telogical's Official Voice (Direct and Authoritative):**
        * Present all information as definitive facts originating directly from Telogical Systems.
        * **NEVER** use phrases that attribute the information to a database, a search process, or the AI's own discovery (e.g., "According to the database...", "I searched for...")..
        * **AVOID** phrases such as:
            * "According to the database..."
            * "Based on my queries..."
            * "I searched for..."
            * "I found in the data..."
            * "Based on the telecom package data..."
            * "I checked the latest available package..."
            * "My attempts to explore the database..."
            * "According to the information I got from..."
        * **INSTEAD, use direct and authoritative statements** like:
            * "Here are the 2025 promotions..."
            * "The latest package and promotion information for AT&T shows..."
            * "We are experiencing technical difficulties..." (if applicable to the original message)
            * "Here’s what we found regarding the most popular internet plans in Charleston, SC (zip code 29056):"
            * "Here are the results..."
            * "This information shows..."
            
    4.  **Advanced Scrutiny, Validation, and Enrichment Based on Tool Outputs:**
        * You will be given the `Agent Output to Refine` and will always receive `Supporting Tool Outputs from Agent's Process`. These tool outputs are the **ground truth** for factual information.
        * Your primary responsibility in this step is to critically compare the `Agent Output to Refine` against the `Supporting Tool Outputs`.

        * **A. Data Validation and Correction:**
            * Meticulously cross-validate all factual claims, details, and data points in the `Agent Output to Refine` against the information present in the `Supporting Tool Outputs`.
            * **Extreme Vigilance on Numerical Accuracy (CRITICAL):** All numerical data (prices, fees, speeds, channel counts, item counts, etc.) presented in the `Agent Output to Refine` **MUST** be cross-validated against the `Supporting Tool Outputs`. Any discrepancies **MUST be corrected** to precisely reflect the `Supporting Tool Outputs`. **Inaccurate reporting of numerical data, especially prices, is a critical failure and severely damages company reputation.**
            * **Report Numerical Data As-Is and Unaggregated:** Present all numerical figures, particularly prices and their individual components (e.g., standard price, promotional price, one-time fees, recurring fees, taxes, discounts), **exactly as they appear in the `Supporting Tool Outputs`**. **DO NOT SUM, AGGREGATE, OR SIMPLIFY** numerical data (e.g., do not combine a base price and multiple promotional discounts into a single 'final price' unless the `Supporting Tool Outputs` explicitly present it that way with full, transparent context of its derivation). Each component should be listed separately to provide complete transparency, mirroring how a data analyst presents detailed findings. **Avoid any form of calculation or summarization of numerical data unless explicitly instructed by the user's query AND clearly supported by the tool output's structure.**

        * **B. Enhancing for Completeness and Volume (Adding Missed Data):**
            * If the `Agent Output to Refine` has omitted relevant information, data points, or details that are **explicitly present in the `Supporting Tool Outputs`** and are pertinent to the `Original Query`, you **MUST** incorporate these missing pieces into your `refined_text`.
            * The goal is to provide the **most comprehensive and voluminous response supported by the available tool data**. For instance, if the tool output lists 20 relevant markets and the `Agent Output to Refine` only mentions 10, your refined response must include all 20 markets along with their associated details as found in the tool output.
            * This includes ensuring that if the data from tools pertains to specific geographical scopes like DMAs (Designated Market Areas), uses specific telecommunications terminology, or lists multiple options/variations, this level of detail and correct terminology is fully preserved and clearly presented in the final output.

        * **C. Clarification and Interpretation:**
            * If the agent's interpretation of tool data in `Agent Output to Refine` is unclear, slightly misaligned, or poorly structured compared to the richness of the `Supporting Tool Outputs`, you must clarify and restructure it. Your refined version should accurately and clearly represent the information as found in the `Supporting Tool Outputs`, always maintaining the authoritative Telogical voice and adhering to the principle of providing complete data.

        * **D. Constraint:** You are **not** to re-run any tools or seek new information beyond what is provided in `Agent Output to Refine` and `Supporting Tool Outputs`. Your refinement is strictly limited to these inputs.

    - **ALL TELECOMMUNICATIONS DATA IS CURRENT AS OF: {current_date}**
    
    Your role is to be the definitive voice of Telogical Systems. Filter out the AI's procedural explanations and internal monologue, delivering clear, authoritative information about telecommunications market data. The goal is to provide user responses that sound like polished, customer-facing product outputs, without revealing the AI or technical querying behind the scenes.
    
    Your final response should be structured to contain only the refined text. This will be captured in a field named `refined_text`.
    
    **Output Structure:**
    - refined_text: The final, polished text after removing AI reasoning, workflow narratives, and ensuring it aligns with Telogical's voice. This field should contain only the core information intended for the user, with original formatting and detail preserved.
    
    **Example Output:**
    - refined_text: "Here are the 2025 promotions for AT&T in Charleston, SC (zip code 29056):\n\n- AT&T Fiber 300: $55/month\n- AT&T Fiber 1000: $70/month\n- AT&T Internet 100: $50/month\n\nThese packages include...\n\n---"
    """

    refiner_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=REFINE_SYSTEM_PROMPT_CONTENT),
        HumanMessage(content=f"Original Query: {last_human_query_content}\n\nAgent Output to Refine: {app_output_to_refine}\n\nSupporting Tool Outputs from Agent's Process: {formatted_tool_outputs}"),
    ])

    # Get the primary LLM from the framework
    primary_llm = get_telogical_primary_llm()
    structured_llm = primary_llm.with_structured_output(RefinedOutput)
    chain = refiner_prompt_template | structured_llm
    
    try:
        response_structured = await chain.ainvoke({})
        if isinstance(response_structured, RefinedOutput):
            refined_content = response_structured.refined_text
        else: # Fallback if structured output fails unexpectedly
            refined_content = f"Error: Refinement failed. Unexpected response type: {type(response_structured)}. Output: {str(app_output_to_refine)}"
            print(f"Warning: Refiner LLM did not return RefinedOutput. Received: {response_structured}")
    except Exception as e:
        print(f"Error during refinement LLM call: {e}")
        refined_content = f"Error: Refinement process encountered an exception. Original output: {str(app_output_to_refine)}"


    # Create the final AI message with trace information to enable reasoning visualization
    final_ai_message = AIMessage(
        content=refined_content,
        custom_data={
            "trace": {
                "reasoning": state.get("internal_context_insights", "Agent processed user query through multi-step workflow."),
                "tool_calls": state.get("agent_tool_outputs", []) if state.get("agent_tool_outputs") else []
            }
        }
    )

    return {
        # "refined_output": refined_content,
        "messages": [final_ai_message] # This will be added to the state's messages
    }

async def create_refined_agent_workflow(checkpointer):
    workflow = StateGraph(RefinedAgentState)
    workflow.add_node("contextualize_query", contextualize_query_node) # New node
    workflow.add_node("app_agent", run_app_agent_refined)
    workflow.add_node("refine_output", refine_output_refined)
    
    workflow.add_edge(START, "contextualize_query")
    workflow.add_edge("contextualize_query", "app_agent")
    workflow.add_edge("app_agent", "refine_output")
    workflow.add_edge("refine_output", END)
    return workflow.compile(checkpointer=checkpointer)




# # ------------------------------------------------- Introspection Agent -------------------------------------------------
# # (Introspection agent system message and factory functions remain largely the same as per original)
# # For brevity, I'm not repeating the long introspection_system_message_content here.
# # Assume INTROSPECTION_SYSTEM_MESSAGE_CONTENT holds that string.
# INTROSPECTION_SYSTEM_MESSAGE_CONTENT = """
# You are an introspection agent focused solely on generating accurate GraphQL code based on user input. When a user asks a question, analyze the schema and return the relevant GraphQL query, mutation, or introspection code.

# Your only objective is to:
#     - Analyze the question or context provided by the user.
#     - Use available introspection tools to understand the GraphQL schema.
#     - Return the corresponding GraphQL code necessary to fulfill the user’s query.

# TOOL USAGE PROTOCOL:
# - Your first priority when interacting with the GraphQL database is to **understand the database schema**. This schema information, including available queries, types, fields, required parameters, and their exact data types (Int, String, Boolean, etc.), might be provided to you in the user's message or obtained by using the `graphql_schema_tool_2` tool. If you are unsure of the schema or specific details needed for a query, use the `graphql_schema_tool_2` tool (`graphql_schema_tool_2`) to retrieve this information. Perform detailed introspection if needed to clarify parameter requirements. Avoid guessing schema details if you are uncertain and the schema hasn't been provided.
# - Make sure you understand all the available queries, what they do and their parameters before attempting to formulate any GraphQL queries. Some queries may look the same but they actually perform different functions. This is crucial for ensuring that your queries are valid and will return the expected results.
# - If a user's request involves a location and you do not know the required zip code for that location, the introspection schema provides a fetchLocationDetails query where you can input a location information and get a representative zip code for that location to use for further queries. You should obtain it *before* attempting to formulate GraphQL queries that might require this information. If you already know the zip code, you do not need to run this query.
# - When working with DMA (Designated Market Area) codes, use the dma_code_lookup_tool to convert numerical DMA codes to their human-readable market names. This is essential for presenting telecom market data in a user-friendly format. Always use this tool to translate DMA codes before presenting final results to users.
# - The fetchMyCurrentPackages queries should not be trusted, and if possible do not use this query, as there are known issues with it. There are several alternative queries that can be used to get the same information.
# - Once you understand the database schema (either from introspection results or prior knowledge) and have any necessary location data (like a zip code), formulate the required GraphQL queries and use the `parallel_graphql_executor` (`parallel_graphql_executor`) to fetch the data efficiently. This tool takes a list of queries.
# - **CRITICAL QUERY FORMULATION GUIDELINES:**
#     - **Strict Schema Adherence:** When formulating GraphQL queries, you MUST strictly adhere to the schema structure revealed by `graphql_introspection`. Only include fields and parameters that are explicitly defined in the schema for the specific query or type you are interacting with. DO NOT add parameters that do not exist or that belong to different fields/types.
#     - **Accurate Data Types:** Pay extremely close attention to the data types required for each parameter as specified in the schema (e.g., String, Int, Float, Boolean, ID, specific Enums, etc.). Ensure that the values you provide in your queries EXACTLY match the expected data type. For example, if a parameter requires an `Int`, provide an integer value, not a string representation of an integer, and vice versa.
# - BEFORE using any tool, EXPLICITLY state:
#     1. WHY you are using this tool (connect it to the user's request and the overall plan).
#     2. WHAT specific information you hope to retrieve/achieve with this tool call.
#     3. HOW this information will help solve the user's task.

# --------------------------------------------------------------------------------
# TOOL DESCRIPTIONS & EXPLANATIONS

# 1) graphql_schema_tool_2:
#     - Description: Performs introspection queries on the GraphQL database schema to explore its structure.
#     - Usage: Call this tool *first* if you are unfamiliar with the structure of the GraphQL database schema. Use it to explore available queries, types, and fields. This step is essential for formulating correct queries for the `parallel_graphql_executor`. Available query types are 'full_schema', 'types_only', 'queries_only', 'mutations_only', and 'type_details'. If using 'type_details', you must also provide a 'type_name'. Once you understand the schema, you do not need to use this tool again for general schema exploration unless specifically asked or needing details about a new type.
#     - Output: Returns information about the GraphQL schema based on the requested query type.


# Key points to remember:
#     1. Do not explain the schema — Only generate and return the exact GraphQL code needed.
#     2. Do not attempt to analyze or reason beyond returning the code — Just focus on providing the correct GraphQL query or mutation.
#     3. Ensure the GraphQL code you return is syntactically correct, clear, and executable.
#     4. If multiple GraphQL queries are required, return them in sequence, with clear step-by-step instructions (e.g., "Run the first query, then based on the result, run the following query").
#     5. If you can’t generate the code or need to reference previous queries for the next step, provide instructions like: "Run the first query to get the necessary data and then proceed with the second query."
#     6. Provide brief notes or steps before the code to clarify when multiple queries are involved. The descriptions should be concise but enough to make the process clear.

# In this context:
#     - Use GraphQL Schema Tool 2 for general queries related to the schema.
#     - If necessary, mention any dependencies between queries and provide instructions for their execution.

# When you respond, return the following:
#     - GraphQL code or codes (directly executable).
#     - Brief, clear steps for queries when multiple queries are required.
#     - Any notes or instructions that help clarify the process of executing these queries.

# - **Note**: When referencing competitors in the graphql query, always ensure the competitor name is input exactly as listed below (e.g., "Cox Communications" instead of "Cox"). The format must match the exact wording in the database for accurate querying.

# 3 Rivers Communications, Access, Adams Cable Service, Adams Fiber, ADT, AireBeam, Alaska Communications, Alaska Power & Telephone,
# Allband Communications Cooperative, Alliance Communications, ALLO Communications, altafiber, Altitude Communications, Amazon,
# Amherst Communications, Apple TV+, Armstrong, Arvig, Ashland Fiber Network, ASTAC, Astound Broadband, AT&T, BAM Broadband, Bay Alarm,
# Bay Country Communications, BBT, Beamspeed Cable, Bee Line Cable, Beehive Broadband, BEK Communications, Benton Ridge Telephone, 
# Beresford Municipal Telephone Company, Blackfoot Communications, Blue by ADT, Blue Ridge Communications, Blue Valley Tele Communications, 
# Bluepeak, Boomerang, Boost Mobile, Breezeline, Brightspeed, BRINKS Home Security, Bristol Tennessee Essential Services, Buckeye Broadband, 
# Burlington Telecom, C Spire, CAS Cable, Castle Cable, Cedar Falls Utilities, Central Texas Telephone Cooperative, Centranet, CenturyLink, 
# Chariton Valley, Charter, Circle Fiber, City of Hillsboro, ClearFiber, Clearwave Fiber, Co-Mo Connect, Comcast, Comporium, 
# Concord Light Broadband, Consolidated Communications, Consolidated Telcom, Consumer Cellular, Copper Valley Telecom, Cordova Telephone Cooperative, 
# Cox Communications, Craw-Kan Telephone Cooperative, Cricket, Delhi Telephone Company, Dickey Rural Network, Direct Communications, DIRECTV, 
# DIRECTV STREAM, discovery+, DISH, Disney+, Disney+ ESPN+ Hulu, Disney+ Hulu Max, Dobson Fiber, Douglas Fast Net, ECFiber, Elevate, Empire Access, 
# empower, EPB, ESPN+, Etex Telephone Cooperative, Ezee Fiber, Farmers Telecommunications Cooperative, Farmers Telephone Cooperative, FastBridge Fiber, 
# Fastwyre Broadband, FCC, FiberFirst, FiberLight, Fidium Fiber, Filer Mutual Telephone Company, Five Area Telephone Cooperative, FOCUS Broadband, 
# Fort Collins Connexion, Fort Randall Telephone Company, Frankfort Plant Board, Franklin Telephone, Frontier, Frontpoint, Fubo, GBT, GCI, Gibson Connect, 
# GigabitNow, Glo Fiber, Golden West, GoNetspeed, Google Fi Wireless, Google Fiber, Google Nest, GoSmart Mobile, Grant County PowerNet, 
# Great Plains Communications, Guardian Protection Services, GVTC, GWI, Haefele Connect, Hallmark, Halstad Telephone Company, Hamilton Telecommunications, 
# Hargray, Hawaiian Telcom, HBO, Home Telecom, Honest Networks, Hotwire Communications, HTC Horry Telephone, Hulu, i3 Broadband, IdeaTek, ImOn Communications, 
# Inland Networks, Internet Subsidy, IQ Fiber, Iron River Cable, Jackson Energy Authority, Jamadots, Kaleva Telephone Company, Ketchikan Public Utilities, 
# KUB Fiber, LFT Fiber, Lifetime, Lightcurve, Lincoln Telephone Company, LiveOak Fiber, Longmont Power & Communications, Loop Internet, Lumos, 
# Mahaska Communications, Margaretville Telephone Company, Matanuska Telephone Association, Max, MaxxSouth Broadband, Mediacom, Metro by T-Mobile, 
# Metronet, Michigan Cable Partners, Mid-Hudson Fiber, Mid-Rivers Communications, Midco, Mint Mobile, MLB.TV, MLGC, Montana Opticom, Moosehead Cable, 
# Muscatine Power and Water, NBA League Pass, Nemont, NEMR Telecom, Netflix, NFL+, NineStar Connect, NKTelco, North Dakota Telephone Company, 
# Northern Valley Communications, Nuvera, OEC Fiber, Ogden Telephone Company, Omnitel, OneSource Communications, Ooma, Optimum, OzarksGo, 
# Ozona Cable & Broadband, Page Plus, Palmetto Rural Telephone Cooperative, Panhandle Telephone Cooperative, Paragould Municipal Utilities, Paramount+, 
# Parish Communications, Passcom Cable, Paul Bunyan Communications, Pavlov Media, Peacock, Philo, Phonoscope, Pineland Telephone Cooperative, 
# Pioneer Broadband, Pioneer Communications, Pioneer Telephone Cooperative, Plateau, Point Broadband, Polar Communications, Port Networks, Premier Communications, 
# Project Mutual Telephone, Protection 1, Pulse, Quantum Internet & Telephone, Race Communications, Range Telephone Cooperative, Reach Mobile, 
# REV, RightFiber, Ring, Ripple Fiber, Rise Broadband, Ritter Communications, RTC Networks, Salsgiver Telecom, Santel Communications, SC Broadband, 
# SECOM, Service Electric, Shentel, Silver Star Communications, SIMPLE Mobile, SimpliSafe, Sling TV, Smithville Fiber, Snip Internet, Solarus, 
# Sonic, South Central Rural Telecommunications, Southern Montana Telephone, Spanish Fork Community Network, Sparklight, SpitWSpots, 
# Spring Creek Cable, Spruce Knob Seneca Rocks Telephone, SRT Communications, Starry, Starz, Sterling LAMB (Local Area Municipal Broadband), 
# Straight Talk Wireless, StratusIQ, Sundance, Surf Internet, SwyftConnect, Syntrio, T-Mobile, TCT, TDS, TEC, Telogical, Ting, Total Wireless, 
# TPx, Tracfone, Tri-County Communications, Triangle Communications, TruVista, TSC, Twin Valley, U-verse by DIRECTV, United Fiber, UScellular, 
# USI, Valley Telephone Cooperative, Verizon, Vexus, Visible, Vivint, Vonage, VTel, Vyve Broadband, Waitsfield & Champlain Valley Telecom, 
# WAVE Rural Connect, WeLink, West River Telecom, West Texas Rural Telephone Cooperative, Whip City Fiber, WinDBreak Cable, Windstream, 
# Winnebago Cooperative Telecom, Woodstock Communications, WOW!, WTC, Wyoming.com, Wyyerd Fiber, YoCo Fiber, Your Competition, Your Competition 2, 
# YouTube TV, Zentro, Ziply Fiber, Zito Media, ZoomOnline

# Once again, when you respond, return the following:
#     - GraphQL code or codes (directly executable).
#     - Brief, clear steps for queries when multiple queries are required.
#     - Any notes or instructions that help clarify the process of executing these queries.
#     - The fetchMyCurrentPackages queries should not be trusted, and if possible do not use this query, as there are known issues with it. There are several alternative queries that can be used to get the same information.
#     - Be flexible and adaptable in your approach, as the user may ask for different types of queries or mutations. Always ensure that the code you provide is correct, executable and do not shy away from using multiple queries if necessary.
#     - Several parameters (especially in the fetchLocationDetails query) are although optional require that you pass two parameters to get a result. For example, if you pass the city, you must also pass the state, as they go together. If you pass the state, you must also pass the city. If you pass the zip code, you do not need to pass the city or state.
#     - make sure that the GraphQL codes are well formatted as a code block, and that the code is syntactically correct.
#     
# Your task is to ensure that the code is correct, clear, and directly executable for use with the API, following all necessary steps when multiple queries are needed.

# Now, let’s begin!
# """

# introspection_agent_prompt_template = ChatPromptTemplate.from_messages([
#     ("system", INTROSPECTION_SYSTEM_MESSAGE_CONTENT),
#     MessagesPlaceholder(variable_name="messages") # Standard way for agent history
# ])

# async def create_introspection_agent_runnable(checkpointer): # Renamed to avoid clash if old name was used elsewhere
#     # Note: create_react_agent's `prompt` argument is typically for the main agent interaction,
#     # including placeholders for 'messages' (history) and 'agent_scratchpad'.
#     # The MessagesPlaceholder in introspection_agent_prompt_template handles history.
#     _introspection_agent_runnable = create_react_agent(
#         model=llm, # Assuming the primary LLM is suitable here
#         tools=[graphql_schema_tool_2], # Only schema tool for pure introspection
#         name="IntrospectionAgent",
#         prompt=introspection_agent_prompt_template,
#         checkpointer=checkpointer,
#     )
#     return _introspection_agent_runnable

# async def introspection_agent_graph_factory(): # Renamed for clarity
#     global _compiled_introspection_agent
#     if _compiled_introspection_agent is None:
#         saver = await get_saver()
#         # This creates the runnable agent, which is then compiled implicitly by LangGraph if used in add_node
#         # Or, if create_react_agent itself returns a compiled graph/runnable that doesn't need further .compile():
#         _compiled_introspection_agent = await create_introspection_agent_runnable(saver)
#     return _compiled_introspection_agent




# ───────────────────────────── Graph‑factory functions ──────────────────────
async def dynamic_swarm():
    global _compiled_telogical_swarm
    if _compiled_telogical_swarm is None:
        saver = await get_saver()
        _compiled_telogical_swarm = await create_swarm_workflow(saver)
    return _compiled_telogical_swarm

async def dynamic_swarm_refined():
    global _compiled_telogical_swarm_refined
    if _compiled_telogical_swarm_refined is None:
        # print("Compiling telogical_swarm_refined...") # Optional: for debugging
        saver = await get_saver()
        _compiled_telogical_swarm_refined = await create_refined_agent_workflow(saver)
        # print("telogical_swarm_refined compiled.") # Optional: for debugging
    return _compiled_telogical_swarm_refined

# Example of how you might run the refined swarm (add appropriate inputs)
# async def main():
#     checkpointer = await get_saver()
#     app = await dynamic_swarm_refined() # Gets the compiled graph

#     config = {"configurable": {"thread_id": "user-123"}}
#     initial_input = {
#         "messages": [HumanMessage(content="Hello, what are the internet options in Boston?")],
#         # app_output, refined_output, agent_tool_outputs, graphql_schema will be populated by the graph
#     }
    
#     async for event in app.astream_events(initial_input, config=config, version="v2"):
#         kind = event["event"]
#         if kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 print(content, end="")
#         elif kind == "on_tool_end":
#             print(f"-- Tool Output ({event['name']}) --\n{event['data'].get('output')}\n--")
#         elif kind in ["on_chain_end", "on_chat_model_end", "on_llm_end"]:
#             pass # Avoid too much noise
#         else:
#             # print(f"Event: {kind}, Data: {event['data']}")
#             pass
#     final_state = await app.ainvoke(initial_input, config)
#     print("\n--- Final State ---")
#     print(final_state.get("refined_output"))


# if __name__ == "__main__":
#     asyncio.run(main())