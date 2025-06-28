"""This module provides example tools for for the LangChain platform.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast, Dict, Literal, Union
from typing_extensions import Annotated
import json
from enum import Enum
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
from typing import List, Optional, Dict, Any
import traceback
from langchain.tools.base import StructuredTool
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import AzureChatOpenAI
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta
import asyncio
import aiohttp
import random 
import re
import pandas as pd 
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.tools import BaseTool, Tool
from langgraph_swarm import create_handoff_tool, create_swarm, add_active_agent_router
import requests
import logging
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

# --- Constants and Configuration ---
WYSCOUT_API_BASE_URL = os.getenv("WYSCOUT_API_BASE_URL", "https://apirest.wyscout.com/v2")
DEFAULT_AUTH_TOKEN = os.getenv("DEFAULT_AUTH_TOKEN", "YOUR_WYSCOUT_API_TOKEN")  # Replace with your actual token
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 60)) # Increased timeout for potentially large event payloads

# --- Literal types for Pydantic validation, transcribed from documentation ---
FETCH_RELATIONS_LITERAL = Literal[
    'teams', 'players', 'match', 'coaches', 'referees', 'formations', 'substitutions'
]

# Defines the possible values for the 'details' API query parameter.
# - 'tag': If included, it expands the tag information within the event data.
DETAIL_RELATIONS_LITERAL = Literal['tag']

# Defines objects that can be excluded from the API response to reduce payload size.
# - 'possessions': Excludes the detailed possession sequence object from events.
# - 'names': Excludes the full names of players, teams, etc., leaving only the IDs.
# - 'positions': Excludes the detailed position data.
# - 'formations': Excludes the formation data bundled with the context.
EXCLUDE_OBJECTS_LITERAL = Literal['possessions', 'names', 'positions', 'formations']

# Defines the distinct periods of a soccer match.
# - '1H': First Half
# - '2H': Second Half
# - '1E': First Period of Extra Time
# - '2E': Second Period of Extra Time
# - 'P': Penalty Shootout
# The tool uses this list for the 'filter_by_period' client-side filtering option.
PRIMARY_EVENT_TYPE_LITERAL = Literal[
    'acceleration', 'clearance', 'corner', 'duel', 'fairplay', 'free_kick', 
    'game_interruption', 'goal_kick', 'goalkeeper_exit', 'infraction', 
    'interception', 'offside', 'own_goal', 'pass', 'penalty', 'pressing_attempt', 
    'received_pass', 'shot', 'shot_against', 'throw_in', 'touch', 
    'postmatch_penalty', 'postmatch_penalty_faced'
]
MATCH_PERIOD_LITERAL = Literal['1H', '2H', '1E', '2E', 'P']


class MatchEventsInput(BaseModel):
    """
    Input schema for the powerful match events tool.
    Fetches the full event stream for a match and provides optional client-side filtering.
    """
    match_id: int = Field(..., description="The unique Wyscout ID of the match.")

    # --- API-level parameters to control the initial fetch ---
    fetch_relations: Optional[List[FETCH_RELATIONS_LITERAL]] = Field(None, description="List of related context objects to fetch alongside the events.")
    detail_relations: Optional[List[DETAIL_RELATIONS_LITERAL]] = Field(None, description="List of objects to expand with details within the events.")
    exclude_objects: Optional[List[EXCLUDE_OBJECTS_LITERAL]] = Field(None, description="List of objects to exclude from the API response to reduce payload size.")

    # --- Client-side filtering parameters to process the fetched data ---
    filter_by_primary_types: Optional[List[PRIMARY_EVENT_TYPE_LITERAL]] = Field(None, description="[Filter] Return only events with these primary types (e.g., ['shot', 'pass']).")
    filter_by_player_id: Optional[int] = Field(None, description="[Filter] Return only events performed by this player ID.")
    filter_by_team_id: Optional[int] = Field(None, description="[Filter] Return only events performed by this team ID.")
    filter_by_period: Optional[List[MATCH_PERIOD_LITERAL]] = Field(None, description="[Filter] Return only events from these match periods (e.g., ['1H', 'P']).")


class WyscoutMatchEventsTool:
    """A tool to retrieve and filter the event stream of a soccer match."""

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        clean_params = {}
        for k, v in (params or {}).items():
            if v is not None:
                clean_params[k] = ','.join(v) if isinstance(v, list) else v
        
        try:
            async with session.get(url, headers=headers, params=clean_params, timeout=self.timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            return [{"error": f"API Error: {e.status}", "message": e.message}]
        except Exception as e:
            return [{"error": "An unexpected error occurred", "details": str(e)}]

    async def _get_match_events_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronously fetches and then filters match events."""
        try:
            input_data = MatchEventsInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        endpoint = f"/matches/{input_data.match_id}/events"
        params = {
            "fetch": input_data.fetch_relations,
            "details": input_data.detail_relations,
            "exclude": input_data.exclude_objects
        }

        async with aiohttp.ClientSession() as session:
            # Step 1: Fetch the full event data from the API
            full_events_list = await self._make_request(session, endpoint, params)

        # Check if API call returned an error
        if isinstance(full_events_list, list) and len(full_events_list) > 0 and 'error' in full_events_list[0]:
            return full_events_list[0]

        # Step 2: Apply client-side filters if provided
        filtered_events = full_events_list
        if input_data.filter_by_period:
            filtered_events = [e for e in filtered_events if e.get('matchPeriod') in input_data.filter_by_period]
        if input_data.filter_by_team_id:
            filtered_events = [e for e in filtered_events if e.get('team', {}).get('id') == input_data.filter_by_team_id]
        if input_data.filter_by_player_id:
            filtered_events = [e for e in filtered_events if e.get('player', {}).get('id') == input_data.filter_by_player_id]
        if input_data.filter_by_primary_types:
            filtered_events = [e for e in filtered_events if e.get('type', {}).get('primary') in input_data.filter_by_primary_types]

        return {
            "match_id": input_data.match_id,
            "total_events_fetched": len(full_events_list),
            "total_events_returned": len(filtered_events),
            "events": filtered_events
        }

    def get_match_events(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async match events fetcher."""
        return asyncio.run(self._get_match_events_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_match_events = StructuredTool(
    name="wyscout_match_events",
    description=(
        "Retrieves the full event stream for a given match and provides powerful client-side filtering. "
        "Useful for analyzing specific situations, like all shots by a player or all duels in the second half."
    ),
    func=WyscoutMatchEventsTool().get_match_events,
    args_schema=MatchEventsInput,
)