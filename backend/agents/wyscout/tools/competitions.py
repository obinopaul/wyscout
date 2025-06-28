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
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 30))

class CompetitionInfoInput(BaseModel):
    """
    Input schema for the unified competitions tool.
    Use 'areaId' to list all competitions in an area.
    Use 'wyId' and a 'get_*' flag to get details about a specific competition.
    """
    areaId: Optional[int] = Field(
        None,
        description="The numeric ID of an area to list all its competitions. Use this OR wyId."
    )
    wyId: Optional[int] = Field(
        None,
        description="The unique Wyscout ID of a specific competition. Use this OR areaId."
    )

    # --- Actions for a specific competition (requires wyId) ---
    get_details: bool = Field(
        False,
        description="Set to True to retrieve details for the competition specified by 'wyId'."
    )
    get_matches: bool = Field(
        False,
        description="Set to True to retrieve the list of matches for the competition specified by 'wyId'."
    )
    get_players: bool = Field(
        False,
        description="Set to True to retrieve the list of players for the competition specified by 'wyId'."
    )
    get_seasons: bool = Field(
        False,
        description="Set to True to retrieve the list of seasons for the competition specified by 'wyId'."
    )
    get_teams: bool = Field(
        False,
        description="Set to True to retrieve the list of teams for the competition specified by 'wyId'."
    )

    # --- Optional parameters for sub-lists ---
    fetch_competition_context: bool = Field(
        False,
        description="For 'get_matches', 'get_players', 'get_seasons', or 'get_teams', set to True to include the full competition object in the response."
    )
    
    # --- Parameters for 'get_seasons' ---
    active_seasons_only: Optional[bool] = Field(
        None,
        description="For 'get_seasons', set to True to retrieve only active seasons."
    )

    # --- Parameters for 'get_players' ---
    limit: Optional[int] = Field(
        None,
        description="For 'get_players', limit the number of results (max 100 per page)."
    )
    page: Optional[int] = Field(
        None,
        description="For 'get_players', specify the page number for pagination."
    )
    search_query: Optional[str] = Field(
        None,
        description="For 'get_players', a string to search for players within the competition."
    )

    @model_validator(mode='before')
    def check_id_provided(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that exactly one of wyId or areaId is provided."""
        if 'wyId' in values and values.get('wyId') is not None and 'areaId' in values and values.get('areaId') is not None:
            raise ValueError("Provide either 'wyId' for a specific competition or 'areaId' to list competitions, but not both.")
        if ('wyId' not in values or values.get('wyId') is None) and ('areaId' not in values or values.get('areaId') is None):
            raise ValueError("You must provide either 'wyId' for a specific competition or 'areaId' to list competitions.")
        return values
        
    @model_validator(mode='before')
    def check_action_for_wyid(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that if wyId is provided, at least one get_* action is True."""
        if values.get('wyId') is not None:
            actions = ['get_details', 'get_matches', 'get_players', 'get_seasons', 'get_teams']
            if not any(values.get(action) for action in actions):
                raise ValueError("If you provide a 'wyId', you must set at least one 'get_*' flag (e.g., 'get_details') to True.")
        return values


class WyscoutCompetitionTool:
    """
    A unified and robust tool to interact with the Wyscout competitions API endpoints.
    """

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token. Please provide a valid token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str,
                            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper function to make a single asynchronous API request."""
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        # Filter out None values from params
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            async with session.get(url, headers=headers, params=clean_params, timeout=self.timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            return {"error": f"API Error: {e.status}", "message": e.message}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    async def _get_competition_info_async(self, **kwargs) -> Dict[str, Any]:
        """
        Asynchronously fetches competition information based on the provided arguments.
        """
        try:
            input_data = CompetitionInfoInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        results = {}
        tasks = []

        async with aiohttp.ClientSession() as session:
            if input_data.areaId is not None:
                # Mode 1: List all competitions for an area
                params = {"areaId": input_data.areaId}
                tasks.append(self._make_request(session, "/competitions", params))

            elif input_data.wyId is not None:
                # Mode 2: Get details for a specific competition
                base_endpoint = f"/competitions/{input_data.wyId}"
                fetch_param = {'fetch': 'competition'} if input_data.fetch_competition_context else {}

                if input_data.get_details:
                    tasks.append(self._make_request(session, base_endpoint))
                if input_data.get_matches:
                    tasks.append(self._make_request(session, f"{base_endpoint}/matches", fetch_param))
                if input_data.get_seasons:
                    season_params = fetch_param.copy()
                    if input_data.active_seasons_only is not None:
                        season_params['active'] = 'true' if input_data.active_seasons_only else 'false'
                    tasks.append(self._make_request(session, f"{base_endpoint}/seasons", season_params))
                if input_data.get_teams:
                    tasks.append(self._make_request(session, f"{base_endpoint}/teams", fetch_param))
                if input_data.get_players:
                    player_params = fetch_param.copy()
                    player_params.update({
                        "limit": input_data.limit,
                        "page": input_data.page,
                        "search": input_data.search_query,
                    })
                    tasks.append(self._make_request(session, f"{base_endpoint}/players", player_params))

            api_responses = await asyncio.gather(*tasks)

            # Assign responses to the correct keys
            if input_data.areaId is not None:
                results['competition_list'] = api_responses[0]
            else:
                response_index = 0
                if input_data.get_details:
                    results['details'] = api_responses[response_index]
                    response_index += 1
                if input_data.get_matches:
                    results['matches'] = api_responses[response_index]
                    response_index += 1
                if input_data.get_seasons:
                    results['seasons'] = api_responses[response_index]
                    response_index += 1
                if input_data.get_teams:
                    results['teams'] = api_responses[response_index]
                    response_index += 1
                if input_data.get_players:
                    results['players'] = api_responses[response_index]

        return results

    def get_competition_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async competition info fetcher."""
        return asyncio.run(self._get_competition_info_async(**kwargs))


# Create the LangChain StructuredTool instance
wyscout_competition_info = StructuredTool(
    name="wyscout_competition_info",
    description=(
        "A comprehensive tool to retrieve information about soccer competitions from Wyscout. "
        "Can list all competitions in a given area or fetch details, matches, players, seasons, or teams for a specific competition."
    ),
    func=WyscoutCompetitionTool().get_competition_info,
    args_schema=CompetitionInfoInput,
)
