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


# --- Helper Models for Complex Filters ---
class GameWeekInterval(BaseModel):
    """Defines a start and end week for career standings filters."""
    startWeek: int
    endWeek: int

# --- Main Input Schema ---
class SeasonInfoInput(BaseModel):
    """
    Input schema for the unified seasons tool.
    You must provide a 'wyId' for a season and set at least one 'get_*' flag to True.
    """
    wyId: int = Field(..., description="The unique Wyscout ID of the season.")

    # --- Action Flags to Specify Desired Data ---
    get_details: bool = Field(False, description="[Action] Retrieve basic details for the season.")
    get_assistmen: bool = Field(False, description="[Action] Retrieve the list of top assist providers for the season.")
    get_career_stats: bool = Field(False, description="[Action] Retrieve team performance stats for the season.")
    get_fixtures: bool = Field(False, description="[Action] Retrieve all match fixtures for the season.")
    get_matches: bool = Field(False, description="[Action] Retrieve the list of played matches in the season.")
    get_players: bool = Field(False, description="[Action] Retrieve the list of all players in the season.")
    get_scorers: bool = Field(False, description="[Action] Retrieve the list of top goal scorers for the season.")
    get_standings: bool = Field(False, description="[Action] Retrieve the league or group standings for the season.")
    get_teams: bool = Field(False, description="[Action] Retrieve the list of all teams in the season.")
    get_transfers: bool = Field(False, description="[Action] Retrieve all player transfers during the season.")
    
    # --- Optional Parameters for 'get_details' ---
    detail_relations: Optional[List[Literal['competition']]] = Field(None, description="For 'get_details', expands the competition object.")

    # --- Optional Parameters for 'get_assistmen', 'get_scorers' ---
    leader_details: Optional[List[Literal['players', 'teams']]] = Field(None, description="For 'get_assistmen' or 'get_scorers', expands player and team objects.")
    leader_fetch: Optional[List[Literal['season', 'competition']]] = Field(None, description="For 'get_assistmen' or 'get_scorers', includes the full season/competition objects.")

    # --- Optional Parameters for 'get_career_stats' ---
    career_details: Optional[List[Literal['team', 'round']]] = Field(None, description="For 'get_career_stats', expands team and round objects.")
    career_gameweek: Optional[int] = Field(None, description="For 'get_career_stats', filters standings for a specific gameweek.")
    career_gameweek_interval: Optional[GameWeekInterval] = Field(None, description="For 'get_career_stats', filters standings for a range of gameweeks.")

    # --- Optional Parameters for 'get_fixtures', 'get_transfers' ---
    from_date: Optional[str] = Field(None, description="For 'get_fixtures' or 'get_transfers', the start date in YYYY-MM-DD format.")
    to_date: Optional[str] = Field(None, description="For 'get_fixtures' or 'get_transfers', the end date in YYYY-MM-DD format.")
    fixture_details: Optional[List[Literal['matches', 'players', 'teams']]] = Field(None, description="For 'get_fixtures', expands related objects.")

    # --- Optional Parameters for 'get_players' ---
    player_list_details: Optional[List[Literal['currentTeam']]] = Field(None, description="For 'get_players', expands the player's current team object.")
    limit: Optional[int] = Field(None, description="For 'get_players', limit the number of results (max 100).")
    page: Optional[int] = Field(None, description="For 'get_players', specify the page number for pagination.")
    
    # --- Optional Parameters for 'get_standings' ---
    standings_round_id: Optional[int] = Field(None, description="For 'get_standings', filters for a specific round ID (e.g., a playoff stage).")
    standings_details: Optional[List[Literal['teams']]] = Field(None, description="For 'get_standings', expands the team objects.")
    
    # --- Generic Fetch Parameter ---
    fetch_context: Optional[List[Literal['season', 'competition']]] = Field(None, description="A generic fetch parameter for endpoints that support it ('matches', 'players', 'teams', 'standings').")

    @model_validator(mode='before')
    def check_at_least_one_action_is_true(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that at least one get_* action flag is set to True."""
        actions = [
            'get_details', 'get_assistmen', 'get_career_stats', 'get_fixtures', 
            'get_matches', 'get_players', 'get_scorers', 'get_standings', 
            'get_teams', 'get_transfers'
        ]
        if not any(values.get(action) for action in actions):
            raise ValueError("You must set at least one 'get_*' flag to True to specify what data to retrieve.")
        return values

class WyscoutSeasonTool:
    """A unified tool to get all season-related data from the Wyscout API."""

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper function to make a single asynchronous API request."""
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            async with session.get(url, headers=headers, params=clean_params, timeout=self.timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            return {"error": f"API Error: {e.status}", "message": e.message}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    async def _get_season_info_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronously fetches season information based on the provided arguments."""
        try:
            input_data = SeasonInfoInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        results = {}
        tasks = []
        base_endpoint = f"/seasons/{input_data.wyId}"

        async with aiohttp.ClientSession() as session:
            def add_task(sub_endpoint, params, key):
                full_endpoint = f"{base_endpoint}{sub_endpoint}"
                tasks.append(self._make_request(session, full_endpoint, params))
                return key

            keys = []
            if input_data.get_details:
                keys.append(add_task("", {"details": ",".join(input_data.detail_relations or [])}, "details"))
            if input_data.get_assistmen:
                keys.append(add_task("/assistmen", {"details": ",".join(input_data.leader_details or []), "fetch": ",".join(input_data.leader_fetch or [])}, "assistmen"))
            if input_data.get_career_stats:
                params = {"details": ",".join(input_data.career_details or []), "gameWeek": input_data.career_gameweek, "gameWeekInterval": input_data.career_gameweek_interval.model_dump_json() if input_data.career_gameweek_interval else None}
                keys.append(add_task("/career", params, "career_stats"))
            if input_data.get_fixtures:
                params = {"details": ",".join(input_data.fixture_details or []), "fromDate": input_data.from_date, "toDate": input_data.to_date, "fetch": ",".join(input_data.fetch_context or [])}
                keys.append(add_task("/fixtures", params, "fixtures"))
            if input_data.get_matches:
                keys.append(add_task("/matches", {"fetch": ",".join(input_data.fetch_context or [])}, "matches"))
            if input_data.get_players:
                params = {"details": ",".join(input_data.player_list_details or []), "limit": input_data.limit, "page": input_data.page, "fetch": ",".join(input_data.fetch_context or [])}
                keys.append(add_task("/players", params, "players"))
            if input_data.get_scorers:
                 keys.append(add_task("/scorers", {"details": ",".join(input_data.leader_details or []), "fetch": ",".join(input_data.leader_fetch or [])}, "scorers"))
            if input_data.get_standings:
                params = {"details": ",".join(input_data.standings_details or []), "roundId": input_data.standings_round_id, "fetch": ",".join(input_data.fetch_context or [])}
                keys.append(add_task("/standings", params, "standings"))
            if input_data.get_teams:
                keys.append(add_task("/teams", {"fetch": ",".join(input_data.fetch_context or [])}, "teams"))
            if input_data.get_transfers:
                params = {"details": "teams,player", "fromDate": input_data.from_date, "toDate": input_data.to_date}
                keys.append(add_task("/transfers", params, "transfers"))

            api_responses = await asyncio.gather(*tasks)

            for key, response in zip(keys, api_responses):
                results[key] = response
        
        return results

    def get_season_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async season info fetcher."""
        return asyncio.run(self._get_season_info_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_season_info = StructuredTool(
    name="wyscout_season_info",
    description="A comprehensive tool to retrieve all types of data for a specific soccer season, including stats, standings, players, fixtures, and more.",
    func=WyscoutSeasonTool().get_season_info,
    args_schema=SeasonInfoInput,
)