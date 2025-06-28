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

# --- Pydantic Models for Different Stat Contexts ---
class MatchStatsContext(BaseModel):
    """Context for retrieving stats about a single match."""
    match_id: int = Field(..., description="The Wyscout ID of the match.")
    get_team_level_stats: bool = Field(False, description="Set to True to get the team-level advanced stats for this match.")
    get_all_players_stats: bool = Field(False, description="Set to True to get a list of advanced stats for every player in this match.")
    use_sides_for_team_stats: bool = Field(False, description="For team-level stats, set True to label teams as 'home' and 'away'.")

class PlayerStatsContext(BaseModel):
    """Context for retrieving stats for a single player."""
    player_id: int = Field(..., description="The Wyscout ID of the player.")
    competition_id: Optional[int] = Field(None, description="The competition ID. Required for season-long stats.")
    season_id: Optional[int] = Field(None, description="The season ID. If omitted for season-long stats, the current season is used.")
    match_id: Optional[int] = Field(None, description="Provide a match ID to get this player's stats for only that single match.")

    @model_validator(mode='before')
    def check_context_requirements(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('match_id') is None and values.get('competition_id') is None:
            raise ValueError("For player stats, you must provide either a 'match_id' (for single-match stats) or a 'competition_id' (for season-long stats).")
        return values

class TeamStatsContext(BaseModel):
    """Context for retrieving stats for a single team."""
    team_id: int = Field(..., description="The Wyscout ID of the team.")
    competition_id: Optional[int] = Field(None, description="The competition ID. Required for season-long stats.")
    season_id: Optional[int] = Field(None, description="The season ID. If omitted for season-long stats, the current season is used.")
    match_id: Optional[int] = Field(None, description="Provide a match ID to get this team's stats for only that single match.")

    @model_validator(mode='before')
    def check_context_requirements(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('match_id') is None and values.get('competition_id') is None:
            raise ValueError("For team stats, you must provide either a 'match_id' (for single-match stats) or a 'competition_id' (for season-long stats).")
        return values

class AdvancedStatsInput(BaseModel):
    """
    Input schema for the unified advanced stats tool.
    You must specify EXACTLY ONE of the following contexts: 'match_context', 'player_context', or 'team_context'.
    """
    match_context: Optional[MatchStatsContext] = None
    player_context: Optional[PlayerStatsContext] = None
    team_context: Optional[TeamStatsContext] = None

    # --- Generic fetch/details parameters to be applied where supported ---
    fetch: Optional[List[str]] = Field(None, description="List of related objects to fetch (e.g., 'competition', 'season', 'player').")
    details: Optional[List[str]] = Field(None, description="List of related objects to detail (e.g., 'teams', 'match', 'player').")

    @model_validator(mode='before')
    def check_exactly_one_context(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        contexts = ['match_context', 'player_context', 'team_context']
        provided_contexts = [c for c in contexts if values.get(c) is not None]
        if len(provided_contexts) != 1:
            raise ValueError(f"You must provide exactly one of the following contexts: {', '.join(contexts)}")
        return values

class WyscoutAdvancedStatsTool:
    """A unified tool to get all advanced statistics from the Wyscout API."""

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # (Standard _make_request helper function)
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        if 'details' in clean_params and isinstance(clean_params['details'], list):
            clean_params['details'] = ','.join(clean_params['details'])
        if 'fetch' in clean_params and isinstance(clean_params['fetch'], list):
            clean_params['fetch'] = ','.join(clean_params['fetch'])
        try:
            async with session.get(url, headers=headers, params=clean_params, timeout=self.timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            return {"error": f"API Error: {e.status}", "message": e.message}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    async def _get_advanced_stats_async(self, **kwargs) -> Dict[str, Any]:
        try:
            input_data = AdvancedStatsInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        results = {}
        tasks = []
        keys = []
        
        async with aiohttp.ClientSession() as session:
            # --- CONTEXT 1: Match Stats ---
            if input_data.match_context:
                ctx = input_data.match_context
                base_endpoint = f"/matches/{ctx.match_id}/advancedstats"
                if ctx.get_team_level_stats:
                    params = {"details": input_data.details, "useSides": 'true' if ctx.use_sides_for_team_stats else 'false'}
                    tasks.append(self._make_request(session, base_endpoint, params))
                    keys.append("match_team_stats")
                if ctx.get_all_players_stats:
                    params = {"details": input_data.details, "fetch": input_data.fetch}
                    tasks.append(self._make_request(session, f"{base_endpoint}/players", params))
                    keys.append("match_all_players_stats")

            # --- CONTEXT 2: Player Stats ---
            elif input_data.player_context:
                ctx = input_data.player_context
                if ctx.match_id: # Single-match stats
                    endpoint = f"/players/{ctx.player_id}/matches/{ctx.match_id}/advancedstats"
                    params = {"details": input_data.details, "fetch": input_data.fetch}
                    tasks.append(self._make_request(session, endpoint, params))
                    keys.append("player_single_match_stats")
                else: # Season-long stats
                    endpoint = f"/players/{ctx.player_id}/advancedstats"
                    params = {"compId": ctx.competition_id, "seasonId": ctx.season_id, "details": input_data.details, "fetch": input_data.fetch}
                    tasks.append(self._make_request(session, endpoint, params))
                    keys.append("player_season_stats")

            # --- CONTEXT 3: Team Stats ---
            elif input_data.team_context:
                ctx = input_data.team_context
                if ctx.match_id: # Single-match stats
                    endpoint = f"/teams/{ctx.team_id}/matches/{ctx.match_id}/advancedstats"
                    params = {"details": input_data.details, "fetch": input_data.fetch}
                    tasks.append(self._make_request(session, endpoint, params))
                    keys.append("team_single_match_stats")
                else: # Season-long stats
                    endpoint = f"/teams/{ctx.team_id}/advancedstats"
                    params = {"compId": ctx.competition_id, "seasonId": ctx.season_id, "details": input_data.details, "fetch": input_data.fetch}
                    tasks.append(self._make_request(session, endpoint, params))
                    keys.append("team_season_stats")

            api_responses = await asyncio.gather(*tasks)
            for key, response in zip(keys, api_responses):
                results[key] = response

        return results

    def get_advanced_stats(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async advanced stats fetcher."""
        return asyncio.run(self._get_advanced_stats_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_advanced_stats = StructuredTool(
    name="wyscout_advanced_stats",
    description="A comprehensive tool to retrieve advanced statistics. You must specify EXACTLY ONE context: 'match_context' (for stats about a single match), 'player_context' (for a player's performance), or 'team_context' (for a team's performance).",
    func=WyscoutAdvancedStatsTool().get_advanced_stats,
    args_schema=AdvancedStatsInput,
)