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

class TeamInfoInput(BaseModel):
    """
    Input schema for the unified teams tool.
    You must provide a 'wyId' for a team and set at least one 'get_*' flag to True.
    """
    wyId: int = Field(..., description="The unique Wyscout ID of the team.")

    # --- Action Flags to Specify Desired Data ---
    get_details: bool = Field(False, description="[Action] Retrieve basic details for the team.")
    get_career: bool = Field(False, description="[Action] Retrieve the team's historical career across seasons.")
    get_fixtures: bool = Field(False, description="[Action] Retrieve the team's upcoming fixtures.")
    get_matches: bool = Field(False, description="[Action] Retrieve a list of the team's played matches.")
    get_squad: bool = Field(False, description="[Action] Retrieve the team's current player squad and coach.")
    get_transfers: bool = Field(False, description="[Action] Retrieve the team's transfer history.")

    # --- Optional Parameters for Filtering and Detailing ---
    season_id: Optional[int] = Field(
        None, 
        description="For 'get_matches' or 'get_squad', specify a season ID to filter the results."
    )
    from_date: Optional[str] = Field(
        None, 
        description="For 'get_fixtures' or 'get_transfers', the start date in YYYY-MM-DD format."
    )
    to_date: Optional[str] = Field(
        None, 
        description="For 'get_fixtures' or 'get_transfers', the end date in YYYY-MM-DD format."
    )
    
    # --- Contextual Fetch/Detail Parameters ---
    career_fetch: Optional[List[Literal['team']]] = Field(None, description="For 'get_career', fetches the full team object.")
    career_details: Optional[List[Literal['season', 'competition']]] = Field(None, description="For 'get_career', expands season and competition objects.")
    
    matches_fetch: Optional[List[Literal['team']]] = Field(None, description="For 'get_matches', fetches the full team object.")
    
    squad_fetch: Optional[List[Literal['team']]] = Field(None, description="For 'get_squad', fetches the full team object.")
    
    transfers_details: Optional[List[Literal['teams', 'player']]] = Field(None, description="For 'get_transfers', expands the player and other teams' objects.")

    @model_validator(mode='before')
    def check_at_least_one_action_is_true(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that at least one get_* action flag is set to True."""
        actions = [
            'get_details', 'get_career', 'get_fixtures', 'get_matches', 'get_squad', 'get_transfers'
        ]
        if not any(values.get(action) for action in actions):
            raise ValueError("You must set at least one 'get_*' flag to True to specify what data to retrieve.")
        return values

class WyscoutTeamTool:
    """A unified tool to get all team-related data from the Wyscout API."""

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

    async def _get_team_info_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronously fetches team information based on the provided arguments."""
        try:
            input_data = TeamInfoInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        results = {}
        tasks = []
        keys = []
        base_endpoint = f"/teams/{input_data.wyId}"

        async with aiohttp.ClientSession() as session:
            def add_task(sub_endpoint, params, key):
                full_endpoint = f"{base_endpoint}{sub_endpoint}"
                tasks.append(self._make_request(session, full_endpoint, params))
                keys.append(key)

            if input_data.get_details:
                add_task("", None, "details")
            if input_data.get_career:
                params = {"fetch": ",".join(input_data.career_fetch or []), "details": ",".join(input_data.career_details or [])}
                add_task("/career", params, "career")
            if input_data.get_fixtures:
                params = {"fromDate": input_data.from_date, "toDate": input_data.to_date}
                add_task("/fixtures", params, "fixtures")
            if input_data.get_matches:
                params = {"seasonId": input_data.season_id, "fetch": ",".join(input_data.matches_fetch or [])}
                add_task("/matches", params, "matches")
            if input_data.get_squad:
                params = {"seasonId": input_data.season_id, "fetch": ",".join(input_data.squad_fetch or [])}
                add_task("/squad", params, "squad")
            if input_data.get_transfers:
                params = {"fromDate": input_data.from_date, "toDate": input_data.to_date, "details": ",".join(input_data.transfers_details or [])}
                add_task("/transfers", params, "transfers")

            api_responses = await asyncio.gather(*tasks)

            for key, response in zip(keys, api_responses):
                results[key] = response
        
        return results

    def get_team_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async team info fetcher."""
        return asyncio.run(self._get_team_info_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_team_info = StructuredTool(
    name="wyscout_team_info",
    description="A comprehensive tool to retrieve all types of data for a specific soccer team, including details, career, fixtures, matches, squad, and transfers.",
    func=WyscoutTeamTool().get_team_info,
    args_schema=TeamInfoInput,
)