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

# Get absolute path to the data directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
data_dir = os.path.join(project_root, "data")

# Use absolute paths for the CSV files
ZIP_CODE_CSV_PATH = os.path.join(data_dir, os.getenv("ZIP_CODE_CSV_PATH", "geo-data.csv"))
DMA_CSV_PATH = os.path.join(data_dir, os.getenv("DMA_CSV_PATH", "DMAs.csv"))

# Debug message to verify path
# print(f"Looking for ZIP code file at: {ZIP_CODE_CSV_PATH}")
# print(f"Looking for DMA file at: {DMA_CSV_PATH}")
# --- Constants (Replace with your actual values or load from environment) ---
DEFAULT_GRAPHQL_ENDPOINT = os.getenv("TELOGICAL_GRAPHQL_ENDPOINT_2", "YOUR_GRAPHQL_ENNDPOINT_HERE")
DEFAULT_AUTH_TOKEN = os.getenv("TELOGICAL_AUTH_TOKEN_2", "YOUR_AUTH_TOKEN_HERE")
DEFAULT_LOCALE = os.getenv("TELOGICAL_LOCALE", "YOUR_LOCALE_HERE")
# Path to CSV files in the data folder
DEFAULT_TIMEOUT = 30  # seconds for each GraphQL request


# --- Telogical LLM ---
TELOGICAL_MODEL_ENDPOINT_GPT = os.getenv("TELOGICAL_MODEL_ENDPOINT_GPT")
TELOGICAL_API_KEY_GPT = os.getenv("TELOGICAL_API_KEY_GPT")
TELOGICAL_MODEL_DEPLOYMENT_GPT = os.getenv("TELOGICAL_MODEL_DEPLOYMENT_GPT")
TELOGICAL_MODEL_API_VERSION_GPT = os.getenv("TELOGICAL_MODEL_API_VERSION_GPT")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")



# ---------------------------------------------------------------------------
# Tool 2: GraphQL Query Tool (Parallel Execution)
# ---------------------------------------------------------------------------

# --- Constants and Configuration ---
# In a real-world scenario, these would be in a configuration file or environment variables
WYSCOUT_API_BASE_URL = os.getenv("WYSCOUT_API_BASE_URL", "https://apirest.wyscout.com/v2")
DEFAULT_AUTH_TOKEN = os.getenv("DEFAULT_AUTH_TOKEN", "YOUR_WYSCOUT_API_TOKEN")  # Replace with your actual token
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 30))


class PlayerInfoInput(BaseModel):
    """Input schema for the unified player information tool."""
    wyId: int = Field(..., description="The unique Wyscout ID of the player.")
    get_details: bool = Field(False, description="Set to True to retrieve the player's basic details.")
    get_career: bool = Field(False, description="Set to True to retrieve the player's career history.")
    get_contract_info: bool = Field(False, description="Set to True to retrieve the player's contract information.")
    get_fixtures: bool = Field(False, description="Set to True to retrieve the player's upcoming fixtures.")
    get_matches: bool = Field(False, description="Set to True to retrieve the player's recent matches.")
    get_transfers: bool = Field(False, description="Set to True to retrieve the player's transfer history.")
    details_relations: Optional[List[str]] = Field(None, description="For 'get_details', a comma-separated list of related objects to detail (e.g., 'currentTeam').")
    career_fetch: Optional[List[str]] = Field(None, description="For 'get_career', a comma-separated list of related objects to fetch (e.g., 'player').")
    career_details: Optional[List[str]] = Field(None, description="For 'get_career', a comma-separated list of related objects to detail (e.g., 'team,competition,season').")
    contract_fetch: Optional[List[str]] = Field(None, description="For 'get_contract_info', a comma-separated list of related objects to fetch (e.g., 'player').")
    fixtures_from_date: Optional[str] = Field(None, description="For 'get_fixtures', the start date in YYYY-MM-DD format.")
    fixtures_to_date: Optional[str] = Field(None, description="For 'get_fixtures', the end date in YYYY-MM-DD format.")
    matches_season_id: Optional[int] = Field(None, description="For 'get_matches', the ID of the season to retrieve matches from.")
    matches_fetch: Optional[List[str]] = Field(None, description="For 'get_matches', a comma-separated list of related objects to fetch (e.g., 'player').")
    transfers_fetch: Optional[List[str]] = Field(None, description="For 'get_transfers', a comma-separated list of related objects to fetch (e.g., 'player').")
    transfers_details: Optional[List[str]] = Field(None, description="For 'get_transfers', a comma-separated list of related objects to detail (e.g., 'teams').")

class WyscoutPlayerTool:
    """
    A unified tool to interact with the Wyscout player API endpoints.
    """

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token. Please provide a valid token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper function to make a single API request."""
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        try:
            async with session.get(url, headers=headers, params=params, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"API Error: {response.status}", "details": await response.text()}
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": str(e)}

    async def _get_player_info_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronously fetches player information based on the provided arguments."""
        input_data = PlayerInfoInput(**kwargs)
        results = {}

        async with aiohttp.ClientSession() as session:
            tasks = []
            if input_data.get_details:
                params = {"details": ",".join(input_data.details_relations)} if input_data.details_relations else {}
                tasks.append(self._make_request(session, f"/players/{input_data.wyId}", params))
            if input_data.get_career:
                params = {
                    "fetch": ",".join(input_data.career_fetch) if input_data.career_fetch else "",
                    "details": ",".join(input_data.career_details) if input_data.career_details else ""
                }
                tasks.append(self._make_request(session, f"/players/{input_data.wyId}/career", params))
            if input_data.get_contract_info:
                params = {"fetch": ",".join(input_data.contract_fetch)} if input_data.contract_fetch else {}
                tasks.append(self._make_request(session, f"/players/{input_data.wyId}/contractinfo", params))
            if input_data.get_fixtures:
                params = {
                    "fromDate": input_data.fixtures_from_date,
                    "toDate": input_data.fixtures_to_date,
                }
                tasks.append(self._make_request(session, f"/players/{input_data.wyId}/fixtures", {k: v for k, v in params.items() if v is not None}))
            if input_data.get_matches:
                params = {
                    "seasonId": input_data.matches_season_id,
                    "fetch": ",".join(input_data.matches_fetch) if input_data.matches_fetch else ""
                }
                tasks.append(self._make_request(session, f"/players/{input_data.wyId}/matches", {k: v for k, v in params.items() if v is not None}))
            if input_data.get_transfers:
                params = {
                    "fetch": ",".join(input_data.transfers_fetch) if input_data.transfers_fetch else "",
                    "details": ",".join(input_data.transfers_details) if input_data.transfers_details else ""
                }
                tasks.append(self._make_request(session, f"/players/{input_data.wyId}/transfers", params))

            api_responses = await asyncio.gather(*tasks)

            response_keys = [
                "details", "career", "contract_info", "fixtures", "matches", "transfers"
            ]
            active_requests = [key for key, requested in zip(response_keys, [
                input_data.get_details, input_data.get_career, input_data.get_contract_info,
                input_data.get_fixtures, input_data.get_matches, input_data.get_transfers
            ]) if requested]

            for key, response in zip(active_requests, api_responses):
                results[key] = response

        return results

    def get_player_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async player info fetcher."""
        return asyncio.run(self._get_player_info_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_player_info = StructuredTool(
    name="wyscout_player_info",
    description="A unified tool to get comprehensive information about a soccer player from Wyscout. Select the types of information you need by setting the corresponding 'get_*' flags to True.",
    func=WyscoutPlayerTool().get_player_info,
    args_schema=PlayerInfoInput,
)