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

class IdSearchInput(BaseModel):
    """Input schema for the unified ID search tool."""
    search_term: str = Field(
        ..., 
        description="The name of the entity to search for (e.g., 'Lionel Messi', 'FC Barcelona', 'Collina')."
    )
    entity_type: Literal['player', 'team', 'competition', 'referee'] = Field(
        ...,
        description="The type of entity to search for. Must be one of 'player', 'team', 'competition', or 'referee'."
    )
    gender: Optional[Literal['men', 'women']] = Field(
        None,
        description="Optional: Filter search results by gender. Primarily used for 'player' or 'referee' searches."
    )
    limit: int = Field(5, description="The maximum number of potential matches to return.")


class WyscoutIdSearch:
    """
    A robust tool to search for the Wyscout ID (wyId) of a player, team, competition, or referee.
    It returns a list of potential matches to handle ambiguity, allowing the agent to choose the correct entity.
    """

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token.")

    def _parse_player_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # (Existing parser from before)
        parsed = []
        for res in results:
            team_info = res.get('currentTeam', {}) or {}
            passport_info = res.get('passportArea', {}) or {}
            parsed.append({
                "name": res.get('shortName'),
                "wyId": res.get('wyId'),
                "position": res.get('role', {}).get('name'),
                "current_team_name": team_info.get('name'),
                "nationality": passport_info.get('name')
            })
        return parsed

    def _parse_team_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # (Existing parser from before)
        parsed = []
        for res in results:
            area_info = res.get('area', {}) or {}
            parsed.append({
                "name": res.get('name'),
                "official_name": res.get('officialName'),
                "wyId": res.get('wyId'),
                "country": area_info.get('name'),
            })
        return parsed

    def _parse_competition_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # (Existing parser from before)
        parsed = []
        for res in results:
            area_info = res.get('area', {}) or {}
            parsed.append({
                "name": res.get('name'),
                "wyId": res.get('wyId'),
                "country": area_info.get('name'),
            })
        return parsed
        
    def _parse_referee_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[NEW] Extracts key contextual information for referee search results."""
        parsed = []
        for res in results:
            passport_info = res.get('birthArea', {}) or {}
            parsed.append({
                "name": res.get('shortName'),
                "wyId": res.get('wyId'),
                "nationality": passport_info.get('name')
            })
        return parsed

    async def _search_id_async(self, search_term: str, entity_type: str, gender: Optional[str], limit: int) -> Dict[str, Any]:
        """Asynchronously searches for an entity and returns a list of possibilities."""
        headers = {"Authorization": self.auth_token}
        params = {}
        endpoint = ""
        
        # This tool now intelligently chooses the best API endpoint based on the entity type
        if entity_type == 'referee':
            # Use the generic /search endpoint for referees, as it's the documented method
            endpoint = "/search"
            params = {"objType": "referee", "query": search_term, "gender": gender}
        else:
            # Use the more specific endpoints for other types, which may provide richer data
            endpoint = f"/{entity_type}s"
            params = {"search": search_term}

        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        clean_params = {k: v for k, v in params.items() if v is not None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=clean_params, timeout=self.timeout) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    results = data if isinstance(data, list) else data.get(f'{entity_type}s', [])

                    if not results:
                        return {"message": f"No {entity_type} found matching '{search_term}'."}

                    # Apply the appropriate parser based on entity type
                    parser_map = {
                        'player': self._parse_player_results,
                        'team': self._parse_team_results,
                        'competition': self._parse_competition_results,
                        'referee': self._parse_referee_results # NEW
                    }
                    parsed_results = parser_map[entity_type](results)
                    return {"potential_matches": parsed_results[:limit]}

        except aiohttp.ClientResponseError as e:
            return {"error": f"API Error: {e.status}", "message": e.message}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    def find_id(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async ID searcher."""
        return asyncio.run(self._search_id_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_id_search = StructuredTool(
    name="wyscout_id_search",
    description="The definitive tool to search for the Wyscout ID (wyId) of a player, team, competition, or referee. Can be filtered by gender. Returns a list of potential matches with contextual details to help resolve ambiguities.",
    func=WyscoutIdSearch().find_id,
    args_schema=IdSearchInput,
)