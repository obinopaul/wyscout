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

WYSCOUT_API_BASE_URL = os.getenv("WYSCOUT_API_BASE_URL", "https://apirest.wyscout.com/v2")
DEFAULT_AUTH_TOKEN = os.getenv("DEFAULT_AUTH_TOKEN", "YOUR_WYSCOUT_API_TOKEN")  # Replace with your actual token
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 30))

class MatchInfoInput(BaseModel):
    """Input schema for the unified match information tool."""
    wyId: int = Field(..., description="The unique Wyscout ID of the match.")
    get_details: bool = Field(True, description="Set to True to retrieve the match's general details.")
    get_formations: bool = Field(False, description="Set to True to retrieve the match's formation information.")
    use_sides: Optional[bool] = Field(
        False, description="For 'get_details', flag to change teamId labels to 'home' or 'away'."
    )
    details_relations: Optional[List[Literal[
        'coaches', 
        'players', 
        'teams', 
        'competition', 
        'round', 
        'season'
    ]]] = Field(
        None,
        description="For 'get_details', a list of related objects to expand with full details in the response."
    )
    formations_fetch: Optional[List[Literal[
        'teams', 
        'players'
    ]]] = Field(
        None, 
        description="For 'get_formations', a list of related objects to fetch and include in the response."
    )


class WyscoutMatchTool:
    """
    A unified and robust tool to interact with the Wyscout match API endpoints,
    retrieving match details and/or formations.
    """

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        """
        Initializes the tool with authentication and configuration.
        """
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token. Please provide a valid token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str,
                            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper function to make a single asynchronous API request."""
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        try:
            async with session.get(url, headers=headers, params=params, timeout=self.timeout) as response:
                response.raise_for_status()  # Will raise an exception for 4xx/5xx status codes
                return await response.json()
        except aiohttp.ClientResponseError as e:
            return {"error": f"API Error: {e.status}", "message": e.message, "url": str(e.request_info.url)}
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": "An unexpected error occurred", "details": str(e)}

    async def _get_match_info_async(self, **kwargs) -> Dict[str, Any]:
        """
        Asynchronously fetches match information based on the provided arguments,
        making parallel calls if both details and formations are requested.
        """
        input_data = MatchInfoInput(**kwargs)
        results = {}
        tasks = []

        async with aiohttp.ClientSession() as session:
            # Prepare the request for match details
            if input_data.get_details:
                detail_params = {}
                if input_data.use_sides:
                    detail_params['useSides'] = 'true'
                if input_data.details_relations:
                    detail_params['details'] = ",".join(input_data.details_relations)
                tasks.append(
                    self._make_request(session, f"/matches/{input_data.wyId}", detail_params)
                )

            # Prepare the request for match formations
            if input_data.get_formations:
                formation_params = {}
                if input_data.formations_fetch:
                    formation_params['fetch'] = ",".join(input_data.formations_fetch)
                tasks.append(
                    self._make_request(session, f"/matches/{input_data.wyId}/formations", formation_params)
                )

            if not tasks:
                return {"warning": "No data requested. Set 'get_details' or 'get_formations' to True."}

            api_responses = await asyncio.gather(*tasks)

            # Assign responses to the correct keys in the final result dictionary
            response_index = 0
            if input_data.get_details:
                results['details'] = api_responses[response_index]
                response_index += 1
            if input_data.get_formations:
                results['formations'] = api_responses[response_index]

        return results

    def get_match_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async match info fetcher."""
        return asyncio.run(self._get_match_info_async(**kwargs))


# Create the LangChain StructuredTool instance
wyscout_match_info = StructuredTool(
    name="wyscout_match_info",
    description=(
        "Retrieves detailed information and/or team formations for a specific soccer match from Wyscout. "
        "Specify the match 'wyId' and set flags for which data to retrieve."
    ),
    func=WyscoutMatchTool().get_match_info,
    args_schema=MatchInfoInput,
)
