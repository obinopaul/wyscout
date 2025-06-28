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

class AreaInfoInput(BaseModel):
    """Input schema for the areas list tool. This tool takes no parameters."""
    pass


class WyscoutAreaTool:
    """
    A robust tool to retrieve a list of all areas from the Wyscout API,
    augmented with documented custom area information.
    """
    
    # Static data transcribed directly from the API documentation
    DOCUMENTED_CUSTOM_AREAS = [
        {"name": "Asia", "id": 1, "alpha2code": "ASX", "alpha3code": "AS"},
        {"name": "Africa", "id": 2, "alpha2code": "AFX", "alpha3code": "AF"},
        {"name": "N/C America", "id": 3, "alpha2code": "NCX", "alpha3code": "NC"},
        {"name": "South America", "id": 4, "alpha2code": "SAX", "alpha3code": "SA"},
        {"name": "Oceania", "id": 5, "alpha2code": "OCX", "alpha3code": "OC"},
        {"name": "Europe", "id": 6, "alpha2code": "EUX", "alpha3code": "EU"},
        {"name": "Chinese Taipei", "id": 49, "alpha2code": "CTX", "alpha3code": "CT"},
        {"name": "England", "id": 67, "alpha2code": "ENX", "alpha3code": "EN"}, # Note: Corrected based on common usage; original doc had a typo
        {"name": "Northern Ireland", "id": 144, "alpha2code": "NIX", "alpha3code": "NI"},
        {"name": "Scotland", "id": 164, "alpha2code": "SCX", "alpha3code": "SCT"},
        {"name": "Tahiti", "id": 187, "alpha2code": "TAX", "alpha3code": "TA"},
        {"name": "Wales", "id": 208, "alpha2code": "WAX", "alpha3code": "WA"},
        {"name": "Zanzibar", "id": 212, "alpha2code": "ZAX", "alpha3code": "ZA"},
        {"name": "Timor-Leste", "id": 213, "alpha2code": "LSX", "alpha3code": "LS"},
        {"name": "Kosovo", "id": 228, "alpha2code": "KSX", "alpha3code": "KS"},
        {"name": "France, metropolitan", "id": 298, "alpha2code": "FXX", "alpha3code": "FX"},
        {"name": "Netherlands antilles", "id": 305, "alpha2code": "ANX", "alpha3code": "AN"},
        {"name": "World", "id": 320, "alpha2code": "WOX", "alpha3code": "WO"}
    ]

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str) -> List[Dict[str, Any]]:
        """Helper function to make a single asynchronous API request."""
        headers = {"Authorization": self.auth_token}
        url = f"{WYSCOUT_API_BASE_URL}{endpoint}"
        try:
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            return [{"error": f"API Error: {e.status}", "message": e.message}]
        except Exception as e:
            return [{"error": "An unexpected error occurred", "details": str(e)}]

    async def _get_areas_async(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Asynchronously fetches the list of all areas and combines it with
        the static list of documented custom areas.
        """
        endpoint = "/areas"
        async with aiohttp.ClientSession() as session:
            live_areas = await self._make_request(session, endpoint)
        
        return {
            "live_api_areas": live_areas,
            "documented_custom_areas": self.DOCUMENTED_CUSTOM_AREAS
        }

    def get_areas(self, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """Synchronous wrapper for the async areas fetcher."""
        return asyncio.run(self._get_areas_async())

# Create the LangChain StructuredTool instance
wyscout_area_list = StructuredTool(
    name="wyscout_area_list",
    description=(
        "Retrieves a comprehensive list of geographic areas from Wyscout. "
        "Returns a dictionary with two keys: 'live_api_areas' for the full list of countries from the API, "
        "and 'documented_custom_areas' for a static list of special regions like continents, England, Scotland, etc. "
        "This tool requires no parameters."
    ),
    func=WyscoutAreaTool().get_areas,
    args_schema=AreaInfoInput,
)