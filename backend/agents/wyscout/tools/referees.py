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

class RefereeInfoInput(BaseModel):
    """Input schema for the referee details tool."""
    wyId: int = Field(..., description="The unique Wyscout ID of the referee.")
    include_image_data: bool = Field(
        False,
        description="Set to True to include the referee's photo as a base64 encoded string. Note: This will significantly increase the size of the response payload."
    )


class WyscoutRefereeTool:
    """A robust tool to retrieve details for a specific referee from the Wyscout API."""

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

    async def _get_referee_info_async(self, **kwargs) -> Dict[str, Any]:
        """Asynchronously fetches referee information."""
        try:
            input_data = RefereeInfoInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        endpoint = f"/referees/{input_data.wyId}"
        params = {}
        if input_data.include_image_data:
            params['imageDataURL'] = 'true'

        async with aiohttp.ClientSession() as session:
            result = await self._make_request(session, endpoint, params)
        
        return result

    def get_referee_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async referee info fetcher."""
        return asyncio.run(self._get_referee_info_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_referee_info = StructuredTool(
    name="wyscout_referee_info",
    description="Retrieves detailed information for a specific referee by their Wyscout ID. Can optionally include the referee's photo.",
    func=WyscoutRefereeTool().get_referee_info,
    args_schema=RefereeInfoInput,
)