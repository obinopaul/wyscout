"""This module provides example tools for for the LangChain platform.

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

class VideoInfoInput(BaseModel):
    """
    Input schema for the video tool. You must specify a match_id and set at least one action to True.
    """
    match_id: int = Field(..., description="The unique Wyscout ID of the match.")

    # --- "Safe" Informational Actions (Do NOT consume video minutes) ---
    check_available_qualities: bool = Field(
        False, 
        description="[Safe Action] Set to True to check which video qualities are available for the match."
    )
    check_period_offsets: bool = Field(
        False, 
        description="[Safe Action] Set to True to get the start/end second for each period (1H, 2H, etc.) of the match video."
    )

    # --- "Costly" Link Generation Action (CONSUMES video minutes) ---
    generate_video_links: bool = Field(
        False, 
        description="[COSTLY ACTION] Set to True to generate video links. WARNING: This will consume your video usage minutes."
    )

    # --- Parameters for Link Generation ---
    start_second: Optional[int] = Field(None, description="For link generation, the start second of a custom video clip.")
    end_second: Optional[int] = Field(None, description="For link generation, the end second of a custom video clip.")
    quality: Optional[Literal['lq', 'sd', 'hd', 'fullhd']] = Field(None, description="For link generation, request a specific video quality.")
    
    # --- General Parameter ---
    fetch_match_details: bool = Field(False, description="For 'check_period_offsets' or 'generate_video_links', set True to fetch the full match object.")

    @model_validator(mode='before')
    def check_at_least_one_action(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        actions = ['check_available_qualities', 'check_period_offsets', 'generate_video_links']
        if not any(values.get(action) for action in actions):
            raise ValueError(f"You must set at least one action flag to True: {', '.join(actions)}.")
        return values
        
    @model_validator(mode='before')
    def check_costly_action_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        costly_params = ['start_second', 'end_second', 'quality']
        if any(values.get(p) is not None for p in costly_params) and not values.get('generate_video_links'):
            raise ValueError(f"Parameters {costly_params} can only be used when 'generate_video_links' is set to True.")
        return values


class WyscoutVideoTool:
    """A tool to safely query video info and generate video links from the Wyscout API."""

    def __init__(self, auth_token: str = DEFAULT_AUTH_TOKEN, timeout: int = DEFAULT_TIMEOUT):
        self.auth_token = auth_token
        self.timeout = timeout
        if self.auth_token == "YOUR_WYSCOUT_API_TOKEN":
            print("Warning: Using a placeholder Wyscout API token.")

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    async def _get_video_info_async(self, **kwargs) -> Dict[str, Any]:
        try:
            input_data = VideoInfoInput(**kwargs)
        except ValueError as e:
            return {"error": "Invalid input", "details": str(e)}

        results = {}
        tasks = []
        keys = []
        
        async with aiohttp.ClientSession() as session:
            def add_task(sub_endpoint, params, key):
                full_endpoint = f"/videos/{input_data.match_id}{sub_endpoint}"
                tasks.append(self._make_request(session, full_endpoint, params))
                keys.append(key)

            if input_data.check_available_qualities:
                add_task("/qualities", None, "available_qualities")
            
            if input_data.check_period_offsets:
                params = {"fetch": 'match'} if input_data.fetch_match_details else None
                add_task("/offsets", params, "period_offsets")
            
            if input_data.generate_video_links:
                params = {
                    "start": input_data.start_second,
                    "end": input_data.end_second,
                    "quality": input_data.quality,
                    "fetch": 'match' if input_data.fetch_match_details else None
                }
                add_task("", params, "video_links")

            api_responses = await asyncio.gather(*tasks)
            for key, response in zip(keys, api_responses):
                results[key] = response

        return results

    def get_video_info(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for the async video info fetcher."""
        return asyncio.run(self._get_video_info_async(**kwargs))

# Create the LangChain StructuredTool instance
wyscout_video_tool = StructuredTool(
    name="wyscout_video_tool",
    description=(
        "A tool for video information and clips. Provides 'safe' methods to check available qualities and period offsets which DO NOT consume usage minutes. "
        "It also provides a 'costly' method to generate video links. "
        "WARNING: Set 'generate_video_links' to True only when you intend to consume video usage credits."
    ),
    func=WyscoutVideoTool().get_video_info,
    args_schema=VideoInfoInput,
)