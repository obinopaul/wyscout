import os
from functools import cache
from typing import TypeAlias, Optional, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from backend.core.settings import settings
from backend.schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    VertexAIModelName,
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


ModelT: TypeAlias = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | FakeToolModel
)

# Cache of Telogical-specific models (used by dynamic agents)
_telogical_primary_llm: Optional[AzureChatOpenAI] = None
_telogical_secondary_llm: Optional[ChatOpenAI] = None


def get_telogical_primary_llm() -> AzureChatOpenAI:
    """
    Get the primary Telogical LLM (Azure OpenAI)
    """
    global _telogical_primary_llm
    if _telogical_primary_llm is None:
        # Get Telogical model configuration from environment variables
        endpoint = os.getenv("TELOGICAL_MODEL_ENDPOINT_GPT")
        api_key = os.getenv("TELOGICAL_API_KEY_GPT")
        deployment = os.getenv("TELOGICAL_MODEL_DEPLOYMENT_GPT")
        api_version = os.getenv("TELOGICAL_MODEL_API_VERSION_GPT")
        
        if not all([endpoint, api_key, deployment, api_version]):
            raise ValueError("Missing required Telogical model configuration")
            
        _telogical_primary_llm = AzureChatOpenAI(
            azure_deployment=deployment,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    
    return _telogical_primary_llm


def get_telogical_secondary_llm() -> ChatOpenAI:
    """
    Get the secondary Telogical LLM (OpenAI)
    """
    global _telogical_secondary_llm
    if _telogical_secondary_llm is None:
        # Using standard OpenAI model
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("TELOGICAL_SECONDARY_MODEL", "gpt-4.1-nano-2025-04-14")
        
        if not api_key:
            raise ValueError("Missing required OpenAI API key for secondary LLM")
            
        _telogical_secondary_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key
        )
    
    return _telogical_secondary_llm


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")
