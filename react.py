"""
Reasoning engine that decides which tool to call
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
async def triple(num: float) -> float:
    """
    :param num: a number to triple
    :return: the number tripled ->  multiplied by 3
    """
    print("---EXECUTE TRIPLING TOOL---")
    await asyncio.sleep(0.1)  # Simulate async work
    return num * 3


tavily_tool = TavilySearch(max_results=1)
tools = [tavily_tool, triple]

# Get API key and ensure it's set
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in your .env file."
    )

# Allow model to be configured via environment variable, with fallback options
# Options: gemini-2.5-flash (20 req/day free), gemini-2.5-flash-lite (1000 req/day free), gemini-1.5-flash
model = os.getenv(
    "GEMINI_MODEL", "gemini-2.5-flash-lite"
)  # Default to lite for higher quota

llm = ChatGoogleGenerativeAI(
    model=model, temperature=0, google_api_key=api_key
).bind_tools(tools)
