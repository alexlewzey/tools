import os
import warnings
from datetime import datetime

from google.adk.agents.llm_agent import Agent

warnings.filterwarnings("ignore", category=UserWarning, message=r"\[EXPERIMENTAL\]")
os.environ.setdefault("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", "true")


def get_size(breed) -> dict:
    size = {"spaniel": "small", "labrador": "large"}[breed]
    return {"size": size}


def get_color(breed) -> dict:
    color = {"spaniel": "liver and white", "labrador": "choc"}[breed]
    return {"color": color}


def get_time() -> dict:
    return {"time": datetime.now().isoformat()}


root_agent = Agent(
    model="gemini-flash-lite-latest",
    name="root_agent",
    description="Tell us the current time or breed attributes.",
    instruction="You are a helpful assistant.",
    tools=[get_time, get_size, get_color],
)
