from datetime import datetime

from google.adk import Agent, Event, Workflow
from pydantic import BaseModel

city_generator_agent = Agent(
    name="city_generator_agent",
    model="gemini-flash-latest",
    instruction=(
        "Return the name of a random city, return the name only and nothing else."
    ),
    output_schema=str,
)


class CityTime(BaseModel):
    city: str
    time_info: str


def lookup_time_function(node_input: str):
    return CityTime(time_info=datetime.now().isoformat(), city=node_input)


city_report_agent = Agent(
    name="city_report_agent",
    model="gemini-flash-latest",
    instruction=(
        "Output the following line:\n"
        "It is {CityTime.time_info} in {CityTime.city} right now."
    ),
    output_schema=str,
)


def completed_message_function(node_input: str):
    return Event(message=f"{node_input}\n WORKFLOW COMPLETE.")


root_agent = Workflow(
    name="root_agent",
    edges=[
        (
            "START",
            city_generator_agent,
            lookup_time_function,
            city_report_agent,
            completed_message_function,
        )
    ],
)
