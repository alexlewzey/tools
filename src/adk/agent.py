"""
Hi, I would like to complain that I need more customer support as a bug with my
software and it arrived later than it should have done.
"""

from google.adk import Agent, Event, Workflow

process_message = Agent(
    name="process_message",
    model="gemini-flash-lite-latest",
    instruction=(
        "Categorize the user message as either BUG, CUSTOMER_SUPPORT or LOGISTICS. "
        "If the message applies to more than one category, return them, comma "
        "separated."
    ),
    output_schema=str,
)


def router(node_input: str):
    routes = node_input.split(",")
    routes = [route.strip() for route in routes]
    return Event(route=routes)


def response_bug():
    return Event(message="BUG")


def response_support():
    return Event(output="SUPPORT")


def response_logistics():
    return Event(message="LOGISTICS")


def triage():
    return Event(message="Triaging the support")


root_agent = Workflow(
    name="routing_workflow",
    edges=[
        ("START", process_message, router),
        (
            router,
            {
                "BUG": response_bug,
                "CUSTOMER_SUPPORT": response_support,
                "LOGISTICS": response_logistics,
            },
        ),
        (response_support, triage),
    ],
)
