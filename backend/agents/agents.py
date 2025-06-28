from dataclasses import dataclass
from typing import Callable, Awaitable, Any

from langgraph.pregel import Pregel

from langgraph.graph.state import CompiledStateGraph
from backend.agents.wyscout.agent import dynamic_swarm_refined
from backend.agents.research_assistant import research_assistant
from backend.schema import AgentInfo

DEFAULT_AGENT = "telogical-assistant"  # For testing purposes, use the telogical assistant


@dataclass
class Agent:
    description: str
    graph: Callable[[], Awaitable[CompiledStateGraph]]  # Now accepts an async factory function


agents: dict[str, Agent] = {
    "telogical-assistant": Agent(
        description="A Telogical assistant that can answer telecommunications market intelligence questions.",
        graph=dynamic_swarm_refined,
    ),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", 
        graph=research_assistant
    )
}


# async def get_agent(agent_id: str) -> Pregel:
#     """Get an agent's graph (with async initialization if needed)"""
#     agent = agents[agent_id]
#     return await agent.graph()


async def get_agent(agent_id: str) -> Pregel:
    """Get an agent's graph (with async initialization if needed)"""
    agent = agents[agent_id]

    # Check if agent.graph is already a CompiledStateGraph
    if isinstance(agent.graph, CompiledStateGraph):
        return agent.graph

    # Otherwise, it's a function that returns a CompiledStateGraph
    return await agent.graph()


def get_all_agent_info() -> list[AgentInfo]:
    """Get information about all available agents"""
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
