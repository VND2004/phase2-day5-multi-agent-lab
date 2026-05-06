"""Optional critic agent skeleton for bonus work."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        final_answer = state.final_answer or ""
        citations = set(re.findall(r"\[(\d+)\]", final_answer))
        coverage = len(citations) / max(1, len(state.sources))
        finding = (
            f"Critic check: citation coverage is {coverage:.0%}; "
            f"errors recorded: {len(state.errors)}."
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=finding,
                metadata={"citation_coverage": coverage, "citation_count": len(citations)},
            )
        )
        state.add_trace_event(
            "agent.critic",
            {"citation_coverage": coverage, "citation_count": len(citations)},
        )
        return state
