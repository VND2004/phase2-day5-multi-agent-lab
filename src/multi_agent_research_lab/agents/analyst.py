"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        notes = state.research_notes or "No research notes were produced."
        response = self.llm_client.complete(
            system_prompt="Analyst agent. Structure findings, tradeoffs, risks, and confidence.",
            user_prompt=(
                f"Query: {state.request.query}\n"
                f"Research notes:\n{notes}\n\n"
                "Produce: key claims, supporting evidence, caveats, and failure modes."
            ),
        )
        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("agent.analyst", {"has_research_notes": bool(state.research_notes)})
        return state
