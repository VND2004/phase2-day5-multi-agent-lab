"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self.search_client.search(state.request.query, state.request.max_sources)
        state.sources = sources
        source_lines = "\n".join(
            f"[{index}] {source.title}: {source.snippet}"
            for index, source in enumerate(sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt="Researcher agent. Extract evidence and cite source numbers.",
            user_prompt=(
                f"Query: {state.request.query}\nAudience: {state.request.audience}\n"
                f"Sources:\n{source_lines}\n"
                "Return concise research notes with source numbers."
            ),
        )
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("agent.researcher", {"source_count": len(sources)})
        return state
