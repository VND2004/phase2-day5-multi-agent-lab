"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        source_list = "\n".join(
            f"[{index}] {source.title} - {source.url or 'local source'}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt="Writer agent. Write a clear final answer with citations.",
            user_prompt=(
                f"Query: {state.request.query}\nAudience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or 'None'}\n\n"
                f"Analysis notes:\n{state.analysis_notes or 'None'}\n\n"
                f"Available sources:\n{source_list}\n\n"
                "Write the final response. Cite sources with bracketed numbers where useful."
            ),
        )
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("agent.writer", {"has_final_answer": bool(state.final_answer)})
        return state
