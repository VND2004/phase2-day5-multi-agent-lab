"""LangGraph workflow skeleton."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.supervisor = SupervisorAgent(self.settings)
        self.workers = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }

    def build(self) -> object:
        """Create a graph description.

        The lab can add LangGraph later; this object documents the same conditional
        routing policy and keeps the runtime dependency optional for tests.
        """

        return {
            "nodes": ["supervisor", "researcher", "analyst", "writer"],
            "entrypoint": "supervisor",
            "terminal": "done",
            "conditional_routes": {
                "supervisor": ["researcher", "analyst", "writer", "done"],
                "researcher": "supervisor",
                "analyst": "supervisor",
                "writer": "supervisor",
            },
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        self.build()
        while state.iteration < self.settings.max_iterations:
            with trace_span("workflow.supervisor", {"iteration": state.iteration}) as span:
                state = self.supervisor.run(state)
            state.add_trace_event("span.workflow.supervisor", span)

            route = state.route_history[-1]
            if route == "done":
                return state

            worker = self.workers.get(route)
            if worker is None:
                state.errors.append(f"Unknown route: {route}")
                state.record_route("done")
                return state

            with trace_span(f"workflow.{route}", {"iteration": state.iteration}) as span:
                state = worker.run(state)
            state.add_trace_event(f"span.workflow.{route}", span)

        if not state.final_answer:
            state.errors.append("Workflow stopped before final answer; invoking writer fallback.")
            state = self.workers["writer"].run(state)
        state.record_route("done")
        return state
