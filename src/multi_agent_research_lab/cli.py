"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    state = _run_single_agent(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Markdown report path"),
    ] = Path("reports/benchmark_report.md"),
) -> None:
    """Run baseline and multi-agent benchmark, then write the markdown report."""

    _init()
    _, baseline_metrics = run_benchmark("single-agent baseline", query, _run_single_agent)
    _, multi_metrics = run_benchmark("multi-agent workflow", query, _run_multi_agent)
    report = render_markdown_report([baseline_metrics, multi_metrics])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(Panel.fit(str(output), title="Benchmark Report Written"))


def _run_single_agent(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    response = LLMClient().complete(
        system_prompt=(
            "Single-agent research assistant. "
            "Answer directly with brief citations if known."
        ),
        user_prompt=(
            f"Query: {query}\n"
            "Produce a concise research answer and mention limitations when "
            "sources are unavailable."
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
    state.record_route("single-agent")
    state.add_trace_event("baseline.single_agent", {"model": get_settings().llm_model})
    return state


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


if __name__ == "__main__":
    app()
