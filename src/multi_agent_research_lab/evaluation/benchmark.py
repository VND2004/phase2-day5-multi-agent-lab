"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost estimate, citation coverage, quality, and failure rate."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost(state),
        quality_score=_score_quality(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if state.errors or not state.final_answer else 0.0,
        notes=_summarize_state(state),
    )
    return state, metrics


def _estimate_cost(state: ResearchState) -> float:
    total_tokens = 0
    for result in state.agent_results:
        total_tokens += int(result.metadata.get("input_tokens") or 0)
        total_tokens += int(result.metadata.get("output_tokens") or 0)
    return round(total_tokens * 0.0000002, 6)


def _score_quality(state: ResearchState) -> float:
    score = 2.0
    if state.sources:
        score += 2.0
    if state.research_notes:
        score += 1.5
    if state.analysis_notes:
        score += 1.5
    if state.final_answer:
        score += 2.0
    if _citation_coverage(state) > 0:
        score += 1.0
    if state.errors:
        score -= 2.0
    return max(0.0, min(10.0, round(score, 1)))


def _citation_coverage(state: ResearchState) -> float:
    if not state.sources or not state.final_answer:
        return 0.0
    citations = set(re.findall(r"\[(\d+)\]", state.final_answer))
    return round(min(1.0, len(citations) / len(state.sources)), 2)


def _summarize_state(state: ResearchState) -> str:
    if state.errors:
        return "; ".join(state.errors)
    return f"{len(state.sources)} sources; routes: {' > '.join(state.route_history)}"
