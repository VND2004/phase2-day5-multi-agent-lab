"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        "This report compares a single-agent baseline with the multi-agent workflow.",
        "",
        (
            "| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | "
            "Failure Rate | Notes |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | "
            f"{coverage} | {failure} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Failure Mode",
            "",
            (
                "Primary risk: a live provider, search API, or network call can fail. "
                "The implementation contains deterministic fallbacks, records errors "
                "in shared state, and stops through a "
                "max-iteration guardrail instead of looping indefinitely."
            ),
            "",
            "## Trace",
            "",
            "Each run records route decisions and agent spans in `ResearchState.trace`.",
        ]
    )
    return "\n".join(lines) + "\n"
