"""Streamlit UI for the multi-agent research lab."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import streamlit as st

from multi_agent_research_lab.cli import _run_single_agent
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

DEFAULT_QUERY = "Research GraphRAG state-of-the-art and write a 500-word summary"
REPORT_PATH = Path("reports/benchmark_report.md")


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent Research Lab",
        page_icon="M",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_style()

    st.markdown(
        """
        <div class="lab-hero">
          <div>
            <div class="lab-kicker">NVIDIA-first research workflow</div>
            <h1>Multi-Agent Research Lab</h1>
            <p>
              Run baseline, multi-agent orchestration, and benchmark evaluation from
              one focused workspace.
            </p>
          </div>
          <div class="lab-hero-stat">
            <span>Workflow</span>
            <strong>Supervisor -> Researcher -> Analyst -> Writer</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("System")
        st.write("Provider priority: NVIDIA, then OpenAI-compatible fallback.")
        st.code("reports/benchmark_report.md", language="text")
        st.divider()
        st.caption("Use the main workspace to configure and run experiments.")

    with st.container(border=True):
        st.subheader("Experiment Setup")
        query = st.text_area("Research query", value=DEFAULT_QUERY, height=120)
        controls = st.columns([1.1, 1, 1, 0.8])
        with controls[0]:
            mode = st.segmented_control(
                "Mode",
                ["Multi-agent", "Baseline", "Benchmark"],
                default="Multi-agent",
            )
        with controls[1]:
            audience = st.selectbox(
                "Audience",
                ["technical learners", "business stakeholders", "engineering reviewers"],
            )
        with controls[2]:
            max_sources = st.slider("Max sources", min_value=1, max_value=10, value=5)
        with controls[3]:
            st.write("")
            st.write("")
            run_clicked = st.button("Run experiment", type="primary", use_container_width=True)

    if not run_clicked:
        _render_empty_state()
        return

    if len(query.strip()) < 5:
        st.error("Please enter a query with at least 5 characters.")
        return

    request = ResearchQuery(query=query.strip(), max_sources=max_sources, audience=audience)

    if mode == "Baseline":
        with st.spinner("Running single-agent baseline..."):
            state, metrics = _run_with_metrics("single-agent baseline", request, _baseline_runner)
        _render_run(state, metrics)
    elif mode == "Benchmark":
        with st.spinner("Running baseline and multi-agent benchmark..."):
            baseline_state, baseline_metrics = run_benchmark(
                "single-agent baseline",
                request.query,
                _baseline_runner_for_benchmark(request),
            )
            multi_state, multi_metrics = run_benchmark(
                "multi-agent workflow",
                request.query,
                _multi_runner_for_benchmark(request),
            )
            report = render_markdown_report([baseline_metrics, multi_metrics])
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(report, encoding="utf-8")
        _render_benchmark(baseline_state, multi_state, [baseline_metrics, multi_metrics], report)
    else:
        with st.spinner("Running multi-agent workflow..."):
            state, metrics = _run_with_metrics("multi-agent workflow", request, _multi_runner)
        _render_run(state, metrics)


def _run_with_metrics(
    run_name: str,
    request: ResearchQuery,
    runner: Callable[[ResearchQuery], ResearchState],
) -> tuple[ResearchState, BenchmarkMetrics]:
    started = perf_counter()
    state = runner(request)
    latency = perf_counter() - started
    metric_state, metrics = run_benchmark(run_name, request.query, lambda _: state)
    metrics.latency_seconds = latency
    return metric_state, metrics


def _baseline_runner(request: ResearchQuery) -> ResearchState:
    state = _run_single_agent(request.query)
    state.request.max_sources = request.max_sources
    state.request.audience = request.audience
    return state


def _multi_runner(request: ResearchQuery) -> ResearchState:
    state = ResearchState(request=request)
    return MultiAgentWorkflow().run(state)


def _baseline_runner_for_benchmark(request: ResearchQuery) -> Callable[[str], ResearchState]:
    return lambda _: _baseline_runner(request)


def _multi_runner_for_benchmark(request: ResearchQuery) -> Callable[[str], ResearchState]:
    return lambda _: _multi_runner(request)


def _render_empty_state() -> None:
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    left, middle, right = st.columns([1.2, 1, 1])
    with left, st.container(border=True):
        st.subheader("Multi-agent")
        st.write("Routes the query through research, analysis, and writing stages.")
    with middle, st.container(border=True):
        st.subheader("Baseline")
        st.write("Runs a single-agent answer path for quick comparison.")
    with right, st.container(border=True):
        st.subheader("Benchmark")
        st.write("Compares baseline and multi-agent runs, then writes the markdown report.")


def _render_run(state: ResearchState, metrics: BenchmarkMetrics) -> None:
    st.markdown('<div class="section-label">Run Result</div>', unsafe_allow_html=True)
    _render_metrics([metrics])
    tabs = st.tabs(["Answer", "Sources", "Trace", "State JSON"])
    with tabs[0]:
        st.markdown(state.final_answer or "_No final answer produced._")
    with tabs[1]:
        _render_sources(state)
    with tabs[2]:
        _render_trace(state)
    with tabs[3]:
        st.json(state.model_dump(mode="json"))


def _render_benchmark(
    baseline_state: ResearchState,
    multi_state: ResearchState,
    metrics: list[BenchmarkMetrics],
    report: str,
) -> None:
    st.markdown('<div class="section-label">Benchmark Result</div>', unsafe_allow_html=True)
    _render_metrics(metrics)
    tabs = st.tabs(["Report", "Baseline", "Multi-agent", "Trace"])
    with tabs[0]:
        st.markdown(report)
        st.download_button(
            "Download report",
            data=report,
            file_name="benchmark_report.md",
            mime="text/markdown",
        )
    with tabs[1]:
        st.markdown(baseline_state.final_answer or "_No final answer produced._")
    with tabs[2]:
        st.markdown(multi_state.final_answer or "_No final answer produced._")
        _render_sources(multi_state)
    with tabs[3]:
        st.subheader("Baseline trace")
        _render_trace(baseline_state)
        st.subheader("Multi-agent trace")
        _render_trace(multi_state)


def _render_metrics(metrics: list[BenchmarkMetrics]) -> None:
    for item in metrics:
        st.markdown(f'<div class="metric-title">{item.run_name}</div>', unsafe_allow_html=True)
        columns = st.columns(4)
        columns[0].metric("Latency", f"{item.latency_seconds:.2f}s")
        columns[1].metric("Quality", _format_optional(item.quality_score, "{:.1f}/10"))
        columns[2].metric(
            "Citation coverage",
            _format_optional(item.citation_coverage, "{:.0%}"),
        )
        columns[3].metric("Failure rate", _format_optional(item.failure_rate, "{:.0%}"))


def _render_sources(state: ResearchState) -> None:
    if not state.sources:
        st.info("No sources recorded for this run.")
        return
    for index, source in enumerate(state.sources, start=1):
        with st.expander(f"[{index}] {source.title}", expanded=index == 1):
            st.write(source.snippet)
            if source.url:
                st.link_button("Open source", source.url)
            st.json(source.metadata)


def _render_trace(state: ResearchState) -> None:
    if not state.trace:
        st.info("No trace events recorded.")
        return
    for event in state.trace:
        st.code(f"{event['name']}\n{event['payload']}", language="text")


def _format_optional(value: float | None, template: str) -> str:
    return "N/A" if value is None else template.format(value)


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --lab-bg: #f4f9ff;
          --lab-surface: #ffffff;
          --lab-ink: #172033;
          --lab-ink-soft: #26364a;
          --lab-muted: #607089;
          --lab-line: #d7e4f2;
          --lab-accent: #0f86ff;
          --lab-accent-2: #16b89f;
          --lab-accent-3: #f59e0b;
          --lab-accent-4: #ef5da8;
          --lab-lavender: #f1edff;
          --lab-mint: #e8fff7;
          --lab-sun: #fff7df;
          --lab-rose: #fff0f7;
          --lab-accent-soft: #e7f4ff;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
          background: #f8fbff !important;
          color: var(--lab-ink) !important;
        }
        .stApp {
          background:
            linear-gradient(180deg, #eaf6ff 0%, #f8fbff 320px, #f4f9ff 100%);
          color: var(--lab-ink);
        }
        .stApp, .stApp * {
          color: var(--lab-ink);
        }
        .block-container {
          padding-top: 2rem;
          max-width: 1220px;
        }
        h1, h2, h3 {
          letter-spacing: 0;
          color: var(--lab-ink) !important;
        }
        p, label, span, div {
          color: inherit;
        }
        .lab-hero {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 24px;
          padding: 30px 34px;
          margin-bottom: 22px;
          border: 1px solid rgba(126, 170, 220, 0.55);
          border-radius: 8px;
          background:
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(232,247,255,0.96)),
            radial-gradient(circle at top right, rgba(22,184,159,0.26), transparent 40%),
            radial-gradient(circle at bottom left, rgba(239,93,168,0.18), transparent 42%);
          box-shadow: 0 16px 40px rgba(32, 92, 148, 0.10);
        }
        .lab-hero h1 {
          margin: 6px 0 8px 0;
          font-size: 42px;
          line-height: 1.05;
        }
        .lab-hero p {
          max-width: 760px;
          margin: 0;
          color: var(--lab-muted);
          font-size: 17px;
        }
        .lab-kicker {
          color: var(--lab-accent) !important;
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .lab-hero-stat {
          min-width: 260px;
          padding: 16px 18px;
          border: 1px solid var(--lab-line);
          border-radius: 8px;
          background: linear-gradient(135deg, #ffffff, var(--lab-mint));
        }
        .lab-hero-stat span {
          display: block;
          color: var(--lab-muted) !important;
          font-size: 13px;
          margin-bottom: 6px;
        }
        .lab-hero-stat strong {
          color: var(--lab-ink) !important;
          font-size: 15px;
        }
        .section-label {
          margin: 26px 0 10px;
          color: var(--lab-accent) !important;
          font-size: 12px;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .metric-title {
          margin: 18px 0 8px;
          color: var(--lab-muted) !important;
          font-weight: 700;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #ffffff, #f0f8ff) !important;
          border-right: 1px solid var(--lab-line);
        }
        [data-testid="stSidebar"] * {
          color: var(--lab-ink) !important;
        }
        [data-testid="stMetric"] {
          background: linear-gradient(135deg, #ffffff, var(--lab-accent-soft));
          border: 1px solid var(--lab-line);
          border-radius: 8px;
          padding: 16px 18px;
          box-shadow: 0 10px 26px rgba(32, 92, 148, 0.07);
        }
        [data-testid="stMetric"]:nth-of-type(2n) {
          background: linear-gradient(135deg, #ffffff, var(--lab-mint));
        }
        [data-testid="stMetric"]:nth-of-type(3n) {
          background: linear-gradient(135deg, #ffffff, var(--lab-sun));
        }
        [data-testid="stMetric"]:nth-of-type(4n) {
          background: linear-gradient(135deg, #ffffff, var(--lab-rose));
        }
        [data-testid="stMetric"] * {
          color: var(--lab-ink) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--lab-line) !important;
          background: rgba(255,255,255,0.92) !important;
          box-shadow: 0 12px 30px rgba(32, 92, 148, 0.08);
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 4px;
          border-bottom: 1px solid var(--lab-line);
        }
        .stTabs [data-baseweb="tab"] {
          height: 42px;
          border-radius: 6px 6px 0 0;
          background: #ffffff;
          color: var(--lab-ink) !important;
        }
        .stTabs [aria-selected="true"] {
          background: var(--lab-accent-soft) !important;
          color: var(--lab-accent) !important;
        }
        .stButton > button {
          border-radius: 6px;
          font-weight: 600;
          border: 1px solid transparent;
          background: linear-gradient(135deg, var(--lab-accent), #46b4ff) !important;
          color: #ffffff !important;
        }
        .stButton > button * {
          color: #ffffff !important;
        }
        .stDownloadButton > button {
          border-radius: 6px;
          background: linear-gradient(135deg, var(--lab-accent-2), #52d6bd) !important;
          color: #ffffff !important;
        }
        .stDownloadButton > button * {
          color: #ffffff !important;
        }
        div[data-testid="stExpander"] {
          background: #ffffff !important;
          border: 1px solid var(--lab-line) !important;
          border-radius: 8px;
        }
        div[data-testid="stExpander"] * {
          color: var(--lab-ink) !important;
        }
        textarea, input, select, [data-baseweb="textarea"], [data-baseweb="input"],
        [data-baseweb="select"], [data-baseweb="slider"] {
          background: #ffffff !important;
          color: var(--lab-ink) !important;
          border-color: var(--lab-line) !important;
        }
        [data-baseweb="select"] *,
        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *,
        [role="listbox"] *,
        [role="option"] * {
          background: #ffffff !important;
          color: var(--lab-ink) !important;
        }
        [data-baseweb="slider"] div {
          color: var(--lab-ink) !important;
        }
        pre, code, [data-testid="stCodeBlock"] {
          background: #f6f9ff !important;
          color: var(--lab-ink-soft) !important;
          border-color: var(--lab-line) !important;
        }
        [data-testid="stAlert"] {
          background: #fff7df !important;
          color: var(--lab-ink) !important;
          border: 1px solid #f4d58d !important;
        }
        [data-testid="stJson"] {
          background: #ffffff !important;
          color: var(--lab-ink) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
