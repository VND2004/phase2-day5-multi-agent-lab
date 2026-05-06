# Lab 20: Multi-Agent Research System

Repo nay trien khai research assistant gom **Supervisor + Researcher + Analyst + Writer**
va benchmark voi **single-agent baseline**.

## Architecture

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> sources + research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Tao `.env` voi NVIDIA OpenAI-compatible API:

```bash
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1/
AGENT_MODEL=openai/gpt-oss-20b
```

`OPENAI_API_KEY`/`OPENAI_MODEL` van duoc ho tro nhu fallback tuong thich, nhung
neu co `NVIDIA_API_KEY` thi he thong uu tien NVIDIA qua OpenAI-compatible SDK.

## Run

```bash
pytest
ruff check src tests
mypy src
```

Baseline:

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Multi-agent workflow:

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Benchmark report:

```bash
python -m multi_agent_research_lab.cli benchmark \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary" \
  --output reports/benchmark_report.md
```

## Implemented Requirements

1. NVIDIA-first LLM client using `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, and `AGENT_MODEL`.
2. Deterministic fallback when provider/network/optional SDK is unavailable.
3. Local mock search client returning structured `SourceDocument` records.
4. Supervisor routing policy with `max_iterations` guardrail.
5. Researcher, Analyst, Writer, and optional Critic agent implementations.
6. Workflow orchestration with trace events for routing and agent spans.
7. Benchmark metrics for latency, estimated cost, quality, citation coverage, and failure rate.
8. Markdown report rendered to `reports/benchmark_report.md`.

## Deliverables

- `reports/benchmark_report.md`: comparison of single-agent and multi-agent runs.
- `ResearchState.trace`: per-run route decisions and agent spans.
- Failure-mode explanation: included in the benchmark report.
