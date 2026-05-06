# Lab Guide: Multi-Agent Research System

## Scenario

Build a research assistant that can receive a long question, gather relevant
information, analyze evidence, and write a final response. The lab compares:

1. Single-agent baseline: one agent handles the full task.
2. Multi-agent workflow: Supervisor coordinates Researcher, Analyst, and Writer.

## Important Rules

- Do not add an agent unless its responsibility is clear.
- Every agent must have a separate responsibility.
- Shared state must be explicit enough to debug handoffs.
- Every workflow step must add a trace/log event.
- Benchmark the system instead of judging output by intuition alone.

## Milestone 1: Baseline

Files:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

Implemented: the baseline calls the NVIDIA/OpenAI-compatible LLM client and uses
a deterministic local fallback when the live provider is unavailable.

## Milestone 2: Supervisor

Files:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

Implemented routing policy:

- Call Researcher when research notes are missing.
- Call Analyst when analysis notes are missing.
- Call Writer when final answer is missing.
- Stop when final answer exists.
- Enforce max iterations and route to writer/stop fallback.

## Milestone 3: Worker Agents

Files:

- `agents/researcher.py`
- `agents/analyst.py`
- `agents/writer.py`

Implemented workers:

- Researcher collects source documents and research notes.
- Analyst extracts claims, evidence, caveats, and failure modes.
- Writer synthesizes the final answer with citation instructions.

## Milestone 4: Trace And Benchmark

Files:

- `observability/tracing.py`
- `evaluation/benchmark.py`
- `evaluation/report.py`

Benchmark metrics:

| Metric | Measurement |
|---|---|
| Latency | Wall-clock time |
| Cost | Estimated from recorded token counts |
| Quality | Lightweight 0-10 rubric based on completed artifacts |
| Citation coverage | Cited source numbers divided by source count |
| Failure rate | Failed run count divided by total run count |

## Exit Ticket

1. Use multi-agent when tasks benefit from separation of retrieval, reasoning,
   writing, and inspection.
2. Avoid multi-agent when the query is simple enough that added latency and
   orchestration complexity do not improve quality or safety.
