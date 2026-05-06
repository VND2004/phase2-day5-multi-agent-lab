# Design Template

## Problem

Build a research assistant that accepts a long-form query, gathers source snippets,
analyzes the evidence, writes a final answer, and records trace/benchmark metadata.

## Why multi-agent?

A single-agent baseline is simpler and faster for small questions, but it blends
retrieval, analysis, writing, and quality control into one opaque step. The
multi-agent workflow separates responsibility so route decisions, evidence, and
failure modes are easier to inspect.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route work and stop safely | Shared state | Next route | Max-iteration fallback |
| Researcher | Collect sources and notes | Query | Sources, research notes | Local mock search fallback |
| Analyst | Structure claims and risks | Research notes | Analysis notes | Continues with empty-note caveat |
| Writer | Synthesize final answer | Research and analysis notes | Final answer | Deterministic LLM fallback |

## Shared state

`ResearchState` stores the query, route history, sources, research notes, analysis
notes, final answer, agent results, trace events, and errors. These fields make
handoffs explicit and allow benchmark/report generation after the run.

## Routing policy

Supervisor starts with `researcher`, then `analyst`, then `writer`, then `done`.
If the max-iteration guardrail is reached before an answer exists, the workflow
invokes a writer fallback and stops.

## Guardrails

- Max iterations: `MAX_ITERATIONS`, default 6.
- Timeout: `TIMEOUT_SECONDS`, default 60 seconds for provider calls.
- Retry: LLM calls retry twice with exponential backoff.
- Fallback: local deterministic LLM/search behavior when provider/search is unavailable.
- Validation: Pydantic schemas validate query, sources, and benchmark metrics.

## Benchmark plan

Run the same query through the baseline and multi-agent workflow. Measure latency,
estimated token cost, quality score, citation coverage, failure rate, and trace notes.
