# Benchmark Report

## Summary

This report compares a single-agent baseline with the multi-agent workflow.

| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Failure Rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent baseline | 10.12 | 0.0004 | 4.0 | 0% | 0% | 0 sources; routes: single-agent |
| multi-agent workflow | 17.65 | 0.0020 | 9.0 | 0% | 0% | 5 sources; routes: researcher > analyst > writer > done |

## Failure Mode

Primary risk: a live provider, search API, or network call can fail. The implementation contains deterministic fallbacks, records errors in shared state, and stops through a max-iteration guardrail instead of looping indefinitely.

## Trace

Each run records route decisions and agent spans in `ResearchState.trace`.
