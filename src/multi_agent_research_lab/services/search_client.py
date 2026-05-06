"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with a local mock corpus.

    The mock keeps the lab runnable without a paid search provider while preserving
    the same shape a Tavily/Bing/SerpAPI adapter would return.
    """

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        normalized = query.lower()
        corpus = [
            SourceDocument(
                title="Anthropic: Building Effective Agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Effective agent systems start with simple workflows, clear tool use, "
                    "strong evaluation, and escalation only when autonomy adds value."
                ),
                metadata={"topics": ["agents", "guardrails", "workflow"]},
            ),
            SourceDocument(
                title="LangGraph Concepts",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "LangGraph models agent applications as stateful graphs with nodes, "
                    "edges, conditional routing, persistence, and human-in-the-loop control."
                ),
                metadata={"topics": ["langgraph", "state", "routing"]},
            ),
            SourceDocument(
                title="NVIDIA NIM OpenAI-Compatible API",
                url="https://docs.api.nvidia.com/nim/reference/openai-api",
                snippet=(
                    "NVIDIA hosted models can be called through an OpenAI-compatible "
                    "chat completions interface by setting API key, base URL, and model."
                ),
                metadata={"topics": ["nvidia", "llm", "api"]},
            ),
            SourceDocument(
                title="Multi-Agent Evaluation Notes",
                url=None,
                snippet=(
                    "Compare single-agent and multi-agent systems with latency, quality, "
                    "citation coverage, cost, and failure-rate metrics."
                ),
                metadata={"topics": ["benchmark", "quality", "evaluation"]},
            ),
            SourceDocument(
                title="GraphRAG Design Pattern",
                url=None,
                snippet=(
                    "GraphRAG combines retrieval with graph-structured relationships so "
                    "answers can connect entities, evidence, and communities."
                ),
                metadata={"topics": ["graphrag", "retrieval", "analysis"]},
            ),
        ]
        scored = sorted(
            corpus,
            key=lambda doc: self._score(normalized, doc),
            reverse=True,
        )
        return scored[:max_results]

    def _score(self, query: str, document: SourceDocument) -> int:
        topics = " ".join(document.metadata.get("topics", []))
        haystack = f"{document.title} {document.snippet} {topics}".lower()
        return sum(1 for token in query.split() if token.strip(".,:;!?") in haystack)
