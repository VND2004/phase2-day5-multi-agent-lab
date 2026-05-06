"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """OpenAI-compatible LLM client.

    NVIDIA NIM uses an OpenAI-compatible API, so the same SDK path works with
    `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, and `AGENT_MODEL`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with a deterministic local fallback."""

        if not self.settings.llm_api_key:
            return self._fallback_response(system_prompt, user_prompt, "missing API key")

        try:
            return self._complete_with_provider(system_prompt, user_prompt)
        except Exception as exc:  # pragma: no cover - depends on live provider/network
            logger.warning("LLM provider failed, using local fallback: %s", exc)
            return self._fallback_response(system_prompt, user_prompt, str(exc))

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=2))
    def _complete_with_provider(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install the llm extra to use live LLM calls: pip install -e .[llm]"
            ) from exc

        client = OpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.timeout_seconds,
        )
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=None,
        )

    def _fallback_response(self, system_prompt: str, user_prompt: str, reason: str) -> LLMResponse:
        role = system_prompt.splitlines()[0].strip() if system_prompt else "Local assistant"
        content = (
            f"{role}\n\n"
            f"Local deterministic response used because the live LLM was unavailable ({reason}).\n"
            f"Task summary: {user_prompt.strip()[:900]}"
        )
        estimated_input = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        estimated_output = max(1, len(content) // 4)
        return LLMResponse(
            content=content,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
        )
