from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.llm_model
    assert settings.nvidia_base_url.startswith("https://")
    assert settings.max_iterations >= 1
