import pytest
from llm.plugins import pm


@pytest.fixture
def registered():
    class Capture:
        def __init__(self):
            self.pairs = []

        def __call__(self, model, async_model=None, aliases=None):
            self.pairs.append((model, async_model, aliases))

    return Capture()


def test_plugin_is_installed(monkeypatch):
    # LLM skips entry-point loading under pytest (sys._called_from_test),
    # so load manually the same way load_plugins() does:
    monkeypatch.setenv("LITHOSAI_API_KEY", "test-key")
    pm.load_setuptools_entrypoints("llm")
    try:
        names = [mod.__name__ for mod in pm.get_plugins()]
        assert "llm_lithosai" in names
    finally:
        for name, mod in pm.list_name_plugin():
            if name == "llm_lithosai":
                pm.unregister(mod)


def test_models_register_with_key(monkeypatch, registered):
    monkeypatch.setenv("LITHOSAI_API_KEY", "test-key")
    from llm_lithosai import LithosAIChat, MODELS, register_models

    register_models(registered)
    assert [(m.model_id, m.model_name) for m, _, _ in registered.pairs] == [
        (f"lithosai/{model_id}", model_id) for model_id in MODELS
    ]
    for model, async_model, _ in registered.pairs:
        assert isinstance(model, LithosAIChat)
        # Namespaced IDs, raw API IDs in model_name
        assert model.model_id.startswith("lithosai/")
        assert model.model_name == model.model_id.removeprefix("lithosai/")
        # Each sync model registered in an (sync, async) pair
        assert async_model.model_id == model.model_id
        assert async_model.model_name == model.model_name


def test_registered_model_ids_match_docs():
    from llm_lithosai import MODELS

    assert set(MODELS) == {
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "deepseek-ai/DeepSeek-V4.1-Flash-fast",
        "deepseek-ai/DeepSeek-V4.1-Flash-ultra",
        "zai-org/GLM-5.3",
        "zai-org/GLM-5.3-Flash",
        "zai-org/GLM-5.3-Flash-ultra",
    }


def test_no_registration_without_key(monkeypatch, registered):
    monkeypatch.delenv("LITHOSAI_API_KEY", raising=False)
    from llm_lithosai import register_models

    register_models(registered)
    assert registered.pairs == []


def test_options_accept_lithosai_values():
    from llm_lithosai import LithosAIOptions

    options = LithosAIOptions(
        reasoning_effort="ultra",
        max_tokens=8192,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        stop="</answer>",
        frequency_penalty=0.2,
        presence_penalty=-0.2,
        seed=42,
    )
    assert options.reasoning_effort == "ultra"
    assert options.max_tokens == 8192
    assert options.temperature == 0.5
    assert options.top_p == 0.9
    assert options.top_k == 40
    assert options.stop == "</answer>"
    assert options.frequency_penalty == 0.2
    assert options.presence_penalty == -0.2
    assert options.seed == 42


def test_options_reject_packaged_openai_only_helpers():
    from pydantic import ValidationError

    from llm_lithosai import LithosAIOptions

    # reasoning_effort validates against the LithosAI enum
    with pytest.raises(ValidationError):
        LithosAIOptions(reasoning_effort="bogus")
    # Ranges match the API
    with pytest.raises(ValidationError):
        LithosAIOptions(temperature=3)
    with pytest.raises(ValidationError):
        LithosAIOptions(top_p=0)
    with pytest.raises(ValidationError):
        LithosAIOptions(top_k=0)
    with pytest.raises(ValidationError):
        LithosAIOptions(max_tokens=0)


def test_options_omit_unsupported_parameters():
    from llm_lithosai import LithosAIOptions

    # LithosAI returns 400 for logit_bias and logprobs, so they must not
    # be exposed as options.
    assert "logit_bias" not in LithosAIOptions.model_fields
    assert "logprobs" not in LithosAIOptions.model_fields
    assert "top_logprobs" not in LithosAIOptions.model_fields


def test_models_are_text_only():
    from llm_lithosai import LithosAIChat

    assert not LithosAIChat("zai-org/GLM-5.3").attachment_types


def test_model_str():
    from llm_lithosai import LithosAIChat

    assert str(LithosAIChat("zai-org/GLM-5.3")) == "LithosAI: lithosai/zai-org/GLM-5.3"
