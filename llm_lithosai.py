from enum import Enum

import llm
from pydantic import Field

from llm.default_plugins.openai_models import (
    AsyncChat,
    Chat,
)

API_BASE = "https://api.lithosai.cloud/v1"

MODELS = (
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-fast",
    "deepseek-ai/DeepSeek-V4.1-Flash-ultra",
    "zai-org/GLM-5.3",
    "zai-org/GLM-5.3-Flash",
    "zai-org/GLM-5.3-Flash-ultra",
)

# LithosAI documents a text-only Chat Completions API, so no model
# declares attachment types.


class LithosAIReasoningEffortEnum(str, Enum):
    # LithosAI extends the usual tiers with "ultra" (DeepSeek only).
    none = "none"
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"
    ultra = "ultra"


class LithosAIOptions(llm.Options):
    # Only options the LithosAI Chat Completions API accepts. It rejects
    # OpenAI's logit_bias and logprobs (even logprobs=false) and any n
    # other than 1, so nothing here inherits from the OpenAI
    # SharedOptions class.
    temperature: float | None = Field(
        description=(
            "What sampling temperature to use, between 0 and 2. Higher values like "
            "0.8 will make the output more random, while lower values like 0.2 will "
            "make it more focused and deterministic."
        ),
        ge=0,
        le=2,
        default=None,
    )
    top_p: float | None = Field(
        description=(
            "An alternative to sampling with temperature, called nucleus sampling, "
            "where the model considers the results of the tokens with top_p "
            "probability mass. So 0.1 means only the tokens comprising the top "
            "10% probability mass are considered. Recommended to use top_p or "
            "temperature but not both."
        ),
        gt=0,
        le=1,
        default=None,
    )
    top_k: int | None = Field(
        description=(
            "Non-standard sampling parameter: only sample from the top K tokens. "
            "LithosAI accepts this alongside temperature and top_p."
        ),
        ge=1,
        default=None,
    )
    max_tokens: int | None = Field(
        description=(
            "Maximum number of completion tokens to generate. This covers the "
            "reasoning and the answer together - if you get an empty response, "
            "raise this value."
        ),
        ge=1,
        default=None,
    )
    reasoning_effort: LithosAIReasoningEffortEnum | None = Field(
        description=(
            "Effort spent on reasoning. One of none, minimal, low, medium, high, "
            "xhigh, max or ultra. DeepSeek-V4.1-Flash accepts all of them; GLM-5.3 "
            "and GLM-5.3-Flash always think and do not accept none or ultra."
        ),
        default=None,
    )
    frequency_penalty: float | None = Field(
        description=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens based "
            "on their existing frequency in the text so far, decreasing the model's "
            "likelihood to repeat the same line verbatim."
        ),
        ge=-2,
        le=2,
        default=None,
    )
    presence_penalty: float | None = Field(
        description=(
            "Number between -2.0 and 2.0. Positive values penalize new tokens based "
            "on whether they appear in the text so far, increasing the model's "
            "likelihood to talk about new topics."
        ),
        ge=-2,
        le=2,
        default=None,
    )
    stop: str | None = Field(
        description=(
            "A string where the API will stop generating further tokens. Stop "
            "sequences are also matched in the reasoning, so a match there ends "
            "the response before the answer starts."
        ),
        default=None,
    )
    seed: int | None = Field(
        description=(
            "Integer seed to attempt to sample deterministically. LithosAI accepts "
            "it, but does not guarantee reproducible output."
        ),
        default=None,
    )
    json_object: bool | None = Field(
        description="Output a valid JSON object {...}. Prompt must mention JSON.",
        default=None,
    )


class _LithosAIShared:
    def __init__(self, model_id):
        # Prefix the LLM-facing model_id with our namespace so these
        # models cannot collide with other plugins (e.g. another plugin
        # serving the same upstream model). The raw provider ID is kept
        # in model_name, which is what gets sent to the API.
        super().__init__(
            model_id=f"lithosai/{model_id}",
            model_name=model_id,
            api_base=API_BASE,
            supports_schema=True,
            supports_tools=True,
        )

    def __str__(self):
        return "LithosAI: {}".format(self.model_id)


class LithosAIChat(_LithosAIShared, Chat):
    needs_key = "lithosai"
    key_env_var = "LITHOSAI_API_KEY"
    Options = LithosAIOptions


class LithosAIAsyncChat(_LithosAIShared, AsyncChat):
    needs_key = "lithosai"
    key_env_var = "LITHOSAI_API_KEY"
    Options = LithosAIOptions


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "lithosai", LithosAIChat.key_env_var)
    if not key:
        return
    for model_id in MODELS:
        register(LithosAIChat(model_id), LithosAIAsyncChat(model_id))
