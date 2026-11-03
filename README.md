# llm-lithosai
[![Tests](https://github.com/james2doyle/llm-lithosai/actions/workflows/test.yml/badge.svg)](https://github.com/james2doyle/llm-lithosai/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/james2doyle/llm-lithosai/blob/main/LICENSE)

Access [LithosAI](https://docs.lithosai.com) hosted models via their
OpenAI-compatible API.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install 'https://github.com/james2doyle/llm-lithosai/archive/refs/heads/main.zip'
```

Or directly with pip:

```bash
pip install git+https://github.com/james2doyle/llm-lithosai
```

## Usage

Create a key on the [API Keys](https://console.lithosai.cloud/keys) page and save it
like this:

```bash
llm keys set lithosai
# <Paste key here>
```

Or export it as an environment variable:

```bash
export LITHOSAI_API_KEY=...
```

Run `llm models` to get the list of models. Model IDs are namespaced with a
`lithosai/` prefix, which is used to avoid conflicts with models from other plugins
that share the same upstream model names.

Run prompts like this:

```bash
llm -m lithosai/zai-org/GLM-5.3 'five great names for a pet ocelot'
```

Control reasoning effort with `none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
`max` or `ultra`:

```bash
llm -m lithosai/deepseek-ai/DeepSeek-V4.1-Flash 'Explain diffusion models' -o reasoning_effort ultra
```

JSON output and tools/schemas work as usual:

```bash
llm -m lithosai/deepseek-ai/DeepSeek-V4.1-Flash 'Invent a dog name' --schema 'name'
```

## Models

| Model family | Model IDs | Context | Max output | Thinking |
| - | - | - | - | - |
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash`, `-fast`, `-ultra` | 1,048,576 | 393,216 | On at effort `high` by default, can be turned off |
| GLM-5.3 | `zai-org/GLM-5.3` | 1,048,576 | 131,072 | Always on, cannot be turned off |
| GLM-5.3-Flash | `zai-org/GLM-5.3-Flash`, `-ultra` | 1,048,576 | 131,072 | Always on, cannot be turned off |

The `-fast` and `-ultra` IDs serve the same model as their base ID with their own
rates and rate limits.

## Options

LithosAI rejects several standard OpenAI parameters (including `logit_bias` and
`logprobs`, even `logprobs: false`, and any `n` other than `1`), so this plugin only
exposes options the API accepts: `temperature`, `top_p`, `top_k`, `max_tokens`,
`reasoning_effort`, `frequency_penalty`, `presence_penalty`, `stop`, `seed` and
`json_object`. Tools and schemas are supported.

Notes:

- `max_tokens` covers the reasoning and the answer together. DeepSeek-V4.1-Flash
  defaults to 8,192 with thinking off, 65,536 with thinking on and 131,072 at effort
  `max`/`ultra`. If you get an empty response, raise this value or lower the effort.
- DeepSeek-V4.1-Flash accepts every effort tier. `none` turns thinking off;
  `minimal`/`low` run as `low`; `medium`/`high`/`xhigh` run as `high`; `max`/`ultra`
  run as `max`.
- GLM-5.3 and GLM-5.3-Flash always think. Do not send `reasoning_effort none` or
  `ultra` - both return 400. Without `reasoning_effort` they run at `max`; only
  `low`, `high` and `max` run as distinct efforts.
- Thinking output arrives in `reasoning_content`. LLM does not display it, but it is
  stored in the logs via `response_json`.
- Stop sequences are matched in the reasoning too, so a match there ends the response
  before the answer starts.
- The documented API is text-only; no image or PDF attachments are supported.
- A key belongs to its organization, and every key in an organization shares the same
  rate limits and credit balance. On a 429 or 5xx, retry with backoff using
  `retry-after-ms` (preferred) or `retry-after`, unless `x-should-retry` is `false`.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-lithosai
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:

```bash
pytest
```
