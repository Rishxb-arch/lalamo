<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/lalamo-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://artifacts.trymirai.com/social/about_us.mp3"><img src="https://img.shields.io/badge/Listen-Podcast-red" alt="Listen to our podcast"></a>
<a href="https://docsend.com/v/76bpr/mirai2025"><img src="https://img.shields.io/badge/View-Deck-red" alt="View our deck"></a>
<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord" alt="Discord"></a>
<a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
<a href="https://docs.trymirai.com/overview/lalamo"><img src="https://img.shields.io/badge/Read-Docs-blue" alt="Read docs"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# lalamo

A set of tools for adapting Large Language Models to on-device inference using the [uzu](https://github.com/trymirai/uzu) inference engine.

## Quick Start

To get the list of [supported models](https://trymirai.com/models), run:

```bash
uv run lalamo list-models
```

To convert a model, run:

```bash
uv run lalamo convert MODEL_REPO
```

Note: on some CPU platform you may be getting an error saying `The precision 'F16_F16_F32' is not supported by dot_general on CPU`. This is due to a bug in XLA, which causes matmuls inside `jax.jit` not work correctly on CPUs. The workaround is to set the environment variable `JAX_DISABLE_JIT=1` when running the conversion.

After that, you can find the converted model in the `models` folder. For more options see `uv run lalamo convert --help`.

## Chat

```bash
uv run lalamo chat MODEL_PATH --message "What is 17 * 19?" --thinking-budget 16
```

`--thinking-budget N` caps how many tokens a model may spend in its reasoning section. At the cap, generation appends that model's end-of-thinking tag and continues with the answer. `0` skips reasoning. Tag tokens are not part of the budget, and they do count toward `--max-tokens`. Omitting the flag leaves generation unchanged. If the prompt already ends with that tag, thinking is closed and the budget does not insert another copy.

Greedy Qwen3-0.6B (`--temperature 0`) on `What is 17 * 19?` with `--thinking-budget 16` (stop token omitted):

```text
<think>
Okay, so I need to figure out what 17 multiplied by
</think>

17 multiplied by 19 is 323. Let me check that again. 17 times 20 would be 340, so subtracting 17 gives 340 - 17 = 323. Yep, that seems right.
```

With no budget, the same prompt was still inside `<think>` after 240 tokens, and the first 16 of those tokens are the prefix above. `--thinking-budget 0` skips reasoning and answers `17 multiplied by 19 is equal to` `17 × 19 = 323`.

Models with no end-of-thinking tag reject the flag. The continuous-batching server does not accept it either; use `chat`, `LanguageModel.generate_tokens`, or `LanguageModel.stream_tokens`.

## Model Support

To add support for a new model, write the corresponding [ModelSpec](lalamo/model_import/model_specs), as shown in the example below:

```python
ModelSpec(
    vendor="Google",
    family="Gemma-3",
    name="Gemma-3-1B-Instruct",
    size="1B",
    quantization=None,
    repo="google/gemma-3-1b-it",
    config_type=HFGemma3TextConfig,
    weights_type=WeightsType.SAFETENSORS,
)
```

## Optional Features

### PyAudio

PyAudio enables audio playback for TTS models and is used as an optional Lalamo feature because it requires [PortAudio](https://www.portaudio.com/) to be installed.

How to:

- **macOS**: `brew install portaudio` ([formula](https://formulae.brew.sh/formula/portaudio))
- **Debian/Ubuntu**: `apt-get install portaudio19-dev python-all-dev`
- **Other Linux**: [PortAudio build instructions](https://www.portaudio.com/docs/v19-doxydocs/compile_linux.html)

Then run :

```bash
uv run --with  pyaudio lalamo path/to/model --replay
```
