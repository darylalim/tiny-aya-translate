# Tiny Aya Global Pipeline

Translate text and documents with the [Cohere Labs Tiny Aya Global model](https://huggingface.co/CohereLabs/tiny-aya-global) on Apple Silicon with MLX.

## Features

- Side-by-side translation with streaming output
- Document translation (PDF, DOCX, PPTX, XLSX, HTML) via Docling — optional
- Swap and download controls
- 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa
- Up to 8K tokens per input and per output
- 8-bit quantized MLX inference on Apple Silicon
- Local inference — no API key required

## Prerequisites

- Apple Silicon Mac
- 8 GB+ RAM recommended (~4 GB during inference)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                # core: text translation
uv sync --extra docs   # also install document translation support
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

First run downloads tiny-aya-global (~3.6 GB); document translation also downloads Docling's layout models on first use. To tune the model or sampling parameters, edit the constants at the top of `streamlit_app.py`.

## Development

```bash
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check streamlit_app.py                             # type check
```

## License

This app loads [`mlx-community/tiny-aya-global-8bit-mlx`](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx) — an 8-bit MLX-quantized fork of [Cohere Labs Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) — under [CC-BY-NC](https://cohere.com/c4ai-cc-by-nc-license) (non-commercial only).
