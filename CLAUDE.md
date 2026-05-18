## Project

Streamlit app for translating text across 67 languages using `mlx-community/tiny-aya-global-8bit-mlx` with local MLX inference on Apple Silicon.

## Stack

- Python 3.13+ with uv for project management
- Streamlit for UI
- mlx-lm for translation inference on Apple Silicon

## Structure

- `streamlit_app.py` — main app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface

## Commands

```bash
uv run streamlit run streamlit_app.py                        # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check streamlit_app.py                             # type check
```

## Conventions

- Pure functions are defined above `import streamlit` with deferred imports for `mlx_lm` inside their bodies, so tests can patch them without loading the model stack
- Config is hardcoded as module-level constants (`MODEL_ID`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `TOP_P`) at the top of `streamlit_app.py`
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search
- Swap button (`:material/swap_horiz:`, `type="tertiary"`, `help=` tooltip) flips languages via `st.session_state` and moves output into input
- `warning_slot = st.container()` is declared above the panels so the translation block (which runs later in the script) can place warnings above the input/output without needing `st.rerun()`
- Side-by-side input/output `st.text_area()` (output bound via `value=`, disabled)
- Translate button (`type="primary"`, `use_container_width=True`) uses a callback + flag pattern (`_do_translate`) and `st.rerun()` to update output
- Download button (`type="secondary"`, `use_container_width=True`) uses `st.download_button` to save translation as `translation.txt`
- Controls row is `st.columns(2)`, mirroring the side-by-side input/output panels
- Translation model loads via `@st.cache_resource def load_model()` using `mlx_lm.load`
- `translate_text` builds a chat prompt, applies the tokenizer chat template, samples via `make_sampler(temp=, top_p=)`, and generates with `mlx_lm.generate`
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` because AppTest runs scripts via `exec()`; the download button is accessed via `at.get("download_button")[0]` since it has no named accessor
- The app uses one model: `mlx-community/tiny-aya-global-8bit-mlx` (CC-BY-NC, non-commercial only)
