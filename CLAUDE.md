## Project

Streamlit app for translating text and documents across 67 languages using `mlx-community/tiny-aya-global-8bit-mlx` with local MLX inference on Apple Silicon.

## Stack

- Python 3.13+ with uv for project management
- Streamlit for UI
- mlx-lm for translation inference on Apple Silicon
- docling (optional `docs` extra) for parsing uploaded documents

## Structure

- `streamlit_app.py` — main app: config, pure functions, Streamlit UI
- `test_streamlit_app.py` — pytest unit tests for pure functions
- `test_streamlit_ui.py` — pytest UI tests for Streamlit interface
- `.streamlit/config.toml` — Nord theme, light and dark modes

## Commands

```bash
uv sync --extra docs                                         # install optional document support
uv run streamlit run streamlit_app.py                        # run the app
uv run pytest test_streamlit_app.py test_streamlit_ui.py -v  # run tests
uv run ruff check --fix .                                    # lint
uv run ruff format .                                         # format
uv run ty check                                              # type check
```

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Conventions

- Pure functions are defined above `import streamlit` with deferred imports for `mlx_lm` and `docling` inside their bodies, so tests can patch them without loading the model stack
- Config is hardcoded as module-level constants (`MODEL_ID`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`, `MAX_INPUT_TOKENS`, `MAX_CHUNK_TOKENS`, `DOCUMENT_TYPES`) at the top of `streamlit_app.py`
- `streamlit_app.py` sets `TRANSFORMERS_VERBOSITY=error` via `os.environ.setdefault` at the top of the file, muting transformers' image-processor alias warnings that Streamlit's module watcher otherwise triggers on every rerun
- Language selectboxes use the flat `LANGUAGES` list (67 items) with collapsed labels and Streamlit's built-in type-to-search; each tab's language bar is wrapped in `st.container(border=True)` as a card (Text tab: from/swap/to; Document tab: from/to)
- Swap button (`:material/swap_horiz:`, `type="tertiary"`, `help=` tooltip) flips languages via `st.session_state` and moves output into input
- `warning_slot = st.container()` is declared above the panels so the translation block (which runs later in the script) can place warnings above the input/output without needing `st.rerun()`
- Side-by-side input/output `st.text_area()`
- Output `text_area` is rendered into an `st.empty()` placeholder (`output_placeholder`) so streaming can replace it progressively
- Streaming fills the placeholder with `st.code(..., language=None, wrap_lines=True, height=450)`, not `text_area` — repeating a widget call in the same script run would collide on the auto-generated element id, but `st.code` is non-widget and replaces freely
- Translate button (`type="primary"`, `width="stretch"`) uses a callback + flag pattern (`_do_translate`); after streaming, the translate block calls `st.rerun()` so the disabled output `text_area` and download button re-render with the final value
- Translate block validates input (`tokenize_prompt` + `MAX_INPUT_TOKENS` check) and wraps streaming in `try/except` (errors → `warning_slot.error`, empty output → `warning_slot.warning`)
- Download button (`type="secondary"`, `width="stretch"`) uses `st.download_button` to save translation as `translation.txt`
- Controls row is `st.columns(2)`, mirroring the side-by-side input/output panels
- Full-width buttons use `width="stretch"` — Streamlit 1.58's replacement for the deprecated `use_container_width`; a source-level test in `test_streamlit_app.py` guards against reintroducing the old arg
- Translation model loads via `@st.cache_resource def load_model()` using `mlx_lm.load`
- `tokenize_prompt` applies the chat template with `tokenize=True` and returns the prompt token ids — its `len()` gates against `MAX_INPUT_TOKENS`, and the same ids are passed to `stream_translate` so the prompt is tokenized exactly once per translation
- `stream_translate` takes pre-tokenized prompt ids, calls `mlx_lm.stream_generate` with `sampler=make_sampler(temp=)`, and yields the cleaned running result after each chunk
- `clean_model_output` strips whitespace and the `<|END_RESPONSE|>` token leaked by the model
- UI tests use `streamlit.testing.v1.AppTest`; mocks target `mlx_lm` because AppTest runs scripts via `exec()`; download buttons have no named accessor, so `at.get("download_button")` returns both (`[0]` Text tab, `[1]` Document tab)
- `st.set_page_config` is the first Streamlit command (right after `import streamlit as st`, before the `@st.cache_resource` decorator runs): sets `page_title`, `page_icon=":material/translate:"`, and `layout="wide"` so the side-by-side text panels get horizontal room; a source-level test asserts the title, icon, and wide layout
- Session-state defaults are seeded with `st.session_state.setdefault(key, default)` (one line each), not `if key not in st.session_state`
- The UI is split into `st.tabs([":material/text_fields: Text", ":material/description: Document"])`: the Text tab is the original side-by-side flow; the Document tab translates uploaded files
- Document functions (`docling_available`, `load_document`, `chunk_document`, `translate_document`) are pure functions with deferred `docling` imports; `docling` is an optional `docs` extra, and the Document tab shows an install hint when `docling_available()` is `False`
- `chunk_document` runs Docling's `HybridChunker`; the token budget lives on a `HuggingFaceTokenizer` (the chunker takes no `max_tokens`), and mlx-lm's `TokenizerWrapper` is unwrapped via `._tokenizer` to reach the raw HF tokenizer
- `cached_document_chunks` (defined after `load_model`, decorated `@st.cache_data(max_entries=8, show_spinner=False)`) wraps `load_document` + `chunk_document` so re-translating the same upload skips the Docling convert+chunk; it fetches the tokenizer from `load_model()` internally rather than taking it as an arg (the tokenizer is unhashable and would defeat `@st.cache_data`'s key hashing). The Document tab calls this wrapper, not the pure functions directly; the pure functions stay for unit tests
- `translate_document` reuses `tokenize_prompt` + `stream_translate` per chunk and yields `(chunk_index, cumulative_text)` on every token; chunks join with `\n\n`
- `MAX_CHUNK_TOKENS` (7000) sits below `MAX_INPUT_TOKENS` to leave room for the per-chunk prompt wrapper (instruction + chat template); `translate_document` still re-checks each chunk and emits a `[Section skipped]` marker for any prompt over `MAX_INPUT_TOKENS`
- The Document tab uses a plain `if st.button():` block (no `_do_translate` flag, no `st.rerun()`): output streams into an `st.code` placeholder, its `doc_source_lang`/`doc_target_lang` selectboxes use distinct keys to avoid widget-id collisions, and the `download_doc` button is rendered last so it picks up `st.session_state.doc_output`
- The Document tab re-renders its `st.code` output only on chunk boundaries — re-sending the whole accumulating document every token would be O(n²); a mid-document failure still saves the partial result to `st.session_state.doc_output`
- The app uses one model: `mlx-community/tiny-aya-global-8bit-mlx` (CC-BY-NC, non-commercial only)
- The app ships a Nord theme in `.streamlit/config.toml` with both `[theme.light]` (Snow Storm) and `[theme.dark]` (Polar Night) variants — defining both is what surfaces the light/dark switch in Streamlit's settings menu; shared typography/shape/Aurora-status colors live in the bare `[theme]` section. `.gitignore` tracks `config.toml` but ignores the rest of `.streamlit/` (e.g. `secrets.toml`)
- Both theme variants set `primaryColor = "#4c6a93"` (deep frost, not the lighter `#88c0d0`/`#5e81ac`): Streamlit renders primary-button labels white at body size, so the primary must clear WCAG AA for normal text (4.5:1) — white-on-`#4c6a93` is 5.5:1. Light-mode `linkColor`/`blueColor` share `#4c6a93` (4.8:1 as link text on the light bg); dark-mode link stays `#81a1c1` (4.6:1). `test_primary_button_text_readable_in_both_modes` and `test_link_text_readable_in_both_modes` compute the ratios (≥ 4.5:1) and guard them
- Ruff lint selection is `["E", "F", "I", "W", "UP", "B", "SIM"]` — pycodestyle, pyflakes, isort, pyupgrade, bugbear, and simplify
- `ty check` covers the whole project; the four `AppTest.get("download_button")` assertions carry inline `# ty: ignore[unresolved-attribute]` because `.get()` returns `Element | Block` and download-button widgets have no typed accessor to narrow to — scoped to those lines so the rule still catches typos elsewhere in the tests
- The `cached_document_chunks` tests reference `streamlit_app.cached_document_chunks` with **no** `# ty: ignore` — the editor's incremental LSP flags it `unresolved-attribute` (the `@st.cache_data` decorator trips it), but the `ty check` CLI (the gate) resolves it fine, so adding an ignore would make the CLI fail with `unused-ignore-comment`. Trust the CLI, not the editor squiggle
