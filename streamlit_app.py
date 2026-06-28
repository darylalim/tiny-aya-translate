import os
from collections.abc import Iterator
from typing import Any

# Mute transformers alias-warning spam triggered by Streamlit's module watcher.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# -- Config ------------------------------------------------------------------

MODEL_ID: str = "mlx-community/tiny-aya-global-8bit-mlx"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 8192
MAX_INPUT_TOKENS: int = 8192

# Chunk budget for document translation: well below MAX_INPUT_TOKENS to leave
# room for the per-chunk prompt wrapper (instruction + chat template).
MAX_CHUNK_TOKENS: int = 7000
DOCUMENT_TYPES: list[str] = ["pdf", "docx", "pptx", "xlsx", "html"]

# -- Languages ---------------------------------------------------------------
# 67 languages across Europe, West Asia, South Asia, Asia Pacific, and Africa.

LANGUAGES: list[str] = [
    # Europe (31)
    "English",
    "Dutch",
    "French",
    "Italian",
    "Portuguese",
    "Romanian",
    "Spanish",
    "Czech",
    "Polish",
    "Ukrainian",
    "Russian",
    "Greek",
    "German",
    "Danish",
    "Swedish",
    "Bokmål",
    "Catalan",
    "Galician",
    "Welsh",
    "Irish",
    "Basque",
    "Croatian",
    "Latvian",
    "Lithuanian",
    "Slovak",
    "Slovenian",
    "Estonian",
    "Finnish",
    "Hungarian",
    "Serbian",
    "Bulgarian",
    # West Asia (5)
    "Arabic",
    "Persian",
    "Turkish",
    "Maltese",
    "Hebrew",
    # South Asia (9)
    "Hindi",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Punjabi",
    "Tamil",
    "Telugu",
    "Nepali",
    "Urdu",
    # Asia Pacific (12)
    "Tagalog",
    "Malay",
    "Indonesian",
    "Vietnamese",
    "Javanese",
    "Khmer",
    "Thai",
    "Lao",
    "Chinese",
    "Burmese",
    "Japanese",
    "Korean",
    # African (10)
    "Amharic",
    "Hausa",
    "Igbo",
    "Malagasy",
    "Shona",
    "Swahili",
    "Wolof",
    "Xhosa",
    "Yoruba",
    "Zulu",
]


# -- Pure functions -----------------------------------------------------------


def build_translation_prompt(
    text: str, source_lang: str, target_lang: str
) -> list[dict[str, str]]:
    """Build the chat messages list for a translation request."""
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"Output only the translation, nothing else.\n\n{text}"
            ),
        }
    ]


def tokenize_prompt(
    text: str, source_lang: str, target_lang: str, tokenizer: Any
) -> list[int]:
    """Apply the chat template and return the prompt token ids."""
    messages = build_translation_prompt(text, source_lang, target_lang)
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )


def clean_model_output(decoded_text: str) -> str:
    """Strip the ``<|END_RESPONSE|>`` end-of-turn marker and surrounding whitespace."""
    return decoded_text.replace("<|END_RESPONSE|>", "").strip()


def stream_translate(
    prompt_ids: list[int],
    model: Any,
    tokenizer: Any,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[str]:
    """Stream cleaned translation chunks from a pre-tokenized prompt."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    accumulated = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        accumulated += response.text
        yield clean_model_output(accumulated)


# -- Document functions (optional docling dependency) -------------------------


def docling_available() -> bool:
    """Return True if the optional ``docling`` dependency is importable."""
    import importlib.util

    return importlib.util.find_spec("docling") is not None


def load_document(file_bytes: bytes, filename: str) -> Any:
    """Parse uploaded file bytes into a ``DoclingDocument``."""
    import io

    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter

    source = DocumentStream(name=filename, stream=io.BytesIO(file_bytes))
    return DocumentConverter().convert(source).document


def chunk_document(
    doc: Any, tokenizer: Any, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:
    """Split a ``DoclingDocument`` into structure-aware text chunks."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    # HybridChunker's token budget lives on the tokenizer; mlx-lm wraps the
    # real Hugging Face tokenizer, so unwrap it via ._tokenizer.
    dl_tokenizer = HuggingFaceTokenizer(
        tokenizer=tokenizer._tokenizer, max_tokens=max_tokens
    )
    chunker = HybridChunker(tokenizer=dl_tokenizer)
    return [chunker.contextualize(chunk=c) for c in chunker.chunk(doc)]


def translate_document(
    chunks: list[str],
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
) -> Iterator[tuple[int, str]]:
    """Translate each chunk; yield ``(index, cumulative_text)`` per token."""
    done: list[str] = []
    for i, chunk in enumerate(chunks):
        prompt_ids = tokenize_prompt(chunk, source_lang, target_lang, tokenizer)
        if len(prompt_ids) > MAX_INPUT_TOKENS:
            # Skip rather than abort the whole document on one oversized chunk.
            done.append("[Section skipped: too long to translate.]")
            yield i, "\n\n".join(p for p in done if p)
            continue
        partial = ""
        for partial in stream_translate(prompt_ids, model, tokenizer):
            yield i, "\n\n".join(p for p in [*done, partial] if p)
        done.append(partial)


import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Tiny Aya Translate",
    page_icon=":material/translate:",
    layout="wide",
)


@st.cache_resource
def load_model() -> tuple[Any, Any]:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    loaded = load(MODEL_ID)
    return loaded[0], loaded[1]


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Translate")


# -- Model loading ------------------------------------------------------------

try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model, tokenizer = None, None
    model_loaded = False

# -- Session state defaults ---------------------------------------------------

st.session_state.setdefault("source_lang", "English")
st.session_state.setdefault("target_lang", "French")
st.session_state.setdefault("translate_input", "")
st.session_state.setdefault("translate_output", "")
st.session_state.setdefault("_do_translate", False)
st.session_state.setdefault("doc_source_lang", "English")
st.session_state.setdefault("doc_target_lang", "French")
st.session_state.setdefault("doc_output", "")


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    st.session_state.translate_output = ""


text_tab, doc_tab = st.tabs(
    [":material/text_fields: Text", ":material/description: Document"]
)

with text_tab:
    # -- Language bar ---------------------------------------------------------

    col_from, col_swap, col_to = st.columns([10, 1, 10], vertical_alignment="center")
    with col_from:
        st.selectbox(
            "From",
            LANGUAGES,
            key="source_lang",
            label_visibility="collapsed",
        )
    with col_swap:
        st.button(
            "",
            key="swap",
            icon=":material/swap_horiz:",
            on_click=swap_languages,
            width="stretch",
            type="tertiary",
            help="Swap languages",
        )
    with col_to:
        st.selectbox(
            "To",
            LANGUAGES,
            key="target_lang",
            label_visibility="collapsed",
        )

    # -- Warning slot (above panels) ------------------------------------------

    warning_slot = st.container()

    # -- Side-by-side text panels ---------------------------------------------

    col_input, col_output = st.columns(2)
    with col_input:
        st.text_area(
            "Input",
            height=450,
            max_chars=30000,
            key="translate_input",
            label_visibility="collapsed",
        )
    with col_output:
        output_placeholder = st.empty()
        output_placeholder.text_area(
            "Output",
            height=450,
            placeholder="Translation",
            disabled=True,
            value=st.session_state.translate_output,
            label_visibility="collapsed",
        )

    # -- Controls row ---------------------------------------------------------

    sub_translate, sub_download = st.columns(
        2, vertical_alignment="center", gap="small"
    )
    with sub_translate:
        st.button(
            "Translate",
            key="translate",
            on_click=request_translate,
            disabled=not model_loaded,
            type="primary",
            width="stretch",
        )
    with sub_download:
        st.download_button(
            "Download",
            key="download",
            data=st.session_state.translate_output,
            file_name="translation.txt",
            mime="text/plain",
            disabled=not st.session_state.translate_output.strip(),
            type="secondary",
            width="stretch",
        )

    # -- Process translation request (below controls) -------------------------

    if st.session_state._do_translate:
        st.session_state._do_translate = False
        current_input = st.session_state.translate_input
        if not current_input.strip():
            warning_slot.warning("Please enter some text first.")
        elif st.session_state.source_lang == st.session_state.target_lang:
            warning_slot.warning("Please pick two different languages.")
        elif (
            n_tok := len(
                prompt_ids := tokenize_prompt(
                    current_input,
                    st.session_state.source_lang,
                    st.session_state.target_lang,
                    tokenizer,
                )
            )
        ) > MAX_INPUT_TOKENS:
            warning_slot.warning(
                f"Input is {n_tok} tokens — please keep it under {MAX_INPUT_TOKENS}."
            )
        else:
            partial = ""
            try:
                with st.spinner("Translating..."):
                    for partial in stream_translate(prompt_ids, model, tokenizer):
                        # Non-widget element so the placeholder can be replaced
                        # mid-script without colliding with the top-of-script
                        # text_area's auto-generated widget id.
                        output_placeholder.code(
                            partial, language=None, wrap_lines=True, height=450
                        )
            except Exception as e:
                warning_slot.error(f"Translation failed: {e}")
            else:
                if not partial.strip():
                    warning_slot.warning("Model produced no output.")
                else:
                    st.session_state.translate_output = partial
                    # Rerun so the disabled output picks up the final value.
                    st.rerun()

with doc_tab:
    if not docling_available():
        st.info(
            "Document translation needs the optional `docling` package. "
            "Install it with `uv sync --extra docs`.",
            icon=":material/download:",
        )
    else:
        # -- Language bar -----------------------------------------------------

        doc_col_from, doc_col_to = st.columns(2)
        with doc_col_from:
            st.selectbox(
                "From",
                LANGUAGES,
                key="doc_source_lang",
                label_visibility="collapsed",
            )
        with doc_col_to:
            st.selectbox(
                "To",
                LANGUAGES,
                key="doc_target_lang",
                label_visibility="collapsed",
            )

        # -- Upload + controls ------------------------------------------------

        uploaded = st.file_uploader(
            "Upload a document",
            type=DOCUMENT_TYPES,
            label_visibility="collapsed",
        )
        translate_doc_clicked = st.button(
            "Translate document",
            key="translate_doc",
            disabled=not (model_loaded and uploaded is not None),
            type="primary",
            width="stretch",
        )

        # -- Warning slot + streamed output -----------------------------------

        doc_warning_slot = st.container()
        doc_output_placeholder = st.empty()
        if st.session_state.doc_output:
            doc_output_placeholder.code(
                st.session_state.doc_output,
                language=None,
                wrap_lines=True,
                height=450,
            )

        # -- Process document translation -------------------------------------

        if translate_doc_clicked and uploaded is not None:
            if st.session_state.doc_source_lang == st.session_state.doc_target_lang:
                doc_warning_slot.warning("Please pick two different languages.")
            else:
                result = ""
                try:
                    with st.spinner("Reading document..."):
                        doc = load_document(uploaded.getvalue(), uploaded.name)
                        chunks = chunk_document(doc, tokenizer)
                    if not chunks:
                        doc_warning_slot.warning(
                            "No translatable text found in the document."
                        )
                    else:
                        progress = st.progress(0.0)
                        status = st.empty()
                        last_rendered = -1
                        for idx, cumulative in translate_document(
                            chunks,
                            st.session_state.doc_source_lang,
                            st.session_state.doc_target_lang,
                            model,
                            tokenizer,
                        ):
                            result = cumulative
                            progress.progress(idx / len(chunks))
                            status.write(
                                f"Translating section {idx + 1} of {len(chunks)}"
                            )
                            # Re-render only on chunk boundaries; re-sending the
                            # whole growing document every token is O(n²).
                            if idx != last_rendered:
                                doc_output_placeholder.code(
                                    result, language=None, wrap_lines=True, height=450
                                )
                                last_rendered = idx
                        progress.progress(1.0)
                        status.empty()
                        doc_output_placeholder.code(
                            result, language=None, wrap_lines=True, height=450
                        )
                        if result.strip():
                            st.session_state.doc_output = result
                        else:
                            doc_warning_slot.warning("Model produced no output.")
                except Exception as e:
                    if result.strip():
                        st.session_state.doc_output = result
                        doc_warning_slot.error(
                            f"Translation failed after partial output: {e}"
                        )
                    else:
                        doc_warning_slot.error(f"Translation failed: {e}")

        # -- Download ---------------------------------------------------------

        st.download_button(
            "Download",
            key="download_doc",
            data=st.session_state.doc_output,
            file_name="translation.md",
            mime="text/markdown",
            disabled=not st.session_state.doc_output.strip(),
            type="secondary",
            width="stretch",
        )
