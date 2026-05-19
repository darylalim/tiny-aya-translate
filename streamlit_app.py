from collections.abc import Iterator
from typing import Any

# -- Config ------------------------------------------------------------------

MODEL_ID: str = "mlx-community/tiny-aya-global-8bit-mlx"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 8192
TOP_P: float = 0.95
MAX_INPUT_TOKENS: int = 8192

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

    sampler = make_sampler(temp=temperature, top_p=TOP_P)
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


import streamlit as st  # noqa: E402


@st.cache_resource
def load_model() -> tuple[Any, Any]:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    loaded = load(MODEL_ID)
    return loaded[0], loaded[1]


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Global Pipeline")
st.caption(
    "Translate text with the "
    "[Cohere Labs Tiny Aya Global model](https://huggingface.co/CohereLabs/tiny-aya-global)."
)


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

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "French"
if "translate_input" not in st.session_state:
    st.session_state.translate_input = ""
if "translate_output" not in st.session_state:
    st.session_state.translate_output = ""
if "_do_translate" not in st.session_state:
    st.session_state._do_translate = False


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


# -- Language bar -------------------------------------------------------------

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
        use_container_width=True,
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

# -- Warning slot (above panels) ---------------------------------------------

warning_slot = st.container()

# -- Side-by-side text panels ------------------------------------------------

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

# -- Controls row -------------------------------------------------------------

sub_translate, sub_download = st.columns(2, vertical_alignment="center", gap="small")
with sub_translate:
    st.button(
        "Translate",
        key="translate",
        on_click=request_translate,
        disabled=not model_loaded,
        type="primary",
        use_container_width=True,
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
        use_container_width=True,
    )

# -- Process translation request (below controls) ---------------------------

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
                st.rerun()  # Re-render so the output text_area picks up the final value
