# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Iterator
from typing import Any

# Mute transformers alias-warning spam triggered by Streamlit's module watcher.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# huggingface_hub reads both of these once, at import, and mlx_lm.load resolves
# MODEL_ID's `main` with one GET to huggingface.co on the first Translate of
# every server process, warm cache included. The model is public, so a cached
# `hf auth login` token has no business riding along as `Authorization: Bearer`
# -- the same flag is checked before get_token(), so it also skips the OAuth
# refresh POST that call can make. Telemetry off drops the daily
# /api/agent-harnesses fetch and the `agent/<harness>` User-Agent segment.
# Neither stops the GET itself. Two routes would, both keeping the cold
# download: a pinned commit `revision=`, or a cache-first load
# (snapshot_download with local_files_only=True, falling back to the download
# on a miss). Both also stop the model following upstream pushes to `main`,
# which is a product decision, not taken here. HF_HUB_OFFLINE=1 stops the GET
# and the cold download both. setdefault leaves a shell override in force,
# which is also the escape hatch if MODEL_ID is ever swapped for a gated
# model: export HF_HUB_DISABLE_IMPLICIT_TOKEN=0 in the launching shell, which
# then sends the cached token again (HF_TOKEN alone is ignored -- the flag is
# checked before get_token(), the only reader of HF_TOKEN).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# -- Config ------------------------------------------------------------------

MODEL_ID: str = "mlx-community/tiny-aya-global-8bit-mlx"
# The favicon is a local file, not a Material icon name: a `:material/…:`
# page_icon makes Streamlit emit a <link rel="shortcut icon"> pointing at
# fonts.gstatic.com, one outbound request per page load that the README's
# privacy promise does not cover. A .svg path is read and
# inlined as a data: URL at set_page_config time, so nothing is fetched and
# it renders offline. Resolved against this file, not the working directory:
# `streamlit run` can be launched from any cwd (it never chdirs; the CLI
# does abspath the script, so __file__ is absolute here), and a path that
# does not resolve fails SILENTLY -- _get_favicon_string swallows the error
# and emits the raw string as the favicon href.
FAVICON_PATH: str = os.path.join(os.path.dirname(__file__), "assets", "favicon.svg")
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 8192
MAX_INPUT_TOKENS: int = 8192

# Shared UI constants. PANEL_HEIGHT sizes the input panel, the empty-state
# output panel, the skeleton and render_output alike.
PANEL_HEIGHT: int = 450
# Input cap. Reaches the browser as HTML maxlength, so it truncates
# silently -- the placeholder names it because nothing else can.
MAX_INPUT_CHARS: int = 30000
# Download name before anything has been translated. Each reset path uses
# it rather than its own literal, so the three of them cannot drift.
DEFAULT_DOWNLOAD_NAME: str = "translation.txt"
SAME_LANGUAGE_WARNING: str = "Please pick two different languages."
NO_OUTPUT_WARNING: str = (
    "The model returned an empty translation. Try again, or rephrase the input."
)

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


import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Tiny Aya Translate",
    page_icon=FAVICON_PATH,
    layout="wide",
)


# show_spinner=False: on a cache miss the decorator's default paints its own
# "Running `load_model()`." spinner at the script cursor -- which, inside the
# translate handler, is below the controls row. ensure_model already shows the
# one indicator for the load, in the slot above the panels; reproduced on a
# cold cache, the default painted a second spinner under the buttons for the
# whole load, the off-the-fold position ensure_model exists to avoid.
@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any]:
    """Load model and tokenizer once per server process.

    ``@st.cache_resource`` is process-global, not per-session -- see
    ``ensure_model``.
    """
    from mlx_lm import load

    loaded = load(MODEL_ID)
    return loaded[0], loaded[1]


def render_output(placeholder: Any, text: str) -> None:
    """Render streamed translation output into ``placeholder``.

    Uses ``st.code`` rather than ``st.text_area`` so the placeholder can be
    replaced mid-script without colliding with a widget's auto-generated
    element id.
    """
    placeholder.code(text, language=None, wrap_lines=True, height=PANEL_HEIGHT)


def ensure_model(slot: Any) -> tuple[Any, Any] | None:
    """Load the model on demand, reporting into ``slot`` -- an ``st.empty()``.

    Called from the translate handler rather than at page load. The weights are
    ~3.6 GB, so loading them before the page renders left the user watching a
    spinner with nothing to type into or pick languages from.
    ``@st.cache_resource`` is process-global -- shared across every session and
    user, not per-session -- so the load happens once per server process, at the
    first translation rather than at startup.

    The spinner is rendered into ``slot`` rather than at the script cursor.
    Inside the translate handler that cursor sits below the panels and the
    controls row, so a bare ``st.spinner`` put the only sign of activity below
    the fold while the error landed at the top of the page. ``slot`` must be an
    ``st.empty()``, not an ``st.container()``: the spinner is transient, and the
    clear message it always sends leaves a container with a phantom child that
    holds a 16px gap for the rest of the run -- mechanism at ``warning_slot``.

    Returns ``None`` when the load fails, having already written the error into
    that same slot. There is no cheap way to know the model loads without
    loading it, so the translate button is no longer pre-emptively disabled; a
    broken install surfaces on click, next to the action that triggered it.
    ``@st.cache_resource`` does not memoize exceptions, so a later click retries
    rather than being dead for the rest of the session.
    """
    try:
        # The first load on a fresh machine pulls ~3.6 GB, which is minutes, not
        # seconds; an unqualified "Loading model..." with no elapsed time is
        # indistinguishable from a hang. show_time and the size both come from
        # the README's own promise about that first click.
        with slot.spinner(
            "Loading model... the first run downloads ~3.6 GB from Hugging Face",
            show_time=True,
        ):
            return load_model()
    except Exception as e:
        # Keep the "Failed to load model: " prefix -- a UI test matches on it.
        slot.error(
            f"Failed to load model: {e}\n\nThe first run downloads ~3.6 GB from "
            "Hugging Face. Check your connection and disk space, then click "
            "Translate again."
        )
        return None


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Translate")

# -- Session state defaults ---------------------------------------------------

st.session_state.setdefault("source_lang", "English")
st.session_state.setdefault("target_lang", "French")
st.session_state.setdefault("translate_input", "")
st.session_state.setdefault("translate_output", "")
st.session_state.setdefault("download_name", DEFAULT_DOWNLOAD_NAME)
# ("warning"|"error", text), or "" -- a message raised by the translate
# block, which runs *below* the panels and then reruns. Without this the
# rerun that repaints the output would also discard the explanation of it.
st.session_state.setdefault("translate_notice", "")
st.session_state.setdefault("_do_translate", False)


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def clear_translate_output() -> None:
    """Drop a settled translation when either language picker moves.

    The settled output is an unlabelled ``st.code`` block -- no label, no
    caption, no placeholder -- so nothing on screen names the pair that
    produced it. Change To from French to German and the card above asserts
    English -> German while the panel below still holds French, with Download
    still offering it.

    Only the output goes; ``translate_input`` is left alone so re-pressing
    Translate is one click. ``swap_languages`` needs no hook of its own -- it
    already clears the output, and Streamlit does not fire ``on_change`` for a
    programmatic session-state write.
    """
    st.session_state.translate_output = ""
    st.session_state.download_name = DEFAULT_DOWNLOAD_NAME


def swap_languages() -> None:
    """Swap source/target languages and move output into input."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    st.session_state.translate_input = st.session_state.translate_output
    # Delegate rather than repeat: whatever "clear the settled translation"
    # comes to mean must mean the same thing on both paths.
    clear_translate_output()


# -- Language bar -------------------------------------------------------------

with st.container(border=True):
    col_from, col_swap, col_to = st.columns([10, 1, 10], vertical_alignment="center")
    with col_from:
        st.selectbox(
            "From",
            LANGUAGES,
            key="source_lang",
            on_change=clear_translate_output,
            label_visibility="collapsed",
        )
    # Bounded and centred, not stretched. Every st.columns child picks up
    # min-width: calc(100% - 1.5rem) under @media (max-width: 640px)
    # regardless of its weight, so below that breakpoint this 1-unit column
    # becomes a full row -- and a tertiary button paints no background or
    # border, so a stretched one became an *invisible* full-row tap target
    # between the two pickers: measured 425x40 at a 500px viewport,
    # clickable edge to edge, one stray tap from moving the output into the
    # input and clearing it. A fixed 40 clamps to the parent, so the
    # desktop column still renders the same button and only the hit area
    # changes. It does not rescue the 640-800px band, where the column
    # shrinks toward the 16px icon and 40 clamps down with it.
    with col_swap.container(horizontal=True, horizontal_alignment="center"):
        st.button(
            "",
            key="swap",
            icon=":material/swap_horiz:",
            on_click=swap_languages,
            width=40,
            type="tertiary",
            help="Swap languages and move the translation into the input",
        )
    with col_to:
        st.selectbox(
            "To",
            LANGUAGES,
            key="target_lang",
            on_change=clear_translate_output,
            label_visibility="collapsed",
        )

# -- Warning slot (above panels) ----------------------------------------------

# st.empty(), not st.container(), because ensure_model's spinner lands here.
# Since Streamlit 1.53 st.spinner is a *transient* element: it arms a 0.5 s
# timer for the create message and sends the clear message unconditionally
# on exit. A warm @st.cache_resource load returns in microseconds, so the
# spinner never paints -- but the clear still arrives, addressed to this
# slot's child 0, and the frontend appends it to a container as a childless
# node. One child is enough to stop the block counting as empty, so a
# container that is otherwise not rendered at all was painted as a 0px flex
# item, and the enclosing block's 16px gap wrapped it: measured, both panels sat
# 16px lower from 178 ms after the click until the closing st.rerun()
# rebuilt the tree (without a rerun, the end-of-run stale sweep removes it
# instead -- either way it lives exactly as long as the run). An st.empty()
# has no child slots: its clear is addressed to the empty element's own
# path in the parent block, so the transient anchors onto the element that
# is already there, and that element's container is display:none -- no
# phantom, warm or cold; a spinner that does paint still clears cleanly.
# Holding one element costs nothing: every writer below is in an if/elif
# chain, and the notice drain fires on the run *after* the one that set it.
# This comment is the record of the mechanism; ensure_model, the guard test
# and CLAUDE.md all point here.
warning_slot = st.empty()
# Drain any notice left by the previous run's translate block. Read once
# and cleared, so it survives exactly the one rerun it was raised for.
if st.session_state.translate_notice:
    _level, _text = st.session_state.translate_notice
    st.session_state.translate_notice = ""
    if _level == "error":
        warning_slot.error(_text)
    else:
        warning_slot.warning(_text)

# -- Side-by-side text panels -------------------------------------------------

col_input, col_output = st.columns(2)
with col_input:
    # max_chars compiles to the HTML maxlength attribute, so an over-long
    # paste is clipped by the browser with no event the server can report
    # on -- the placeholder is the only surface that cap has.
    #
    # Which limit binds depends on the script, and both are real. Measured
    # with the model's own tokenizer: 30,000 characters of English prose is
    # ~5,756 tokens, well inside MAX_INPUT_TOKENS, so for Latin scripts the
    # character cap is what a user actually hits -- silently. The same
    # 30,000 characters of Thai is ~24,897 tokens, so there the token gate
    # fires first, and it says so explicitly with its own count. Naming the
    # character cap here covers the failure that has no other surface; the
    # token gate covers itself.
    st.text_area(
        "Input",
        height=PANEL_HEIGHT,
        max_chars=MAX_INPUT_CHARS,
        placeholder=f"Enter text to translate (up to {MAX_INPUT_CHARS:,} characters)",
        key="translate_input",
        label_visibility="collapsed",
    )
with col_output:
    output_placeholder = st.empty()
    # Settled output goes through render_output -- the same st.code sink the
    # streaming path already uses. Previously the panel swapped to a disabled
    # text_area the instant streaming ended, changing font, weight and colour
    # in one frame: Streamlit paints disabled content at fadedText40, which
    # measured 2.17:1 light and 3.53:1 dark under the stock Streamlit theme
    # the app shipped with at the time, so the finished translation was
    # *less* legible than the placeholder that preceded it, and a disabled
    # textarea is unselectable, so it could not even be copied. Those two
    # figures are history, not a live measurement: under the Reading Room
    # theme the same token composites over the panel to #938f87, 2.57:1
    # light, and #79746d, 3.03:1 dark -- still not a surface to settle
    # prose on, so the swap stays gone. (The font half of that frame is
    # also gone on its own: codeFont = "sans-serif" sets this st.code
    # panel in Source Sans at the input's 14px.)
    # The text_area survives as the empty state only, where it supplies the
    # placeholder and balances the input panel opposite it; its
    # "Translation appears here" is painted at that same fadedText40, so
    # the 2.57/3.03 are what it reads at -- disabled text is exempt from
    # SC 1.4.3, and it is read once.
    if st.session_state.translate_output:
        render_output(output_placeholder, st.session_state.translate_output)
    else:
        output_placeholder.text_area(
            "Output",
            height=PANEL_HEIGHT,
            placeholder="Translation appears here",
            disabled=True,
            value="",
            label_visibility="collapsed",
        )

# -- Controls row -------------------------------------------------------------

# st.columns, not st.container(horizontal=True): the horizontal container is
# a flex row whose children are `flex: 1 1 fit-content`, so a stretched button
# grows from its *intrinsic* width rather than splitting the row evenly.
# Columns are a proportional grid, which is what mirroring the panels needs.
# The reasoning is structural and still holds; the numbers that once backed
# it do NOT. They were measured under the old theme's baseFontSize = 14
# (Translate 460.6px vs Download 465.4px against 461/461 panels, a 14px gap
# against the panels' 16px), and the theme now inherits Streamlit's 16px
# base, which moves every intrinsic width and gap. Remeasure before quoting.
sub_translate, sub_download = st.columns(2, vertical_alignment="center", gap="small")
with sub_translate:
    st.button(
        "Translate",
        key="translate",
        icon=":material/translate:",
        on_click=request_translate,
        type="primary",
        width="stretch",
    )
with sub_download:
    st.download_button(
        "Download",
        key="download",
        icon=":material/download:",
        data=st.session_state.translate_output,
        file_name=st.session_state.download_name,
        mime="text/plain",
        # Downloading changes no server state, and on_click defaults to
        # "rerun" -- which re-executes the whole script for nothing.
        on_click="ignore",
        disabled=not st.session_state.translate_output.strip(),
        type="secondary",
        width="stretch",
    )

# -- Process translation request (below controls) -----------------------------

if st.session_state._do_translate:
    st.session_state._do_translate = False
    current_input = st.session_state.translate_input
    if not current_input.strip():
        warning_slot.warning("Please enter some text first.")
    elif st.session_state.source_lang == st.session_state.target_lang:
        warning_slot.warning(SAME_LANGUAGE_WARNING)
    # The two checks above are free, so they run before the weights load.
    # A failed load has already reported itself, so the chain just ends.
    elif (loaded := ensure_model(warning_slot)) is not None:
        model, tokenizer = loaded
        partial = ""
        # tokenize_prompt is inside the try: apply_chat_template raises on a
        # model whose chat template rejects this message shape, which is
        # reachable by editing MODEL_ID as the README invites. Left outside,
        # that surfaced as a raw Streamlit traceback while every other
        # failure here got the warning-slot treatment.
        try:
            prompt_ids = tokenize_prompt(
                current_input,
                st.session_state.source_lang,
                st.session_state.target_lang,
                tokenizer,
            )
            n_tok = len(prompt_ids)
            if n_tok > MAX_INPUT_TOKENS:
                warning_slot.warning(
                    f"Input is {n_tok} tokens — "
                    f"please keep it under {MAX_INPUT_TOKENS}."
                )
            else:
                # The activity indicator belongs in the panel the result
                # lands in. The script cursor here is below the panels AND
                # the controls row, so a bare spinner spends the run off
                # the fold -- the trap ensure_model already documents.
                # warning_slot is above the fold but sits above the panels,
                # so it would push the whole side-by-side row down for the
                # duration and snap it back. A skeleton at PANEL_HEIGHT
                # reflows nothing: it occupies the box render_output is
                # about to. It also clears the *previous* translation, which
                # used to sit there looking current until the first new
                # token overwrote it.
                #
                # No st.spinner alongside it: that would be a second
                # indicator for one operation, at the very cursor position
                # the paragraph above rules out.
                output_placeholder.skeleton(height=PANEL_HEIGHT)
                for partial in stream_translate(prompt_ids, model, tokenizer):
                    render_output(output_placeholder, partial)
                if partial.strip():
                    st.session_state.translate_output = partial
                    # Named here rather than at the button: target_lang
                    # keeps moving after a translation settles, so a name
                    # built at render time would describe the dropdown
                    # instead of the bytes.
                    st.session_state.download_name = (
                        f"translation-{st.session_state.target_lang}.txt"
                    )
                else:
                    # State must agree with the message. Restoring the
                    # previous translation here -- which an earlier version
                    # did, to clear the skeleton -- put a downloadable
                    # translation of *different* input directly under
                    # "the model returned an empty translation".
                    st.session_state.translate_output = ""
                    st.session_state.download_name = DEFAULT_DOWNLOAD_NAME
                    st.session_state.translate_notice = (
                        "warning",
                        NO_OUTPUT_WARNING,
                    )
                # Every exit path reruns, so the panel is always repainted
                # from translate_output rather than patched in place: that
                # is what lets the empty state come back as its text_area
                # (which cannot be re-emitted in this run -- the widget id
                # would collide) instead of a blank st.code. The notice
                # carries the message across, since this block will not run
                # again. RerunException is a BaseException, so the except
                # below does not swallow it.
                st.rerun()
        except Exception as e:
            # Keep whatever streamed before the failure; `partial` is "" if
            # it raised before the first token or before the stream even
            # started.
            if partial.strip():
                st.session_state.translate_output = partial
                st.session_state.download_name = (
                    f"translation-{st.session_state.target_lang}.txt"
                )
                st.session_state.translate_notice = (
                    "error",
                    f"Translation failed after partial output: {e}",
                )
            else:
                # No partial, so this run produced nothing -- and state
                # has to agree with the message here too. Leaving the
                # previous translation up put a downloadable translation of
                # *different* input under a failure notice, the same defect
                # the empty branch above was rewritten to remove.
                st.session_state.translate_output = ""
                st.session_state.download_name = DEFAULT_DOWNLOAD_NAME
                st.session_state.translate_notice = (
                    "error",
                    f"Translation failed: {e}",
                )
            st.rerun()
