# SPDX-License-Identifier: Apache-2.0
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from streamlit_app import MAX_INPUT_CHARS


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear the model cache between tests.

    ``@st.cache_resource`` is process-global and outlives an AppTest instance,
    so without this a test that loads the (mocked) model would hand its result
    to every test after it. A failed load is not memoized, so only successful
    loads leak.
    """
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create an AppTest instance that has completed its initial render.

    The ``mlx_lm.load`` patch is belt-and-braces: the model loads lazily from
    the translate handler, so an initial render never reaches it (see
    ``test_page_renders_without_loading_the_model``). Any test that goes on to
    click Translate must re-patch -- ``_rerun_with_mocks`` or
    ``_run_inference_test`` -- or the real loader runs and pulls 3.6 GB.
    """
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> MagicMock:
    """Re-run the app with mocked model loading, returning the loader mock.

    Callers that expect a rerun to short-circuit before the model is needed can
    assert on the returned mock.
    """
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())) as load:
        app.run(timeout=60)
    return load


def _make_stream_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    return chunk


def _run_inference_test(input_text: str, chunk_text: str) -> AppTest:
    """Build a fresh AppTest, enter text, click Translate, and return it."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk(chunk_text)]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value(input_text)
        at.button("translate").click()
        at.run(timeout=60)
    return at


# -- Title ---------------------------------------------------------------------


def test_title_is_app_name(app: AppTest) -> None:
    assert str(app.title[0].value) == "Tiny Aya Translate"


# -- Language defaults ---------------------------------------------------------


def test_source_language_default(app: AppTest) -> None:
    assert app.selectbox[0].value == "English"


def test_target_language_default(app: AppTest) -> None:
    assert app.selectbox[1].value == "French"


# -- Swap button ---------------------------------------------------------------


def test_swap_button_exists(app: AppTest) -> None:
    assert app.button("swap") is not None


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "French"
    assert app.selectbox[1].value == "English"


def test_swap_moves_output_to_input() -> None:
    """After translating, swap should move the output into the input field."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("Bonjour")]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

        # Translate "Hello" -> "Bonjour"
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

        # Swap
        at.button("swap").click()
        at.run(timeout=60)

    # Input should now contain the previous output
    assert at.text_area[0].value == "Bonjour"
    # Output should be cleared
    assert at.text_area[1].value == ""


def test_swap_keeps_typed_input_when_nothing_is_translated(app: AppTest) -> None:
    # The move is guarded on a settled output. Unguarded, swapping with
    # nothing translated assigned "" over the typed text -- the one moment
    # swap is most wanted (typed, pair backwards, not yet translated).
    app.text_area[0].set_value("typed but untranslated")
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.text_area[0].value == "typed but untranslated"
    assert app.selectbox[0].value == "French"
    assert app.selectbox[1].value == "English"


def test_swap_clips_a_long_output_to_the_input_cap() -> None:
    # max_chars reaches the browser as maxlength and Streamlit's deserializer
    # clips only widget and default values, so a programmatic session-state
    # write went in whole: before the clip, this landed 30,500 characters in
    # the input, against the placeholder and the README.
    at = _run_inference_test(
        input_text="Hello", chunk_text="x" * (MAX_INPUT_CHARS + 500)
    )
    at.button("swap").click()
    _rerun_with_mocks(at)

    assert at.text_area[0].value == "x" * MAX_INPUT_CHARS
    assert at.session_state["translate_input"] == "x" * MAX_INPUT_CHARS


def test_swap_clips_in_the_browsers_units_not_characters() -> None:
    # maxlength counts UTF-16 code units, and an emoji is two. A character
    # slice would keep 30,000 emoji -- 60,000 units, double the cap -- and a
    # controlled textarea over maxlength rejects every single-key edit that
    # leaves it there, so the input reads as frozen. Half the cap in emoji is
    # exactly the cap in units.
    at = _run_inference_test(
        input_text="Hello", chunk_text="\U0001f600" * (MAX_INPUT_CHARS // 2 + 1)
    )
    at.button("swap").click()
    _rerun_with_mocks(at)

    moved = at.session_state["translate_input"]
    assert moved == "\U0001f600" * (MAX_INPUT_CHARS // 2)
    assert len(moved.encode("utf-16-le")) // 2 == MAX_INPUT_CHARS


# -- Text panels ---------------------------------------------------------------


def test_input_placeholder_names_the_character_cap(app: AppTest) -> None:
    # max_chars reaches the browser as HTML maxlength and truncates a long
    # paste silently, so the placeholder is its only surface. Measured with the
    # model's own tokenizer, 30,000 characters of English is ~5,756 tokens --
    # inside MAX_INPUT_TOKENS -- so for Latin scripts this really is the limit
    # a user hits. The token gate announces itself separately, with its count.
    assert "30,000" in app.text_area[0].placeholder


def test_empty_output_panel_uses_a_text_area(app: AppTest) -> None:
    # Only the EMPTY state is a text_area; a settled translation renders
    # through render_output/st.code (see test_translate_success_shows_result).
    # Two at rest: the input and the empty output.
    assert len(app.text_area) == 2


def test_output_text_area_placeholder(app: AppTest) -> None:
    # An instructional phrase, not a bare noun: this empty-state panel is
    # painted at fadedText40, and when settled output shared that grey a
    # one-word translation and a one-word placeholder were indistinguishable.
    assert app.text_area[1].placeholder == "Translation appears here"


# -- Translate flow ------------------------------------------------------------


def test_translate_button_exists(app: AppTest) -> None:
    assert app.button("translate") is not None


def test_translate_button_enabled_when_model_loaded(app: AppTest) -> None:
    assert not app.button("translate").disabled


def test_translate_success_shows_result() -> None:
    # Settled output renders through render_output/st.code, not the disabled
    # text_area -- which survives only as the empty state.
    at = _run_inference_test(input_text="Hello", chunk_text="Bonjour")
    assert at.get("code")[0].value == "Bonjour"  # ty: ignore[unresolved-attribute]
    # The output panel is no longer a text_area; only the input remains.
    assert len(at.text_area) == 1


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    app.button("translate").click()
    load = _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)
    # The check is free, so it must short-circuit before the weights load.
    load.assert_not_called()


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    app.selectbox[1].set_value("English")
    app.text_area[0].set_value("Hello")
    app.button("translate").click()
    load = _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("two different languages" in str(v) for v in warning_values)
    # Likewise free -- rejecting the pair must not cost a 3.6 GB load.
    load.assert_not_called()


# -- Language switching --------------------------------------------------------


def test_change_source_language(app: AppTest) -> None:
    app.selectbox[0].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[0].value == "Spanish"


def test_change_target_language(app: AppTest) -> None:
    app.selectbox[1].set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox[1].value == "Spanish"


# -- Input constraints ---------------------------------------------------------


def test_input_max_chars_enforced(app: AppTest) -> None:
    app.text_area[0].set_value("x" * 30001)
    _rerun_with_mocks(app)

    value = app.text_area[0].value
    assert value is not None
    assert len(value) <= 30000


def test_translate_too_many_tokens_shows_warning() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = list(range(8193))

    with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello world")
        at.button("translate").click()
        at.run(timeout=60)

    warning_values = [str(w.value) for w in at.warning]
    assert any("8193" in v and "8192" in v for v in warning_values)


def test_translate_at_input_token_limit_succeeds() -> None:
    """Input at exactly MAX_INPUT_TOKENS should translate without warning."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = list(range(8192))

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("OK")]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello world")
        at.button("translate").click()
        at.run(timeout=60)

    assert at.get("code")[0].value == "OK"  # ty: ignore[unresolved-attribute]
    assert not at.warning


def test_translation_error_shows_message() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", side_effect=RuntimeError("OOM")),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    error_values = [str(e.value) for e in at.error]
    assert any("Translation failed" in v and "OOM" in v for v in error_values)


def test_tokenizer_failure_shows_message_not_traceback() -> None:
    # apply_chat_template raises on a model whose chat template rejects this
    # message shape -- reachable by editing MODEL_ID as the README invites.
    # tokenize_prompt sits inside the translate try/except so this gets the same
    # warning-slot treatment as a streaming failure, not a red traceback.
    bad_tokenizer = MagicMock()
    bad_tokenizer.apply_chat_template.side_effect = RuntimeError("no chat template")

    with patch("mlx_lm.load", return_value=(MagicMock(), bad_tokenizer)):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    assert not at.exception
    error_values = [str(e.value) for e in at.error]
    assert any(
        "Translation failed" in v and "no chat template" in v for v in error_values
    )


def test_empty_stream_shows_warning() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", return_value=iter([])),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    warning_values = [str(w.value) for w in at.warning]
    assert any("empty translation" in v for v in warning_values)


def test_end_response_only_stream_shows_warning() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("<|END_RESPONSE|>")]),
        ),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    warning_values = [str(w.value) for w in at.warning]
    assert any("empty translation" in v for v in warning_values)


# -- Download button -----------------------------------------------------------


def test_download_button_exists(app: AppTest) -> None:
    assert len(app.get("download_button")) == 1


def test_download_button_label(app: AppTest) -> None:
    assert app.get("download_button")[0].label == "Download"  # ty: ignore[unresolved-attribute]


def test_download_button_disabled_when_output_empty(app: AppTest) -> None:
    assert app.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]


def test_download_button_enabled_when_output_present() -> None:
    at = _run_inference_test(input_text="Hello", chunk_text="Bonjour")
    assert not at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]


# -- Output text area ----------------------------------------------------------


def test_empty_output_panel_is_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Lazy model loading --------------------------------------------------------


def test_page_renders_without_loading_the_model() -> None:
    """The UI paints before the weights are touched.

    The model is loaded from the translate handler, not at page load, so a
    broken install still gets a full page -- language pickers and a usable
    input -- with no error until the user actually asks to translate.
    """
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")) as load:
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    load.assert_not_called()
    assert len(at.selectbox) == 2
    assert len(at.text_area) == 2
    assert not at.error


def test_model_load_failure_shows_error_on_translate() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_translate_button_stays_enabled_after_a_failed_load() -> None:
    # Click first, so the load actually fails before the assertion -- asserting
    # on a fresh render would only re-test the initial state, since nothing
    # loads the model there.
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.text_area[0].set_value("Hello")
        at.button("translate").click()
        at.run(timeout=60)

    # A failed load must not disable the button: @st.cache_resource does not
    # memoize the exception, so a retry works without reloading the page.
    assert not at.button("translate").disabled


# -- Settled translation and download name -------------------------------------


def test_changing_a_language_clears_the_settled_translation(app: AppTest) -> None:
    # The settled output is an unlabelled st.code block, so nothing on screen
    # names the pair that produced it -- a stale panel would silently
    # contradict the card above it.
    app.session_state["translate_output"] = "Bonjour"
    app.session_state["download_name"] = "translation-French.txt"
    _rerun_with_mocks(app)
    assert len(app.get("code")) == 1

    app.selectbox[1].set_value("German")
    _rerun_with_mocks(app)

    assert app.session_state["translate_output"] == ""
    assert app.session_state["download_name"] == "translation.txt"
    assert not app.get("code")


def test_unrelated_reruns_keep_the_settled_translation(app: AppTest) -> None:
    # Guard against an over-eager rewrite that clears on every rerun.
    app.session_state["translate_output"] = "Bonjour"
    _rerun_with_mocks(app)
    _rerun_with_mocks(app)

    assert app.session_state["translate_output"] == "Bonjour"


def test_text_download_name_records_the_target_language() -> None:
    # Captured when the output settles, not read from the picker at render
    # time -- target_lang keeps moving after a translation is done. The
    # rendered file_name is not assertable here: DownloadButton's proto has no
    # file_name field (the bytes go to the media manager and the proto carries
    # only a url), so the wiring is guarded at source level instead by
    # test_text_download_button_uses_the_recorded_name.
    at = _run_inference_test(input_text="Hello", chunk_text="Bonjour")
    assert at.session_state["download_name"] == "translation-French.txt"


def test_download_button_does_not_trigger_a_rerun(app: AppTest) -> None:
    # on_click defaults to "rerun", but a download changes no server state --
    # so every click re-executed the whole script to rebuild an identical page.
    assert app.get("download_button")[0].ignore_rerun  # ty: ignore[unresolved-attribute]


# -- Stale state after a failed or empty translation ---------------------------


def _translate_with(at: AppTest, text: str, **stream_kwargs: Any) -> AppTest:
    """Type ``text`` and click Translate with mlx_lm patched."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", **stream_kwargs),
    ):
        at.text_area[0].set_value(text)
        at.button("translate").click()
        at.run(timeout=60)
    return at


def test_empty_result_does_not_leave_the_previous_translation_downloadable() -> None:
    # The regression this guards: a `finally` restored translate_output to
    # clear the skeleton, so "the model returned an empty translation" rendered
    # above a live, downloadable translation of *different* input.
    at = _run_inference_test(input_text="hello world", chunk_text="Bonjour le monde")
    assert at.session_state["translate_output"] == "Bonjour le monde"

    _translate_with(at, "a completely different sentence", return_value=iter([]))

    assert any("empty translation" in str(w.value) for w in at.warning)
    assert not at.get("code")
    assert at.session_state["translate_output"] == ""
    assert at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]
    assert at.session_state["download_name"] == "translation.txt"


def test_failure_without_output_does_not_leave_a_stale_download() -> None:
    # Same rule as the empty path: state has to agree with the message.
    at = _run_inference_test(input_text="hello world", chunk_text="Bonjour le monde")

    _translate_with(at, "goodbye", side_effect=RuntimeError("OOM"))

    assert any("Translation failed: OOM" in str(e.value) for e in at.error)
    assert at.session_state["translate_output"] == ""
    assert at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]


def test_failure_after_partial_output_keeps_the_partial() -> None:
    def stream_then_die(*_args: object, **_kwargs: object):
        yield _make_stream_chunk("Bonjour le ")
        raise RuntimeError("OOM")

    at = AppTest.from_file("streamlit_app.py")
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        at.run(timeout=60)
    _translate_with(at, "hello world", side_effect=stream_then_die)

    assert any("after partial output" in str(e.value) for e in at.error)
    assert at.session_state["translate_output"] == "Bonjour le"
    assert not at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]
    assert at.session_state["download_name"] == "translation-French.txt"


def test_first_failure_keeps_the_empty_state_panel() -> None:
    # An earlier fix painted a blank st.code here, leaving a featureless grey
    # rectangle where the "Translation appears here" panel had been.
    at = AppTest.from_file("streamlit_app.py")
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        at.run(timeout=60)
    _translate_with(at, "hello", side_effect=RuntimeError("OOM"))

    assert not at.get("code")
    assert len(at.text_area) == 2
