# SPDX-License-Identifier: Apache-2.0
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

from streamlit_app import (
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    LANGUAGES,
    MAX_INPUT_CHARS,
    escape_markdown,
)


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear the model cache between tests.

    ``@st.cache_resource`` is process-global and outlives an AppTest instance,
    so without this a test that loads the (mocked) model would hand its result
    to every test after it. A failed load is not memoized, so only successful
    loads leak.
    """
    st.cache_resource.clear()


def _new_app() -> AppTest:
    """Construct the AppTest with the timeout every run in this file needs.

    ``from_file`` resolves a relative path against the *caller's* file, so
    this must stay in a module beside ``streamlit_app.py``. The default
    timeout is 3 s; set here once rather than as ``timeout=60`` on each
    ``run`` call, so a bare ``at.run()`` anywhere cannot fall back to 3.
    """
    return AppTest.from_file("streamlit_app.py", default_timeout=60)


def _run(at: AppTest) -> AppTest:
    """Run the script and fail if it raised.

    ``AppTest.run`` returns normally when the script raises: the runner
    reports the run as SCRIPT_STOPPED_WITH_SUCCESS "even if we were stopped
    with an exception" (local_script_runner.py), and the traceback lands in
    the element tree as ``at.exception`` with session state intact -- so a
    test that asserts on state alone passes over a crash, and ``at.error``
    (the alert list) does not see it either. Every run in this file goes
    through here rather than through ``at.run`` directly.
    """
    at.run()
    assert not at.exception, [(e.proto.type, e.message) for e in at.exception]
    return at


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
        at = _new_app()
        _run(at)
    return at


def _rerun_with_mocks(app: AppTest) -> MagicMock:
    """Re-run the app with mocked model loading, returning the loader mock.

    Callers that expect a rerun to short-circuit before the model is needed can
    assert on the returned mock.
    """
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())) as load:
        _run(app)
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
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value(input_text)
        at.button("translate").click()
        _run(at)
    return at


# -- Title ---------------------------------------------------------------------


def test_title_is_app_name(app: AppTest) -> None:
    assert str(app.title[0].value) == "Tiny Aya Translate"


# -- Language defaults ---------------------------------------------------------


def test_source_language_default(app: AppTest) -> None:
    assert app.selectbox("source_lang").value == DEFAULT_SOURCE_LANG == "English"


def test_target_language_default(app: AppTest) -> None:
    assert app.selectbox("target_lang").value == DEFAULT_TARGET_LANG == "French"


# -- URL-bound pickers ---------------------------------------------------------


def _app_at(**params: str) -> AppTest:
    """Render the page as if opened at ?key=value&… (bound pickers only)."""
    at = _new_app()
    for key, value in params.items():
        at.query_params[key] = value
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        _run(at)
    return at


def test_pickers_open_at_rest_with_a_clean_url() -> None:
    at = _app_at()
    assert at.selectbox("source_lang").value == DEFAULT_SOURCE_LANG
    assert at.selectbox("target_lang").value == DEFAULT_TARGET_LANG
    assert not dict(at.query_params)


def test_pickers_follow_the_url() -> None:
    at = _app_at(source_lang="French", target_lang="English")
    assert at.selectbox("source_lang").value == "French"
    assert at.selectbox("target_lang").value == "English"


def test_reverse_pair_link_does_not_collapse_to_the_same_language() -> None:
    # The regression the widget-owned defaults exist for. A bound value is
    # dropped from the URL when it equals the *widget's* default and read
    # back against it; with session-state seeds both widgets defaulted to
    # LANGUAGES[0], so ?target_lang=English was dropped as "the default" and
    # the seeded French won -- a shared French -> English link opened as
    # French -> French, the one pair the app refuses. Reproduced before the
    # seeds went. With index= the target's default is French, so English is
    # a real value. The premise is asserted with it: the regression only
    # exists while the target's default is not LANGUAGES[0], and the URL
    # value below is that first entry, so a reordering that put the target
    # default at index 0 would void the test rather than quietly pass it.
    first = LANGUAGES[0]
    assert LANGUAGES.index(DEFAULT_TARGET_LANG) != 0
    at = _app_at(target_lang=first)
    assert at.selectbox("target_lang").value == first


def test_unknown_url_value_falls_back_to_the_default() -> None:
    at = _app_at(source_lang="Klingon")
    assert at.selectbox("source_lang").value == "English"
    assert not at.error
    assert "source_lang" not in dict(at.query_params)


# -- Swap button ---------------------------------------------------------------


def test_swap_button_exists(app: AppTest) -> None:
    # The wording names both side effects (output wiped, input overwritten);
    # a plain "Swap languages" hid them, so the text is pinned, not just the
    # tooltip's presence.
    assert (
        app.button("swap").help
        == "Swap languages and move the translation into the input"
    )


def test_swap_flips_languages(app: AppTest) -> None:
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.selectbox("source_lang").value == "French"
    assert app.selectbox("target_lang").value == "English"
    # Both pickers are bound to the URL, and a callback's session-state
    # write is the sanctioned programmatic route: the URL follows the swap.
    # (A picker change is synced by the frontend, which AppTest does not
    # run, so the URL is asserted here and not after set_value.)
    assert dict(app.query_params) == {
        "source_lang": ["French"],
        "target_lang": ["English"],
    }


def test_swap_moves_output_to_input() -> None:
    """After translating, swap should move the output into the input field."""
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("Bonjour")]),
        ),
    ):
        at = _new_app()
        _run(at)

        # Translate "Hello" -> "Bonjour"
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

        # Swap
        at.button("swap").click()
        _run(at)

    # Input should now contain the previous output
    assert at.text_area("translate_input").value == "Bonjour"
    # Output should be cleared
    assert at.text_area[1].value == ""


def test_swap_keeps_typed_input_when_nothing_is_translated(app: AppTest) -> None:
    # The move is guarded on a settled output. Unguarded, swapping with
    # nothing translated assigned "" over the typed text -- the one moment
    # swap is most wanted (typed, pair backwards, not yet translated).
    app.text_area("translate_input").set_value("typed but untranslated")
    app.button("swap").click()
    _rerun_with_mocks(app)

    assert app.text_area("translate_input").value == "typed but untranslated"
    assert app.selectbox("source_lang").value == "French"
    assert app.selectbox("target_lang").value == "English"


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

    assert at.text_area("translate_input").value == "x" * MAX_INPUT_CHARS
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
    assert "30,000" in app.text_area("translate_input").placeholder


def test_empty_output_panel_uses_a_text_area(app: AppTest) -> None:
    # Only the EMPTY state is a text_area; a settled translation renders
    # through render_output/st.code (see test_translate_success_shows_result).
    # Two at rest: the input and the empty output. The output carries no key
    # (it is never written programmatically), so it is the one widget this
    # file still reaches by index -- text_area[1] -- where the keyed input
    # and pickers go by key, the way the buttons always did.
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
    app.selectbox("target_lang").set_value("English")
    app.text_area("translate_input").set_value("Hello")
    app.button("translate").click()
    load = _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("two different languages" in str(v) for v in warning_values)
    # Likewise free -- rejecting the pair must not cost a 3.6 GB load.
    load.assert_not_called()


# -- Language switching --------------------------------------------------------


def test_change_source_language(app: AppTest) -> None:
    app.selectbox("source_lang").set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox("source_lang").value == "Spanish"


def test_change_target_language(app: AppTest) -> None:
    app.selectbox("target_lang").set_value("Spanish")
    _rerun_with_mocks(app)

    assert app.selectbox("target_lang").value == "Spanish"


# -- Input constraints ---------------------------------------------------------


def test_input_max_chars_enforced(app: AppTest) -> None:
    app.text_area("translate_input").set_value("x" * 30001)
    _rerun_with_mocks(app)

    value = app.text_area("translate_input").value
    assert value is not None
    assert len(value) <= 30000


def test_translate_too_many_tokens_shows_warning() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = list(range(8193))

    with patch("mlx_lm.load", return_value=(MagicMock(), mock_tokenizer)):
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello world")
        at.button("translate").click()
        _run(at)

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
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello world")
        at.button("translate").click()
        _run(at)

    assert at.get("code")[0].value == "OK"  # ty: ignore[unresolved-attribute]
    assert not at.warning


def test_translation_error_shows_message() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", side_effect=RuntimeError("OOM")),
    ):
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

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
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

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
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

    warning_values = [str(w.value) for w in at.warning]
    assert any("empty translation" in v for v in warning_values)


def test_notice_survives_exactly_one_rerun() -> None:
    # translate_notice carries a message across the translate block's closing
    # st.rerun() and is read once and cleared: the rerun that repaints the
    # panel paints the warning, and the next unrelated rerun does not.
    at = _new_app()
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("mlx_lm.stream_generate", return_value=iter([])),
    ):
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)
    assert any("empty translation" in str(w.value) for w in at.warning)
    assert at.session_state["translate_notice"] == ""

    _rerun_with_mocks(at)

    assert not at.warning
    assert not at.error


def test_end_response_only_stream_shows_warning() -> None:
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch(
            "mlx_lm.stream_generate",
            return_value=iter([_make_stream_chunk("<|END_RESPONSE|>")]),
        ),
    ):
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

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


# -- Docked controls -----------------------------------------------------------


def test_controls_row_lives_in_the_bottom_bar(app: AppTest) -> None:
    # Translate and Download render in st.bottom, the one native lever that
    # drops the main block's 160px bottom padding and pins the pair to the
    # viewport at every height. AppTest has no `bottom` accessor in 1.63.0
    # (`main` is root 0, `sidebar` root 1), so the bar is read by root index
    # -- the element tree creates it on demand, so a page that never touches
    # st.bottom fails on the KeyError. Pinned positively: the bar holds
    # exactly Translate and one Download, and main holds only the swap
    # button. Either button moved back into the flow (or into the sidebar)
    # fails its bar line; a stray copy in main fails the main line. The
    # first line is the harness guard: the day AppTest grows a public
    # `bottom`, this fails with that message rather than as an opaque
    # AttributeError on the private tree, and the read below moves to it.
    assert not hasattr(app, "bottom"), "AppTest has a bottom accessor now; use it"
    bottom = app._tree[RootContainer.BOTTOM]
    assert isinstance(bottom, Block)
    assert [b.key for b in bottom.button] == ["translate"]
    assert len(bottom.get("download_button")) == 1
    assert [b.key for b in app.main.button] == ["swap"]
    assert not app.main.get("download_button")


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
        at = _new_app()
        _run(at)

    load.assert_not_called()
    assert len(at.selectbox) == 2
    assert len(at.text_area) == 2
    assert not at.error


def test_model_load_failure_shows_error_on_translate() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_translate_button_stays_enabled_after_a_failed_load() -> None:
    # Click first, so the load actually fails before the assertion -- asserting
    # on a fresh render would only re-test the initial state, since nothing
    # loads the model there.
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

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

    app.selectbox("target_lang").set_value("German")
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
        at.text_area("translate_input").set_value(text)
        at.button("translate").click()
        _run(at)
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


def test_previous_translation_is_not_downloadable_while_the_next_streams() -> None:
    # The bar renders before the translate block, from translate_output and
    # download_name as they stand at the top of the run. Translate's on_click
    # clears both before the script runs, so the run that streams paints
    # Download disabled with no bytes; before that, the previous translation
    # stayed downloadable -- under the previous target's name -- for the
    # whole stream, while the skeleton had already cleared it from the panel.
    # The two end-state tests above cannot see the window, so it is read from
    # inside the stream, on the script thread, after the bar has rendered.
    seen: dict[str, str] = {}

    def record_then_stream(*_args: object, **_kwargs: object):
        seen["output"] = st.session_state["translate_output"]
        seen["name"] = st.session_state["download_name"]
        yield _make_stream_chunk("Au revoir")

    at = _run_inference_test(input_text="hello world", chunk_text="Bonjour le monde")
    assert at.session_state["download_name"] == "translation-French.txt"

    _translate_with(at, "goodbye", side_effect=record_then_stream)

    assert seen == {"output": "", "name": "translation.txt"}
    assert at.session_state["translate_output"] == "Au revoir"
    assert not at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]


def test_failure_after_partial_output_keeps_the_partial() -> None:
    def stream_then_die(*_args: object, **_kwargs: object):
        yield _make_stream_chunk("Bonjour le ")
        raise RuntimeError("OOM")

    at = _new_app()
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        _run(at)
    _translate_with(at, "hello world", side_effect=stream_then_die)

    assert any("after partial output" in str(e.value) for e in at.error)
    assert at.session_state["translate_output"] == "Bonjour le"
    assert not at.get("download_button")[0].disabled  # ty: ignore[unresolved-attribute]
    assert at.session_state["download_name"] == "translation-French.txt"


def test_first_failure_keeps_the_empty_state_panel() -> None:
    # An earlier fix painted a blank st.code here, leaving a featureless grey
    # rectangle where the "Translation appears here" panel had been.
    at = _new_app()
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        _run(at)
    _translate_with(at, "hello", side_effect=RuntimeError("OOM"))

    assert not at.get("code")
    assert len(at.text_area) == 2


# -- Exception text in alerts --------------------------------------------------

# Shaped like a real Python message: unescaped, the dunder renders as bold,
# the tildes as strikethrough, the brackets as a link and the backticks as
# code. Each failure path has its own f-string, so each is pinned through
# its own sink; the escaping itself is unit-tested in test_streamlit_app.py.
_HOSTILE_MESSAGE = "Model.__init__() ~missing~ [1](x) `arg`"


def _the_error_starting(at: AppTest, prefix: str) -> str:
    (value,) = [str(e.value) for e in at.error if str(e.value).startswith(prefix)]
    return value


def test_load_failure_message_is_escaped_for_markdown() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError(_HOSTILE_MESSAGE)):
        at = _new_app()
        _run(at)
        at.text_area("translate_input").set_value("Hello")
        at.button("translate").click()
        _run(at)

    value = _the_error_starting(at, "Failed to load model: ")
    assert escape_markdown(_HOSTILE_MESSAGE) in value
    assert _HOSTILE_MESSAGE not in value


def test_stream_failure_message_is_escaped_for_markdown() -> None:
    at = _new_app()
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        _run(at)
    _translate_with(at, "hello", side_effect=RuntimeError(_HOSTILE_MESSAGE))

    value = _the_error_starting(at, "Translation failed: ")
    assert escape_markdown(_HOSTILE_MESSAGE) in value
    assert _HOSTILE_MESSAGE not in value


def test_partial_failure_message_is_escaped_for_markdown() -> None:
    def stream_then_die(*_args: object, **_kwargs: object):
        yield _make_stream_chunk("Bonjour le ")
        raise RuntimeError(_HOSTILE_MESSAGE)

    at = _new_app()
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        _run(at)
    _translate_with(at, "hello world", side_effect=stream_then_die)

    value = _the_error_starting(at, "Translation failed after partial output: ")
    assert escape_markdown(_HOSTILE_MESSAGE) in value
    assert _HOSTILE_MESSAGE not in value
