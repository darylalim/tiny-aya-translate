# SPDX-License-Identifier: Apache-2.0
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear both Streamlit caches between tests.

    Both are process-global and outlive an AppTest instance. cache_data
    matters for any test that drives cached_document_chunks: its key is
    (file bytes, source language), and the fixtures reuse the same fake
    bytes, so a stale entry would let a later test skip its own LiteParse
    patch and pass or fail for a reason unrelated to what it asserts.
    """
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def app() -> AppTest:
    """Create an AppTest instance that has completed its initial render.

    The ``mlx_lm.load`` patch is belt-and-braces: the model loads lazily from
    the translate handlers, so an initial render never reaches it (see
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


# -- Tabs ----------------------------------------------------------------------


def test_tabs_labelled_text_and_document_with_icons(app: AppTest) -> None:
    assert [t.label for t in app.tabs] == [
        ":material/text_fields: Text",
        ":material/description: Document",
    ]


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
    # Three at rest: the Text tab's input and empty output, plus the
    # Document tab's empty output panel.
    assert len(app.text_area) == 3


def test_output_text_area_placeholder(app: AppTest) -> None:
    # An instructional phrase, not a bare noun: the settled output is painted
    # in the same muted grey, so a one-word translation and a one-word
    # placeholder would be indistinguishable.
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
    # The Text tab's output panel is no longer a text_area; its input and
    # the Document tab's empty output panel remain.
    assert len(at.text_area) == 2


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
    # One in the Text tab, one in the Document tab.
    assert len(app.get("download_button")) == 2


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

    The model is loaded from the translate handlers, not at page load, so a
    broken install still gets a full page -- tabs, language pickers, and a
    usable input -- with no error until the user actually asks to translate.
    """
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")) as load:
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    load.assert_not_called()
    assert len(at.tabs) == 2
    assert len(at.text_area) == 3
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


# -- Document tab --------------------------------------------------------------


def test_document_translate_button_exists(app: AppTest) -> None:
    assert app.button("translate_doc") is not None


def test_document_translate_button_disabled_without_upload(app: AppTest) -> None:
    assert app.button("translate_doc").disabled


def test_document_download_button_disabled_when_no_output(app: AppTest) -> None:
    # The second download button belongs to the Document tab.
    assert app.get("download_button")[1].disabled  # ty: ignore[unresolved-attribute]


def test_document_tab_needs_no_install_hint(app: AppTest) -> None:
    # liteparse is a required dependency now, so the tab is never gated behind
    # an extra and there is nothing for the user to install.
    assert not any("uv sync" in str(i.value) for i in app.info)


def test_document_uploader_offers_no_parser_choice(app: AppTest) -> None:
    # One parser, so no backend radio -- the Text tab's swap button and the two
    # language selectboxes per tab are the only widgets of their kind.
    assert not [r for r in app.radio]


def test_document_uploader_caps_upload_size(app: AppTest) -> None:
    # Streamlit's server default is 200 MB, and a file that size reaches an
    # in-process parser with the weights already resident. The cap is a widget
    # parameter, so it shows up on the uploader's own proto.
    size = app.get("file_uploader")[0].max_upload_size_mb  # ty: ignore[unresolved-attribute]
    assert 0 < size < 200


def test_new_upload_clears_the_previous_translation(app: AppTest) -> None:
    # doc_output outlives the upload it came from. Without the on_change hook a
    # new file rendered the *previous* file's translation, and offered it for
    # download under the new filename.
    app.session_state["doc_output"] = "PREVIOUS FILE TRANSLATION"
    _rerun_with_mocks(app)
    assert len(app.get("code")) == 1
    assert not app.get("download_button")[1].disabled  # ty: ignore[unresolved-attribute]

    app.get("file_uploader")[0].upload("second.pdf", b"%PDF-1.4 fake")  # ty: ignore[unresolved-attribute]
    _rerun_with_mocks(app)

    assert app.session_state["doc_output"] == ""
    assert not app.get("code")
    assert app.get("download_button")[1].disabled  # ty: ignore[unresolved-attribute]


def test_removing_the_upload_also_clears_the_translation(app: AppTest) -> None:
    # Removing the file is the same invalidation as replacing it: the output
    # panel must not keep serving a translation of a document that is no longer
    # loaded. Upload first -- clearing a never-populated uploader is a no-op and
    # fires no change event.
    app.get("file_uploader")[0].upload("first.pdf", b"%PDF-1.4 fake")  # ty: ignore[unresolved-attribute]
    _rerun_with_mocks(app)
    app.session_state["doc_output"] = "FIRST FILE TRANSLATION"
    _rerun_with_mocks(app)
    assert len(app.get("code")) == 1

    app.get("file_uploader")[0].clear()  # ty: ignore[unresolved-attribute]
    _rerun_with_mocks(app)

    assert app.session_state["doc_output"] == ""
    assert not app.get("code")


# -- Download buttons ----------------------------------------------------------


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


def test_document_swap_button_exists(app: AppTest) -> None:
    assert app.button("swap_doc") is not None


def test_document_swap_flips_languages(app: AppTest) -> None:
    # The Document selectboxes follow the Text tab's two.
    app.button("swap_doc").click()
    _rerun_with_mocks(app)

    assert app.selectbox[2].value == "French"
    assert app.selectbox[3].value == "English"


def test_document_swap_leaves_the_output_alone(app: AppTest) -> None:
    # Unlike the Text tab's swap there is no input to move the output into, and
    # the upload has not changed -- clear_doc_output owns that invalidation.
    app.session_state["doc_output"] = "TRANSLATED"
    app.button("swap_doc").click()
    _rerun_with_mocks(app)

    assert app.session_state["doc_output"] == "TRANSLATED"


def test_new_upload_resets_the_download_filename(app: AppTest) -> None:
    app.session_state["doc_download_name"] = "report-French.md"
    app.session_state["doc_meta"] = "report.pdf · English → French"
    app.get("file_uploader")[0].upload("second.pdf", b"%PDF-1.4 fake")  # ty: ignore[unresolved-attribute]
    _rerun_with_mocks(app)

    assert app.session_state["doc_download_name"] == "translation.md"
    assert app.session_state["doc_meta"] == ""


def test_download_buttons_do_not_trigger_a_rerun(app: AppTest) -> None:
    # on_click defaults to "rerun", but a download changes no server state --
    # so every click re-executed both tab bodies to rebuild an identical page.
    for button in app.get("download_button"):
        assert button.ignore_rerun  # ty: ignore[unresolved-attribute]


def test_document_ocr_failure_shows_the_error_not_the_blank_guard() -> None:
    # Only images run OCR, and an image has no text layer to fall back to.
    # With ocr_failure_fatal=False an uncached, unfetchable traineddata came
    # back as the empty fence, and the tab said "No translatable text found"
    # about a scan it had never read. Left fatal, the real error is reported
    # -- as a read failure, not a translation failure: no translation ran.
    from liteparse import ParseError

    def fake_liteparse(**kwargs: Any) -> MagicMock:
        # Mirror LiteParse: non-fatal "continues with partial (native-text)
        # results", of which an image has none, so the parse returns the
        # empty fence; fatal raises.
        parser = MagicMock()
        if kwargs.get("ocr_failure_fatal") is False:
            parser.parse.return_value.text = "```text\n\n```"
        else:
            parser.parse.side_effect = ParseError(
                "OCR failed: error sending request for url (.../eng.traineddata)"
            )
        return parser

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("liteparse.LiteParse", side_effect=fake_liteparse),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.get("file_uploader")[0].upload("scan.png", b"\x89PNG fake")  # ty: ignore[unresolved-attribute]
        at.run(timeout=60)
        at.button("translate_doc").click()
        at.run(timeout=60)

    errors = [str(e.value) for e in at.error]
    assert any("Could not read the document: OCR failed" in v for v in errors)
    assert not any("Translation failed" in v for v in errors)
    assert not any("No translatable text" in str(w.value) for w in at.warning)
    assert at.session_state["doc_output"] == ""
    assert at.get("download_button")[1].disabled  # ty: ignore[unresolved-attribute]


def _translate_document_that_parses_to(name: str, data: bytes, text: str) -> AppTest:
    """Upload ``data`` as ``name``, let LiteParse return ``text``, click Translate."""
    parser = MagicMock()
    parser.parse.return_value.text = text
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("liteparse.LiteParse", return_value=parser),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
        at.get("file_uploader")[0].upload(name, data)  # ty: ignore[unresolved-attribute]
        at.run(timeout=60)
        at.button("translate_doc").click()
        at.run(timeout=60)
    return at


def test_scanned_pdf_is_reported_as_having_no_text_layer() -> None:
    # PDFs never run OCR (the offline policy), so a scan parses to the empty
    # fence and used to be reported as blank. It is not blank, and "no
    # translatable text" sent people looking for a fault in the file.
    at = _translate_document_that_parses_to(
        "scan.pdf", b"%PDF-1.4 fake", "```text\n\n```"
    )

    warnings = [str(w.value) for w in at.warning]
    assert any("no text layer" in v for v in warnings)
    assert not any("No translatable text" in v for v in warnings)
    assert at.session_state["doc_output"] == ""


def test_blank_image_is_reported_as_no_translatable_text() -> None:
    # The same fence from an image means OCR ran and found nothing, which is
    # what "no translatable text" was always meant to say.
    at = _translate_document_that_parses_to(
        "blank.png", b"\x89PNG fake", "```text\n\n```"
    )

    warnings = [str(w.value) for w in at.warning]
    assert any("No translatable text" in v for v in warnings)
    assert not any("no text layer" in v for v in warnings)


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
    assert len(at.text_area) == 3


def test_text_translation_does_not_reset_the_document_languages() -> None:
    # The Text tab's rerun is deferred to the bottom of the script. Taken in
    # place it aborted before `with doc_tab:` registered its widgets, and
    # Streamlit's stale-widget purge then dropped them -- so the setdefault
    # block silently re-seeded the Document pickers to English/French and the
    # next document translation ran the wrong pair.
    streams: list[dict[str, Any]] = [
        {"return_value": iter([_make_stream_chunk("Bonjour")])},
        {"return_value": iter([])},
        {"side_effect": RuntimeError("OOM")},
    ]
    for stream in streams:
        at = AppTest.from_file("streamlit_app.py")
        with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
            at.run(timeout=60)
            at.selectbox[2].set_value("Japanese")
            at.selectbox[3].set_value("German")
            at.run(timeout=60)

        _translate_with(at, "hello world", **stream)

        assert at.selectbox[2].value == "Japanese", stream
        assert at.selectbox[3].value == "German", stream
