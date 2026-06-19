from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def clear_st_cache() -> None:
    """Clear Streamlit's @st.cache_resource between tests."""
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    """Create a patched AppTest instance with mocked model loading."""
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)
    return at


def _rerun_with_mocks(app: AppTest) -> None:
    """Re-run the app with mocked model loading."""
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        app.run(timeout=60)


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


# -- Caption -------------------------------------------------------------------


def test_caption_mentions_cohere_labs_model(app: AppTest) -> None:
    caption_texts = [str(c.value) for c in app.caption]
    assert any("Cohere Labs Tiny Aya Global model" in t for t in caption_texts)


def test_caption_links_to_huggingface(app: AppTest) -> None:
    caption_texts = [str(c.value) for c in app.caption]
    assert any("huggingface.co/CohereLabs/tiny-aya-global" in t for t in caption_texts)


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


def test_input_text_area_has_no_placeholder(app: AppTest) -> None:
    assert app.text_area[0].placeholder == ""


def test_output_uses_text_area(app: AppTest) -> None:
    assert len(app.text_area) == 2


def test_output_text_area_placeholder(app: AppTest) -> None:
    assert app.text_area[1].placeholder == "Translation"


# -- Translate flow ------------------------------------------------------------


def test_translate_button_exists(app: AppTest) -> None:
    assert app.button("translate") is not None


def test_translate_button_enabled_when_model_loaded(app: AppTest) -> None:
    assert not app.button("translate").disabled


def test_translate_success_shows_result() -> None:
    at = _run_inference_test(input_text="Hello", chunk_text="Bonjour")
    assert at.text_area[1].value == "Bonjour"


def test_translate_empty_text_shows_warning(app: AppTest) -> None:
    app.button("translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("Please enter some text first" in str(v) for v in warning_values)


def test_translate_same_language_shows_warning(app: AppTest) -> None:
    app.selectbox[1].set_value("English")
    app.text_area[0].set_value("Hello")
    app.button("translate").click()
    _rerun_with_mocks(app)

    warning_values = [w.value for w in app.warning]
    assert any("two different languages" in str(v) for v in warning_values)


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

    assert at.text_area[1].value == "OK"
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
    assert any("no output" in v for v in warning_values)


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
    assert any("no output" in v for v in warning_values)


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


def test_output_text_area_disabled(app: AppTest) -> None:
    assert app.text_area[1].disabled


# -- Model load failure --------------------------------------------------------


def test_model_load_failure_shows_error() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    error_values = [e.value for e in at.error]
    assert any("Failed to load model" in str(v) for v in error_values)


def test_model_load_failure_disables_translate_button() -> None:
    with patch("mlx_lm.load", side_effect=RuntimeError("download failed")):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    assert at.button("translate").disabled


# -- Document tab --------------------------------------------------------------


def test_document_translate_button_exists(app: AppTest) -> None:
    assert app.button("translate_doc") is not None


def test_document_translate_button_disabled_without_upload(app: AppTest) -> None:
    assert app.button("translate_doc").disabled


def test_document_download_button_disabled_when_no_output(app: AppTest) -> None:
    # The second download button belongs to the Document tab.
    assert app.get("download_button")[1].disabled  # ty: ignore[unresolved-attribute]


def test_document_tab_install_hint_when_docling_missing() -> None:
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name == "docling":
            return None
        return real_find_spec(name, package)

    with (
        patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
    ):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=60)

    info_values = [str(i.value) for i in at.info]
    assert any("uv sync --extra docs" in v for v in info_values)
