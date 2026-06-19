import os
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    build_translation_prompt,
    chunk_document,
    clean_model_output,
    docling_available,
    load_document,
    stream_translate,
    tokenize_prompt,
    translate_document,
)

# -- module configuration ------------------------------------------------------


def test_transformers_verbosity_is_set() -> None:
    # setdefault preserves an existing override, so assert only that it's set.
    assert os.environ.get("TRANSFORMERS_VERBOSITY")


# -- streamlit_app.py width API ------------------------------------------------

_APP_SOURCE = (Path(__file__).parent / "streamlit_app.py").read_text(encoding="utf-8")


def test_no_deprecated_use_container_width() -> None:
    # use_container_width is deprecated in Streamlit 1.58; guard against
    # reintroducing it after the migration to the width API.
    assert "use_container_width" not in _APP_SOURCE


def test_buttons_use_width_stretch() -> None:
    # The five full-width controls (swap, translate, download, translate_doc,
    # download_doc) set width="stretch".
    assert _APP_SOURCE.count('width="stretch"') == 5


# -- .streamlit/config.toml theme ----------------------------------------------

_CONFIG_PATH = Path(__file__).parent / ".streamlit" / "config.toml"


def _load_theme_config() -> dict[str, Any]:
    with _CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def test_theme_config_exists() -> None:
    assert _CONFIG_PATH.is_file()


def test_theme_config_has_theme_section() -> None:
    assert "theme" in _load_theme_config()


# -- LANGUAGES -----------------------------------------------------------------


def test_languages_list_has_67_entries() -> None:
    assert len(LANGUAGES) == 67


def test_languages_list_contains_english() -> None:
    assert "English" in LANGUAGES


def test_languages_list_contains_japanese() -> None:
    assert "Japanese" in LANGUAGES


# -- build_translation_prompt --------------------------------------------------


def test_build_translation_prompt_returns_single_message() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_build_translation_prompt_contains_languages() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    content = result[0]["content"]
    assert "English" in content
    assert "French" in content


def test_build_translation_prompt_contains_text() -> None:
    result = build_translation_prompt("Good morning", "English", "Spanish")
    content = result[0]["content"]
    assert "Good morning" in content


def test_build_translation_prompt_instruction() -> None:
    result = build_translation_prompt("Hello", "English", "French")
    content = result[0]["content"]
    assert "Translate" in content
    assert "Output only the translation" in content


# -- tokenize_prompt -----------------------------------------------------------


def test_tokenize_prompt_returns_token_ids() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]

    assert tokenize_prompt("Hello", "English", "French", mock_tokenizer) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_tokenize_prompt_calls_apply_chat_template_with_tokenize_true() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]

    tokenize_prompt("Hello", "English", "French", mock_tokenizer)

    call_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs["tokenize"] is True
    assert call_kwargs["add_generation_prompt"] is True


def test_tokenize_prompt_uses_translation_prompt() -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1]

    tokenize_prompt("Good morning", "English", "Spanish", mock_tokenizer)

    messages = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert "English" in content
    assert "Spanish" in content
    assert "Good morning" in content


# -- clean_model_output --------------------------------------------------------


def test_clean_model_output_strips_whitespace() -> None:
    assert clean_model_output("  Hello world  ") == "Hello world"


def test_clean_model_output_empty_string() -> None:
    assert clean_model_output("") == ""


def test_clean_model_output_newlines() -> None:
    assert clean_model_output("\n\nBonjour\n\n") == "Bonjour"


def test_clean_model_output_preserves_inner_whitespace() -> None:
    assert clean_model_output("  Hello   world  ") == "Hello   world"


def test_clean_model_output_strips_end_response_token() -> None:
    assert clean_model_output("Bonjour le monde<|END_RESPONSE|>") == "Bonjour le monde"


def test_clean_model_output_strips_end_response_token_with_whitespace() -> None:
    assert (
        clean_model_output("  Bonjour le monde  <|END_RESPONSE|>  ")
        == "Bonjour le monde"
    )


# -- stream_translate ----------------------------------------------------------


def _make_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    return chunk


@patch("mlx_lm.stream_generate")
def test_stream_translate_yields_cleaned_partials(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.return_value = iter([_make_chunk("Bon"), _make_chunk("jour")])

    results = list(
        stream_translate(
            prompt_ids=[1, 2, 3],
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
    )
    assert results == ["Bon", "Bonjour"]


@patch("mlx_lm.stream_generate")
def test_stream_translate_handles_empty_stream(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.return_value = iter([])

    results = list(
        stream_translate(
            prompt_ids=[1, 2, 3],
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
    )
    assert results == []


@patch("mlx_lm.stream_generate")
def test_stream_translate_strips_end_response_token_mid_stream(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.return_value = iter(
        [_make_chunk("Bonjour"), _make_chunk("<|END_RESPONSE|>")]
    )

    results = list(
        stream_translate(
            prompt_ids=[1, 2, 3],
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
    )
    assert results == ["Bonjour", "Bonjour"]


@patch("mlx_lm.stream_generate")
@patch("mlx_lm.sample_utils.make_sampler")
def test_stream_translate_calls_stream_generate_with_correct_params(
    mock_make_sampler: MagicMock,
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.return_value = iter([_make_chunk("Bonjour")])
    mock_make_sampler.return_value = MagicMock()

    list(
        stream_translate(
            prompt_ids=[1, 2, 3, 4, 5],
            model=MagicMock(),
            tokenizer=MagicMock(),
            temperature=0.3,
            max_tokens=500,
        )
    )

    mock_make_sampler.assert_called_once_with(temp=0.3)
    mock_stream_generate.assert_called_once()
    call_kwargs = mock_stream_generate.call_args.kwargs
    assert call_kwargs["prompt"] == [1, 2, 3, 4, 5]
    assert call_kwargs["max_tokens"] == 500
    assert call_kwargs["sampler"] is mock_make_sampler.return_value


@patch("mlx_lm.stream_generate")
def test_stream_translate_does_not_retokenize(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.return_value = iter([_make_chunk("Bonjour")])
    mock_tokenizer = MagicMock()

    list(
        stream_translate(
            prompt_ids=[1, 2, 3],
            model=MagicMock(),
            tokenizer=mock_tokenizer,
        )
    )

    mock_tokenizer.apply_chat_template.assert_not_called()


@patch("mlx_lm.stream_generate")
@patch("mlx_lm.sample_utils.make_sampler")
def test_stream_translate_uses_default_params(
    mock_make_sampler: MagicMock,
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.return_value = iter([_make_chunk("Bonjour")])
    mock_make_sampler.return_value = MagicMock()

    list(
        stream_translate(
            prompt_ids=[1, 2, 3],
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
    )

    mock_make_sampler.assert_called_once_with(temp=streamlit_app.DEFAULT_TEMPERATURE)
    assert (
        mock_stream_generate.call_args.kwargs["max_tokens"]
        == streamlit_app.DEFAULT_MAX_TOKENS
    )


# -- docling_available ---------------------------------------------------------


@patch("importlib.util.find_spec")
def test_docling_available_true_when_spec_found(mock_find_spec: MagicMock) -> None:
    mock_find_spec.return_value = object()
    assert docling_available() is True


@patch("importlib.util.find_spec")
def test_docling_available_false_when_spec_missing(
    mock_find_spec: MagicMock,
) -> None:
    mock_find_spec.return_value = None
    assert docling_available() is False


# -- load_document -------------------------------------------------------------


@patch("docling.document_converter.DocumentConverter")
def test_load_document_returns_converted_document(
    mock_converter_cls: MagicMock,
) -> None:
    expected_doc = MagicMock()
    mock_converter_cls.return_value.convert.return_value.document = expected_doc

    assert load_document(b"file bytes", "sample.pdf") is expected_doc


@patch("docling.document_converter.DocumentConverter")
def test_load_document_builds_stream_with_filename(
    mock_converter_cls: MagicMock,
) -> None:
    load_document(b"data", "report.docx")

    source = mock_converter_cls.return_value.convert.call_args[0][0]
    assert source.name == "report.docx"


@patch("docling.document_converter.DocumentConverter")
def test_load_document_passes_file_bytes_to_stream(
    mock_converter_cls: MagicMock,
) -> None:
    load_document(b"hello bytes", "notes.html")

    source = mock_converter_cls.return_value.convert.call_args[0][0]
    assert source.stream.getvalue() == b"hello bytes"


# -- chunk_document ------------------------------------------------------------


@patch("docling.chunking.HybridChunker")
@patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
def test_chunk_document_returns_contextualized_strings(
    mock_hf_tokenizer: MagicMock, mock_hybrid_chunker_cls: MagicMock
) -> None:
    chunker = mock_hybrid_chunker_cls.return_value
    chunker.chunk.return_value = ["raw_a", "raw_b"]
    chunker.contextualize.side_effect = ["context A", "context B"]

    result = chunk_document(MagicMock(), MagicMock())

    assert result == ["context A", "context B"]


@patch("docling.chunking.HybridChunker")
@patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
def test_chunk_document_handles_empty_document(
    mock_hf_tokenizer: MagicMock, mock_hybrid_chunker_cls: MagicMock
) -> None:
    mock_hybrid_chunker_cls.return_value.chunk.return_value = []

    assert chunk_document(MagicMock(), MagicMock()) == []


@patch("docling.chunking.HybridChunker")
@patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
def test_chunk_document_passes_max_tokens_to_tokenizer(
    mock_hf_tokenizer: MagicMock, mock_hybrid_chunker_cls: MagicMock
) -> None:
    mock_hybrid_chunker_cls.return_value.chunk.return_value = []

    chunk_document(MagicMock(), MagicMock(), max_tokens=1234)

    assert mock_hf_tokenizer.call_args.kwargs["max_tokens"] == 1234


@patch("docling.chunking.HybridChunker")
@patch("docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer")
def test_chunk_document_defaults_to_max_chunk_tokens(
    mock_hf_tokenizer: MagicMock, mock_hybrid_chunker_cls: MagicMock
) -> None:
    mock_hybrid_chunker_cls.return_value.chunk.return_value = []

    chunk_document(MagicMock(), MagicMock())

    assert (
        mock_hf_tokenizer.call_args.kwargs["max_tokens"]
        == streamlit_app.MAX_CHUNK_TOKENS
    )


# -- translate_document --------------------------------------------------------


@patch("mlx_lm.stream_generate")
def test_translate_document_yields_cumulative_per_chunk(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.side_effect = [
        iter([_make_chunk("Bon"), _make_chunk("jour")]),
        iter([_make_chunk("Salut")]),
    ]

    results = list(
        translate_document(
            ["chunk one", "chunk two"],
            "English",
            "French",
            MagicMock(),
            MagicMock(),
        )
    )

    assert results == [
        (0, "Bon"),
        (0, "Bonjour"),
        (1, "Bonjour\n\nSalut"),
    ]


@patch("mlx_lm.stream_generate")
def test_translate_document_handles_empty_chunk_list(
    mock_stream_generate: MagicMock,
) -> None:
    results = list(
        translate_document([], "English", "French", MagicMock(), MagicMock())
    )

    assert results == []
    mock_stream_generate.assert_not_called()


@patch("mlx_lm.stream_generate")
def test_translate_document_tokenizes_each_chunk(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.side_effect = [
        iter([_make_chunk("a")]),
        iter([_make_chunk("b")]),
    ]
    mock_tokenizer = MagicMock()

    list(
        translate_document(
            ["one", "two"], "English", "French", MagicMock(), mock_tokenizer
        )
    )

    assert mock_tokenizer.apply_chat_template.call_count == 2


@patch("mlx_lm.stream_generate")
def test_translate_document_skips_chunk_over_token_limit(
    mock_stream_generate: MagicMock,
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = list(range(9000))

    results = list(
        translate_document(
            ["a very long chunk"],
            "English",
            "French",
            MagicMock(),
            mock_tokenizer,
        )
    )

    assert len(results) == 1
    idx, text = results[0]
    assert idx == 0
    assert "skipped" in text.lower()
    mock_stream_generate.assert_not_called()


@patch("mlx_lm.stream_generate")
def test_translate_document_omits_blank_chunk_output(
    mock_stream_generate: MagicMock,
) -> None:
    mock_stream_generate.side_effect = [
        iter([]),
        iter([_make_chunk("Salut")]),
    ]

    results = list(
        translate_document(
            ["one", "two"], "English", "French", MagicMock(), MagicMock()
        )
    )

    assert results == [(1, "Salut")]
