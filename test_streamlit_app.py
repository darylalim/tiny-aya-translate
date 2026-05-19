from unittest.mock import MagicMock, patch

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    build_translation_prompt,
    clean_model_output,
    stream_translate,
    tokenize_prompt,
)

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

    mock_make_sampler.assert_called_once_with(temp=0.3, top_p=streamlit_app.TOP_P)
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

    mock_make_sampler.assert_called_once_with(
        temp=streamlit_app.DEFAULT_TEMPERATURE, top_p=streamlit_app.TOP_P
    )
    assert (
        mock_stream_generate.call_args.kwargs["max_tokens"]
        == streamlit_app.DEFAULT_MAX_TOKENS
    )
