# SPDX-License-Identifier: Apache-2.0
import os
import re
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from streamlit.commands.page_config import _get_favicon_string

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    OCR_LANGUAGES,
    build_translation_prompt,
    chunk_text,
    clean_model_output,
    count_tokens,
    document_download_name,
    document_meta_line,
    escape_markdown,
    hard_windows,
    heading_level,
    heading_line,
    is_blank_markdown,
    is_pdf,
    leading_headings,
    load_document_markdown,
    pack_by_estimate,
    render_output,
    sentence_units,
    split_oversized,
    stream_translate,
    token_overhead,
    tokenize_prompt,
    translate_document,
)

_APP_SOURCE = (Path(__file__).parent / "streamlit_app.py").read_text(encoding="utf-8")

# -- module configuration ------------------------------------------------------


def test_transformers_verbosity_is_set() -> None:
    # setdefault preserves an existing override, so assert only that it's set.
    assert os.environ.get("TRANSFORMERS_VERBOSITY")


def test_hub_defaults_are_set_before_anything_can_import_the_hub() -> None:
    # mlx_lm.load resolves MODEL_ID's `main` with a GET to huggingface.co on
    # every server process's first Translate; these two defaults keep a cached
    # `hf auth login` token and the agent-harness telemetry out of it.
    # huggingface_hub reads both once, at import, so the check is static: each
    # line is pinned verbatim, sits above `import streamlit` (the first import
    # that could conceivably pull the hub in), and no module-level import of
    # mlx_lm, transformers or huggingface_hub exists anywhere -- every such
    # import is deferred into a function body. Deliberately not asserted
    # through os.environ or huggingface_hub.constants: a shell that exports
    # either flag would hide the lines' removal, and the constants freeze at
    # whatever pytest's collection order happened to import first.
    streamlit_import = _APP_SOURCE.index("import streamlit as st")
    for flag in ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_HUB_DISABLE_TELEMETRY"):
        line = f'os.environ.setdefault("{flag}", "1")'
        assert _APP_SOURCE.count(line) == 1, flag
        assert _APP_SOURCE.index(line) < streamlit_import, flag
    hub_pulling = re.compile(
        r"^(?:from|import) (?:mlx_lm|transformers|huggingface_hub)\b", re.M
    )
    assert not hub_pulling.search(_APP_SOURCE)


# -- streamlit_app.py width API ------------------------------------------------


def test_no_deprecated_use_container_width() -> None:
    # use_container_width is deprecated in Streamlit 1.58; guard against
    # reintroducing it after the migration to the width API.
    assert "use_container_width" not in _APP_SOURCE


def test_buttons_use_width_stretch() -> None:
    # The four full-width controls (translate, download, translate_doc,
    # download_doc) set width="stretch". The two swap buttons are deliberately
    # NOT among them: stretched inside a 1-unit column they became invisible
    # full-row tap targets below the 640px breakpoint, so both are pinned to a
    # fixed width and centred instead.
    assert _APP_SOURCE.count('width="stretch"') == 4
    assert _APP_SOURCE.count("width=40,") == 2


# -- streamlit_app.py shared UI constants --------------------------------------


def test_panel_height_not_hardcoded() -> None:
    # The 450px panel height now lives in PANEL_HEIGHT; guard against re-inlining
    # the magic number at a call site (text_area panels or render_output).
    assert "height=450" not in _APP_SOURCE


def test_warning_strings_defined_once() -> None:
    # Each shared warning lives in exactly one place — its module-level constant.
    # The Text- and Document-tab call sites reference SAME_LANGUAGE_WARNING /
    # NO_OUTPUT_WARNING, so neither raw literal should be duplicated.
    assert _APP_SOURCE.count('"Please pick two different languages."') == 1
    assert _APP_SOURCE.count("empty translation. Try again") == 1


# -- streamlit_app.py page config ----------------------------------------------


def test_set_page_config_sets_title_icon_and_wide_layout() -> None:
    # set_page_config must be the first Streamlit command; it defines the
    # browser-tab title, favicon, and the wide layout for the side-by-side panels.
    # The favicon is a local SVG, deliberately not a `:material/…:` name: that
    # form is fetched from fonts.gstatic.com on every page load, the one
    # outbound request the README's privacy promise would not cover. The path
    # is pinned here and the file's existence below.
    assert "st.set_page_config(" in _APP_SOURCE
    assert 'page_title="Tiny Aya Translate"' in _APP_SOURCE
    assert "page_icon=FAVICON_PATH" in _APP_SOURCE
    assert 'layout="wide"' in _APP_SOURCE
    assert streamlit_app.FAVICON_PATH.endswith(os.path.join("assets", "favicon.svg"))
    # Ask Streamlit's own resolver rather than sniffing the file: it inlines a
    # readable SVG as a data: URL and hands back the raw path, silently, for a
    # missing or malformed one -- so this one assertion covers the fresh-clone
    # case, the SVG regex Streamlit applies, and that no Material name (which
    # would resolve to a fonts.gstatic.com URL) is in play.
    favicon = _get_favicon_string(streamlit_app.FAVICON_PATH)
    assert favicon.startswith("data:image/svg+xml;base64,"), favicon[:60]


# -- .streamlit/config.toml theme ----------------------------------------------

_CONFIG_PATH = Path(__file__).parent / ".streamlit" / "config.toml"


def _load_theme_config() -> dict[str, Any]:
    with _CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def test_theme_config_exists() -> None:
    assert _CONFIG_PATH.is_file()


def test_theme_config_has_theme_section() -> None:
    assert "theme" in _load_theme_config()


def test_theme_config_defines_light_and_dark_modes() -> None:
    # At least one variant must exist for Streamlit to show the light/dark
    # switch -- a lone [theme] section hides it and locks the app to a single
    # mode; this project states both.
    theme = _load_theme_config()["theme"]
    assert "light" in theme
    assert "dark" in theme


def _hex6(color: str) -> str:
    # Streamlit accepts any CSS colour for a theme key, but this file only ever
    # writes hex; expand the #rgb short form and lowercase so a valid spelling
    # cannot crash a ratio test or fail an equality guard on case alone. Names
    # like "teal" are rejected up front rather than mangled by the expansion.
    color = color.strip().lower()
    assert color.startswith("#"), f"not a hex colour: {color}"
    if len(color) == 4:
        color = "#" + "".join(c * 2 for c in color[1:])
    assert re.fullmatch(r"#[0-9a-f]{6}", color), f"not a hex colour: {color}"
    return color


def _relative_luminance(hex_color: str) -> float:
    hex_color = _hex6(hex_color)
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast_ratio(hex_a: str, hex_b: str) -> float:
    lum_a, lum_b = _relative_luminance(hex_a), _relative_luminance(hex_b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _composite(fg: str, bg: str, alpha: float) -> str:
    # Streamlit derives several text tokens by painting textColor at a fixed
    # alpha over a background (caption 0.6, fadedText60/40); this is the colour
    # the browser actually shows for them.
    fg, bg = _hex6(fg), _hex6(bg)
    channels = (
        round(alpha * int(fg[i : i + 2], 16) + (1 - alpha) * int(bg[i : i + 2], 16))
        for i in (1, 3, 5)
    )
    return "#" + "".join(f"{c:02x}" for c in channels)


def test_primary_button_label_readable_in_both_modes() -> None:
    # Streamlit renders primary-button labels white, so this measures label
    # against button fill — normal-text contrast, WCAG AA 4.5:1 (SC 1.4.3).
    # The floor was 3.0 while the theme was Streamlit's stock red (3.30:1); the
    # accent now clears AA (5.80 light / 4.60 dark), and the dark margin is
    # deliberate and narrow — +2 on every RGB channel of the dark primary drops
    # it to 4.48, because the same colour has to stay >= 3:1 as the focus
    # border on the panel (the test below). Re-run the helper before touching
    # either value; config.toml's ACCENT comment carries the numbers.
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        primary = theme[mode]["primaryColor"]
        ratio = _contrast_ratio("#ffffff", primary)
        assert ratio >= 4.5, f"{mode} primaryColor {primary} contrast {ratio:.2f}"


def test_primary_focus_border_clears_non_text_floor() -> None:
    # The accent is also the 1px focus border on the text areas and
    # selectboxes, painted against secondaryBackgroundColor, so it must clear
    # WCAG's 3:1 non-text floor (SC 1.4.11). Dark is 3.05 — the other side of
    # the knife-edge the label test describes: -2 per channel drops it to 2.97.
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        primary = theme[mode]["primaryColor"]
        panel = theme[mode]["secondaryBackgroundColor"]
        ratio = _contrast_ratio(primary, panel)
        assert ratio >= 3.0, (
            f"{mode} primaryColor {primary} on {panel} contrast {ratio:.2f}"
        )


def test_active_tab_label_does_not_regress() -> None:
    # Streamlit paints the selected tab's label in primaryColor as 14px text on
    # backgroundColor. No single primary can pass 4.5:1 both as that label and
    # under a white button label on any ordinary dark page, and the theme takes
    # the button side, so dark is the theme's one knowing AA text failure
    # (3.99:1; light passes at 5.33). The 3.5 floor is a regression guard, not
    # a standard being met — it holds the line against an accent that fades
    # into the page while the tab's 2px indicator still clears 3:1.
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        primary = theme[mode]["primaryColor"]
        bg = theme[mode]["backgroundColor"]
        ratio = _contrast_ratio(primary, bg)
        assert ratio >= 3.5, (
            f"{mode} primaryColor {primary} on {bg} contrast {ratio:.2f}"
        )


def test_link_text_readable_in_both_modes() -> None:
    # linkColor is body-size text on the page background, so it must clear WCAG
    # AA for normal text (4.5:1) in both modes. Dark linkColor is deliberately
    # not the primary: the button-side accent would sit at 3.99 here.
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        link = theme[mode]["linkColor"]
        bg = theme[mode]["backgroundColor"]
        ratio = _contrast_ratio(link, bg)
        assert ratio >= 4.5, f"{mode} linkColor {link} on {bg} contrast {ratio:.2f}"


def test_caption_readable_in_both_modes() -> None:
    # st.caption paints textColor at opacity 0.6 over the page (it never reads
    # grayTextColor), so the Document tab's provenance line is this composite.
    # Stock light sat at 3.69:1 — a documented, accepted gap; the ink is now
    # chosen so both modes clear AA (4.96 light / 5.78 dark).
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        text, bg = theme[mode]["textColor"], theme[mode]["backgroundColor"]
        caption = _composite(text, bg, 0.6)
        ratio = _contrast_ratio(caption, bg)
        assert ratio >= 4.5, f"{mode} caption {caption} on {bg} contrast {ratio:.2f}"


def test_code_background_matches_secondary_background() -> None:
    # The Text tab's input text_area (secondaryBackgroundColor) and output
    # st.code (codeBackgroundColor) sit side by side in one row; unpinned,
    # codeBackgroundColor falls back to a 50/50 blend of the two backgrounds
    # and the pair reads as two different panels. config.toml explains it; this
    # is the guard.
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        code = _hex6(theme[mode]["codeBackgroundColor"])
        panel = _hex6(theme[mode]["secondaryBackgroundColor"])
        assert code == panel, f"{mode}: codeBackgroundColor {code} != {panel}"


def test_theme_fonts_are_bundled_not_fetched() -> None:
    # The README's privacy promise rules out network fonts: a Google Fonts URL
    # in font/headingFont/codeFont fires on every page load, and
    # [[theme.fontFaces]] would need files this repo does not ship; the
    # generic names resolve to the Source Sans/Serif/Code files inside the
    # Streamlit wheel.
    # Every nested table is walked -- [theme.sidebar] and the per-mode
    # sidebars accept font sources too -- and `base` may only name a built-in:
    # a URL base is fetched at config load, and a local file base could carry
    # fontFaces of its own that this file never shows. This project inherits
    # nothing, so the key is simply not allowed to point at a file.
    def walk(table: dict[str, Any], path: str) -> None:
        assert "fontFaces" not in table, f"{path} declares fontFaces"
        base = table.get("base", "dark")
        assert base in {"light", "dark"}, f"{path}.base inherits from a file: {base}"
        for key, value in table.items():
            if isinstance(value, dict):
                walk(value, f"{path}.{key}")
            elif key.lower().endswith("font"):
                assert "://" not in str(value), f"{path}.{key} fetches: {value}"

    walk(_load_theme_config()["theme"], "theme")


def test_usage_stats_are_off() -> None:
    # Streamlit's browser.gatherUsageStats defaults to true, and the frontend
    # then fetches data.streamlit.io/metrics.json (once per browser; cached in
    # localStorage as stMetricsConfig) and POSTs usage events to the Fivetran
    # webhook it names on every page load. The README's privacy promise pins
    # it off here.
    config = _load_theme_config()
    assert config.get("browser", {}).get("gatherUsageStats") is False


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


# -- DOCUMENT_TYPES ------------------------------------------------------------


def test_document_types_are_liteparse_native() -> None:
    # PDFs and images parse with no external binary. Office formats would need
    # LibreOffice on PATH, and HTML and HEIC are rejected by the parser outright
    # ("unsupported file format"), so none are offered rather than accepting an
    # upload that fails at parse time. HEIC is the tempting one -- it is the
    # iPhone camera default, but listing it would only move the failure later.
    assert "pdf" in streamlit_app.DOCUMENT_TYPES
    assert "png" in streamlit_app.DOCUMENT_TYPES
    # Both TIFF spellings, or a .tif scan is unselectable in the file picker.
    assert {"tiff", "tif"} <= set(streamlit_app.DOCUMENT_TYPES)
    assert not {"docx", "pptx", "xlsx", "html", "heic"} & set(
        streamlit_app.DOCUMENT_TYPES
    )


# -- load_document_markdown ----------------------------------------------------


@patch("liteparse.LiteParse")
def test_load_document_markdown_returns_parsed_text(
    mock_liteparse_cls: MagicMock,
) -> None:
    mock_liteparse_cls.return_value.parse.return_value.text = "# Title\n\nBody."

    assert load_document_markdown(b"pdf bytes") == "# Title\n\nBody."
    mock_liteparse_cls.return_value.parse.assert_called_once_with(b"pdf bytes")


@patch("liteparse.LiteParse")
def test_load_document_markdown_requests_quiet_markdown(
    mock_liteparse_cls: MagicMock,
) -> None:
    mock_liteparse_cls.return_value.parse.return_value.text = "Body."
    load_document_markdown(b"data")

    kwargs = mock_liteparse_cls.call_args.kwargs
    assert kwargs["output_format"] == "markdown"
    # quiet: LiteParse otherwise prints timing lines to the server's stderr.
    assert kwargs["quiet"] is True


@patch("liteparse.LiteParse")
def test_load_document_markdown_enables_ocr_for_images(
    mock_liteparse_cls: MagicMock,
) -> None:
    # Regression: forcing ocr_enabled=False made every image type in
    # DOCUMENT_TYPES parse to an empty '```text```' fence. An image has no text
    # layer, so OCR is the only way to read one.
    mock_liteparse_cls.return_value.parse.return_value.text = "Body."
    load_document_markdown(b"\x89PNG\r\n\x1a\n rest of the image")

    assert mock_liteparse_cls.call_args.kwargs["ocr_enabled"] is True


@patch("liteparse.LiteParse")
def test_load_document_markdown_keeps_pdfs_offline(
    mock_liteparse_cls: MagicMock,
) -> None:
    # A PDF carries a real text layer, and LiteParse's auto OCR downloads ~15 MB
    # of Tesseract training data from GitHub the first time it fires. The
    # README's privacy promise keeps the common path offline.
    mock_liteparse_cls.return_value.parse.return_value.text = "Body."
    load_document_markdown(b"%PDF-1.4\nrest of the file")

    assert mock_liteparse_cls.call_args.kwargs["ocr_enabled"] is False


@patch("liteparse.LiteParse")
def test_load_document_markdown_lets_an_ocr_failure_raise(
    mock_liteparse_cls: MagicMock,
) -> None:
    # Only images run OCR, and an image has no text layer to fall back to, so
    # a non-fatal failure could only surface as "No translatable text found".
    # Leaving LiteParse's default (fatal) lets the tab report the real error.
    from liteparse import ParseError

    mock_liteparse_cls.return_value.parse.side_effect = ParseError("OCR failed")

    with pytest.raises(ParseError):
        load_document_markdown(b"\x89PNG fake")
    assert mock_liteparse_cls.call_args.kwargs.get("ocr_failure_fatal") is not False


def test_is_pdf_sniffs_the_magic_not_the_name() -> None:
    assert is_pdf(b"%PDF-1.4\nrest of the file")
    assert is_pdf(b"  \n%PDF-1.7")  # leading whitespace is tolerated
    assert not is_pdf(b"\x89PNG\r\n\x1a\n")
    assert not is_pdf(b"")
    assert not is_pdf(b"PDF without the percent")


@patch("liteparse.LiteParse")
def test_load_document_markdown_blanks_an_empty_fence(
    mock_liteparse_cls: MagicMock,
) -> None:
    # An unreadable file comes back as an empty code fence, which is not blank
    # and would otherwise sail past the tab's "no translatable text" guard and
    # be handed to the model as a prompt.
    mock_liteparse_cls.return_value.parse.return_value.text = "```text\n\n```"

    assert load_document_markdown(b"data") == ""


@patch("liteparse.LiteParse")
def test_load_document_markdown_ocrs_in_the_source_language(
    mock_liteparse_cls: MagicMock,
) -> None:
    # Regression: OCR always ran as English, so a Cyrillic or CJK scan came
    # back as plausible-looking Latin garbage with words silently dropped --
    # non-empty, so it passed the "no translatable text" guard.
    mock_liteparse_cls.return_value.parse.return_value.text = "Body."
    load_document_markdown(b"\x89PNG\r\n\x1a\n data", "Russian")

    assert mock_liteparse_cls.call_args.kwargs["ocr_language"] == "rus"


@patch("liteparse.LiteParse")
def test_load_document_markdown_falls_back_to_english_ocr(
    mock_liteparse_cls: MagicMock,
) -> None:
    # Tesseract publishes no traineddata for Zulu, so asking for it would
    # request a file that does not exist.
    mock_liteparse_cls.return_value.parse.return_value.text = "Body."
    load_document_markdown(b"\x89PNG\r\n\x1a\n data", "Zulu")

    assert mock_liteparse_cls.call_args.kwargs["ocr_language"] == "eng"


def test_ocr_languages_are_all_real_app_languages() -> None:
    # A typo here silently downgrades a language to English OCR.
    assert set(OCR_LANGUAGES) <= set(LANGUAGES)
    assert OCR_LANGUAGES["Japanese"] == "jpn"
    assert OCR_LANGUAGES["Chinese"] == "chi_sim"


def test_is_blank_markdown_keeps_real_fenced_content() -> None:
    assert is_blank_markdown("```text\n\n```")
    assert is_blank_markdown("   \n\n  ")
    assert not is_blank_markdown("```text\nHello\n```")
    assert not is_blank_markdown("# Title")


# -- heading_level -------------------------------------------------------------


def test_heading_level_reads_atx_depth() -> None:
    assert heading_level("# Title") == 1
    assert heading_level("### Section") == 3
    assert heading_level("###### Deepest") == 6


def test_heading_level_zero_for_non_headings() -> None:
    assert heading_level("Just a paragraph.") == 0
    assert heading_level("") == 0
    # Seven hashes is past the ATX limit, and #tag has no space delimiter.
    assert heading_level("####### Too deep") == 0
    assert heading_level("#hashtag not a heading") == 0


def test_heading_level_reads_only_the_first_line() -> None:
    # Blocks split on blank lines, so a heading with its opening body line
    # attached is one block and still counts as a heading.
    assert heading_level("## Section\nFirst body line.") == 2


# -- heading_line --------------------------------------------------------------


def test_heading_line_returns_only_the_first_line() -> None:
    assert heading_line("## Section\nFirst body line.") == "## Section"
    assert heading_line("# Title") == "# Title"


def test_chunk_text_heading_trail_excludes_attached_body_text() -> None:
    # Regression: the trail stored the whole block, so a heading followed
    # immediately by its first line (no blank line between, which is ordinary
    # in parsed markdown) prepended that body text to every later chunk and
    # charged its tokens against the budget the trail reserves.
    text = "# Title\nBody line under the heading\n\n" + "\n\n".join(
        f"para{i} " + "word " * 20 for i in range(3)
    )

    chunks = chunk_text(text, FakeTokenizer(), max_tokens=40)

    tail = next(c for c in chunks if "para2" in c)
    assert "# Title" in tail
    assert "Body line under the heading" not in tail


# -- chunk_text ----------------------------------------------------------------


class FakeTokenizer:
    """Whitespace tokenizer so chunk_text budgets are readable in tests."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


class SpecialTokenTokenizer:
    """Whitespace tokenizer that mimics the model tokenizer's quirks.

    The real tokenizer prepends a BOS token to every ``encode`` call and spends
    a token on a blank line. Both make an assembled chunk cost more than the sum
    of its blocks, which is what pushed chunks over budget before the packer
    charged for separators and measured the finished chunk. ``decode`` leaves
    the BOS visible so a leaked special token fails a test loudly.
    """

    def encode(self, text: str) -> list[str]:
        tokens: list[str] = []
        for i, part in enumerate(text.split("\n\n")):
            if i:
                tokens.append("\n\n")
            tokens.extend(part.split())
        return ["<bos>", *tokens]

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


def _tok_len(text: str) -> int:
    return len(FakeTokenizer().encode(text))


def test_chunk_text_returns_empty_for_blank_input() -> None:
    assert chunk_text("", FakeTokenizer()) == []
    assert chunk_text("   \n\n  \n", FakeTokenizer()) == []


def test_chunk_text_keeps_every_chunk_within_budget() -> None:
    text = "\n\n".join(
        ["# Report", "alpha " * 20, "## Section", "beta " * 30, "gamma " * 25]
    )
    for budget in (10, 25, 60):
        chunks = chunk_text(text, FakeTokenizer(), max_tokens=budget)
        assert chunks, f"budget {budget} produced no chunks"
        assert all(_tok_len(c) <= budget for c in chunks), f"over budget at {budget}"


def test_chunk_text_packs_multiple_paragraphs_into_one_chunk() -> None:
    # Three short paragraphs fit in one generous chunk; packing greedily keeps
    # the chunk count (and so the per-chunk prompt overhead) down.
    text = "one two three\n\nfour five six\n\nseven eight nine"

    assert len(chunk_text(text, FakeTokenizer(), max_tokens=100)) == 1


def test_chunk_text_does_not_split_a_paragraph_that_fits() -> None:
    text = "alpha beta gamma\n\ndelta epsilon zeta"
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=3)

    assert "alpha beta gamma" in chunks
    assert "delta epsilon zeta" in chunks


def test_chunk_text_prepends_enclosing_heading_context() -> None:
    text = "# Report\n\n## Findings\n\n" + "word " * 30
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=20)

    # The body chunk carries the headings it sits under, because each chunk is
    # translated as an independent prompt with no memory of its neighbours.
    body = [c for c in chunks if "word" in c]
    assert body
    assert all(c.startswith("# Report\n\n## Findings") for c in body)


def test_chunk_text_does_not_duplicate_heading_it_opens_with() -> None:
    text = "# Report\n\nbody text here"
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=100)

    assert chunks[0].count("# Report") == 1


def test_chunk_text_uses_enclosing_heading_not_the_following_one() -> None:
    # Regression: the heading trail advances as blocks are read, so a chunk
    # flushed after a later heading was seen must still carry the heading of
    # the section it actually belongs to -- not the next section's.
    text = "## First\n\naaa bbb ccc\n\n## Second\n\nddd eee fff"
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=6)

    first = next(c for c in chunks if "aaa bbb ccc" in c)
    assert "## First" in first
    assert "## Second" not in first


def test_chunk_text_deeper_heading_replaces_sibling_shallower_survives() -> None:
    text = "# Book\n\n## Alpha\n\n## Beta\n\n" + "tail " * 20
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=12)

    tail = next(c for c in chunks if "tail" in c)
    assert "# Book" in tail
    assert "## Beta" in tail
    # Alpha was replaced by its sibling Beta, so it must not linger in context.
    assert "## Alpha" not in tail


def test_chunk_text_splits_oversized_paragraph_on_sentence_boundaries() -> None:
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=4)

    assert len(chunks) > 1
    # Sentences stay intact rather than being cut mid-clause.
    assert all(c.strip().endswith(".") for c in chunks)


def test_chunk_text_hard_splits_a_single_unpunctuated_run() -> None:
    text = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(text, FakeTokenizer(), max_tokens=6)

    assert len(chunks) > 1
    assert all(_tok_len(c) <= 6 for c in chunks)
    # No content is dropped on the hard-split path.
    assert " ".join(chunks).split() == text.split()


def test_chunk_text_respects_budget_when_tokenizer_adds_special_tokens() -> None:
    # Regression: summing per-block counts charges one BOS per block and none of
    # the separators, so chunks used to land over budget with a real tokenizer.
    tok = SpecialTokenTokenizer()
    overhead = len(tok.encode(""))
    text = "\n\n".join(
        ["# Report", "alpha " * 12, "## Section", "beta " * 18, "gamma " * 9]
    )
    for budget in (12, 20, 40, 80):
        chunks = chunk_text(text, tok, max_tokens=budget)
        assert chunks, f"budget {budget} produced no chunks"
        for chunk in chunks:
            size = len(tok.encode(chunk)) - overhead
            assert size <= budget, f"{size} > {budget} for {chunk!r}"


class DriftingTokenizer(SpecialTokenTokenizer):
    """Whitespace tokenizer that spends an extra token where two blocks join.

    A BPE tokenizer merges differently across a block boundary than inside one,
    so an assembled chunk measures slightly more than the sum of its blocks. The
    packer cannot see that drift while packing, so the finished chunk lands just
    over budget -- the exact condition that used to shatter it into one prompt
    per sentence. SpecialTokenTokenizer models the BOS and separator costs the
    packer *does* account for; only this one reproduces the overflow.
    """

    def encode(self, text: str) -> list[str]:
        tokens: list[str] = []
        for i, part in enumerate(text.split("\n\n")):
            if i:
                tokens.append("\n\n")
            words = part.split()
            if i and words:
                tokens.append("<merge>")
            tokens.extend(words)
        return ["<bos>", *tokens]

    def decode(self, ids: list[str]) -> str:
        # Drop the synthetic separator and merge markers so decode inverts
        # encode. A real tokenizer round-trips its ids; leaving these in would
        # write "<merge>" into the chunk text and fail tests for a defect in
        # the fixture rather than in the code. The BOS is still left visible,
        # as in the parent, so a genuinely leaked special token is caught.
        return " ".join(i for i in ids if i not in {"\n\n", "<merge>"})


def test_chunk_text_does_not_shatter_a_full_chunk_when_the_estimate_drifts() -> None:
    # Regression: a packed chunk measuring one token over budget was split on
    # every sentence boundary and the pieces appended straight to the output, so
    # one full chunk became one chunk per sentence. Each is its own translation
    # call stripped of its neighbours, turning a long document into hundreds of
    # prompts. An upper bound on chunk size cannot see this -- only the count.
    tok = DriftingTokenizer()
    overhead = len(tok.encode(""))
    text = "\n\n".join(f"Sentence {i}." for i in range(30))
    budget = 60

    chunks = chunk_text(text, tok, max_tokens=budget)

    assert all(len(tok.encode(c)) - overhead <= budget for c in chunks)
    # 30 two-token blocks against a 60-token budget is a handful of chunks;
    # shattering produced one per sentence.
    assert len(chunks) <= 5, f"chunk count blew up: {len(chunks)}"


def test_chunk_text_drift_repack_keeps_blocks_separated() -> None:
    # Re-packing works on whole blocks, so the blank lines between paragraphs
    # survive. Splitting the assembled body on sentences instead would dissolve
    # every paragraph, list and table boundary in the chunk.
    tok = DriftingTokenizer()
    text = "\n\n".join(f"Sentence {i}." for i in range(30))

    chunks = chunk_text(text, tok, max_tokens=60)

    # Thirty two-token blocks against a sixty-token budget: every chunk should
    # hold many blocks. Splitting the body on sentences consumed the blank
    # lines and emitted each sentence alone, leaving a run of singletons.
    assert all(c.count("Sentence") > 1 for c in chunks), (
        f"chunks were shattered into singletons: "
        f"{[c.count('Sentence') for c in chunks]}"
    )
    # Blocks inside a chunk stay separated by the blank line they were split on.
    assert all(c.count("\n\n") == c.count("Sentence") - 1 for c in chunks)


class ByteFallbackTokenizer:
    """Whitespace-free tokenizer that falls back to UTF-8 bytes for non-ASCII.

    A real BPE tokenizer spends several tokens on a single character in scripts
    outside its vocabulary, so slicing ids at a fixed stride can cut a character
    in half. ``decode`` reassembles the bytes and, like a real tokenizer, yields
    U+FFFD wherever a sequence was cut -- which is what makes the corruption
    visible to a test. The plain fakes round-trip perfectly and cannot see it.
    """

    def encode(self, text: str) -> list[str]:
        tokens: list[str] = ["<bos>"]
        for char in text:
            if char.isascii():
                tokens.append(char)
            else:
                tokens.extend(f"<{byte}>" for byte in char.encode("utf-8"))
        return tokens

    def decode(self, ids: list[str]) -> str:
        out = bytearray()
        for token in ids:
            if token.startswith("<") and token.endswith(">") and token[1:-1].isdigit():
                out.append(int(token[1:-1]))
            else:
                out.extend(token.encode("utf-8"))
        return out.decode("utf-8", errors="replace")


def test_hard_windows_do_not_cut_multibyte_characters() -> None:
    # Regression: a fixed token stride cut characters in half in byte-fallback
    # scripts -- Thai, Lao, Khmer, Burmese and Amharic are all in LANGUAGES.
    # The halves decoded to U+FFFD and the character was lost outright, and
    # that corrupted text was what reached the model as source.
    tok = ByteFallbackTokenizer()
    text = "ประเทศไทย" * 6

    for budget in range(3, 20):
        joined = "".join(hard_windows(text, tok, budget))
        assert "�" not in joined, f"character cut in half at budget {budget}"
        assert joined == text, f"content changed at budget {budget}"


def test_chunk_text_preserves_byte_fallback_scripts() -> None:
    tok = ByteFallbackTokenizer()
    text = "ประเทศไทยตั้งอยู่ในภูมิภาคเอเชีย " * 12

    for budget in (24, 48, 96):
        joined = "".join(chunk_text(text, tok, max_tokens=budget))
        assert "�" not in joined, f"corrupted at budget {budget}"
        # Chunks are stripped, so the space at a chunk boundary is gone;
        # compare the content itself rather than the word split.
        assert joined.replace(" ", "") == text.replace(" ", ""), (
            f"content lost at budget {budget}"
        )


def test_chunk_text_labels_every_piece_when_the_trail_is_still_empty() -> None:
    # Regression: the first chunk opens before any heading has reached the
    # trail, so its headings live in its own blocks. Pieces cut out of its
    # middle contain neither those blocks nor a trail to draw them from, and
    # reached the model as unlabelled fragments.
    tok = DriftingTokenizer()
    text = "# Handbook\n\n## Parts\n\n" + "\n\n".join(f"- Bolt {i}" for i in range(60))

    chunks = chunk_text(text, tok, max_tokens=40)

    assert len(chunks) > 1, "expected the document to split"
    assert all("# Handbook" in c and "## Parts" in c for c in chunks)


def test_leading_headings_stops_at_the_first_body_block() -> None:
    # Only the headings a chunk *opens* with make prepended context redundant.
    # A heading further in starts the next section, so the body ahead of it
    # still needs its own section's heading.
    assert leading_headings(["# A", "## B", "body", "## C"]) == ["# A", "## B"]
    assert leading_headings(["body", "# A"]) == []
    assert leading_headings([]) == []


def test_pack_by_estimate_charges_the_join_cost() -> None:
    # Regression: ignoring the separator between blocks let a group of 985
    # blocks estimated at 5,792 tokens measure 7,760 -- a quarter over budget.
    # The overflow then fell through to the path that strips heading context.
    tok = FakeTokenizer()
    units = ["a b"] * 10

    assert len(pack_by_estimate(units, tok, 6, join_cost=0)) == 4
    assert len(pack_by_estimate(units, tok, 6, join_cost=1)) == 5


def test_chunk_text_drops_context_rather_than_shredding_the_document() -> None:
    # Regression: budget = max(max_tokens - trail, 1) collapsed to one token
    # once the heading trail reached the budget, so the packer emitted roughly
    # a chunk per token -- 13,121 tokens of markdown became 12,082 chunks.
    # Giving up the context is the lesser loss.
    tok = FakeTokenizer()
    heading = "# " + "Section " * 40  # far larger than the budget below
    text = heading + "\n\n" + "\n\n".join(f"para {i} body text" for i in range(20))

    chunks = chunk_text(text, tok, max_tokens=20)

    assert len(chunks) < 20, f"document was shredded into {len(chunks)} chunks"
    assert all(_tok_len(c) <= 20 for c in chunks)


def test_chunk_text_accepts_a_non_positive_budget() -> None:
    # max_tokens reaches this from cached_document_chunks, and a zero budget
    # used to raise ValueError from range()'s zero step.
    assert chunk_text("alpha beta", FakeTokenizer(), max_tokens=0)


def test_chunk_text_ignores_headings_inside_a_code_fence() -> None:
    # Regression: _BLANK_LINE_RE splits a fence at its internal blank line, so
    # a '#' comment inside one read as an H1 and evicted the real heading --
    # every chunk below was then labelled with a line of shell.
    tok = FakeTokenizer()
    text = (
        "# Install\n\n```bash\nmake all\n\n# clean up the build tree\n"
        "make clean\n```\n\n" + "\n\n".join(f"step {i} here" for i in range(12))
    )

    chunks = chunk_text(text, tok, max_tokens=12)

    tail = next(c for c in chunks if "step 11" in c)
    assert "# Install" in tail
    assert "# clean up the build tree" not in tail


def test_chunk_text_resumes_headings_after_a_closed_fence() -> None:
    # The fence flag must clear again, or every heading after the first code
    # block would be ignored.
    tok = FakeTokenizer()
    text = "# One\n\n```\ncode\n```\n\n## Two\n\n" + "\n\n".join(
        f"body {i}" for i in range(10)
    )

    chunks = chunk_text(text, tok, max_tokens=10)

    tail = next(c for c in chunks if "body 9" in c)
    assert "## Two" in tail


def test_chunk_text_does_not_leak_special_tokens_into_output() -> None:
    # The hard-split path slices token ids, so it must drop the BOS prefix
    # first or decode would write it back into the translated text.
    tok = SpecialTokenTokenizer()
    chunks = chunk_text(" ".join(f"w{i}" for i in range(40)), tok, max_tokens=5)

    assert chunks
    assert not any("<bos>" in c for c in chunks)


# -- token_overhead / count_tokens ---------------------------------------------


def test_token_overhead_counts_the_special_prefix() -> None:
    assert token_overhead(SpecialTokenTokenizer()) == 1
    assert token_overhead(FakeTokenizer()) == 0


def test_count_tokens_excludes_the_special_prefix() -> None:
    # Net length, so summing block counts doesn't charge a BOS per block.
    assert count_tokens("one two three", SpecialTokenTokenizer()) == 3
    assert count_tokens("", SpecialTokenTokenizer()) == 0


# -- split_oversized -----------------------------------------------------------


def test_split_oversized_keeps_whole_sentences_when_they_fit() -> None:
    pieces = split_oversized("Alpha one. Beta two.", FakeTokenizer(), 3)

    assert pieces == ["Alpha one.", "Beta two."]


def test_split_oversized_repacks_sentences_up_to_the_budget() -> None:
    # Regression: sentences used to be emitted one per piece regardless of the
    # budget, so a body one token over turned into one prompt per sentence.
    text = "Alpha one. Beta two. Gamma three. Delta four."
    pieces = split_oversized(text, FakeTokenizer(), 4)

    assert pieces == ["Alpha one. Beta two.", "Gamma three. Delta four."]


def test_sentence_units_splits_cjk_written_without_spaces() -> None:
    # Regression: the boundary pattern required whitespace after the ender, but
    # CJK writes 。 flush against the next sentence, so real CJK never split at
    # all and fell through to mid-clause hard token windows. Chinese, Japanese,
    # Korean, Thai and Lao are all in LANGUAGES.
    units = sentence_units("第一文です。第二文です。第三文です。")

    assert units == ["第一文です。", "第二文です。", "第三文です。"]


def test_sentence_units_concatenate_back_to_the_source() -> None:
    # Units carry their own trailing gap, so re-packing them cannot invent a
    # space that the source never had -- which is exactly the CJK case.
    for text in (
        "Alpha one. Beta two.",
        "第一文です。第二文です。",
        "Mixed. 混在です。Tail sentence.",
    ):
        assert "".join(sentence_units(text)) == text


def test_sentence_units_keep_decimals_intact() -> None:
    # The Latin branch still demands whitespace after the ender, so a decimal
    # point is not a sentence boundary.
    assert sentence_units("Pi is 3.14 exactly.") == ["Pi is 3.14 exactly."]


# -- cached_document_chunks ----------------------------------------------------


@patch("streamlit_app.chunk_text")
@patch("streamlit_app.load_document_markdown")
@patch("streamlit_app.load_model")
def test_cached_document_chunks_composes_parse_and_chunk(
    mock_load_model: MagicMock,
    mock_load_markdown: MagicMock,
    mock_chunk_text: MagicMock,
) -> None:
    cached = streamlit_app.cached_document_chunks
    cached.clear()
    tokenizer = MagicMock()
    mock_load_model.return_value = (MagicMock(), tokenizer)
    mock_load_markdown.return_value = "# Doc\n\nBody."
    mock_chunk_text.return_value = ["chunk a", "chunk b"]

    result = cached(b"bytes", "Thai")

    # Pulls the tokenizer from load_model() rather than taking it as an arg,
    # then composes load_document_markdown -> chunk_text. LiteParse sniffs the
    # format itself, so no filename is threaded through -- but the source
    # language is, since it selects the OCR language for image uploads.
    assert result == ["chunk a", "chunk b"]
    mock_load_markdown.assert_called_once_with(b"bytes", "Thai")
    mock_chunk_text.assert_called_once_with(
        "# Doc\n\nBody.", tokenizer, streamlit_app.MAX_CHUNK_TOKENS
    )


@patch("streamlit_app.chunk_text")
@patch("streamlit_app.load_document_markdown")
@patch("streamlit_app.load_model")
def test_cached_document_chunks_forwards_custom_max_tokens(
    mock_load_model: MagicMock,
    _mock_load_markdown: MagicMock,
    mock_chunk_text: MagicMock,
) -> None:
    cached = streamlit_app.cached_document_chunks
    cached.clear()
    mock_load_model.return_value = (MagicMock(), MagicMock())
    mock_chunk_text.return_value = []

    cached(b"x", max_tokens=1234)

    assert mock_chunk_text.call_args[0][2] == 1234


def test_document_tab_calls_cache_wrapper() -> None:
    # AppTest can't drive st.file_uploader, so guard the wiring at the source
    # level: the Document tab must go through cached_document_chunks, not call
    # load_document_markdown/chunk_text directly (which would silently bypass
    # the parse+chunk cache). Strip whitespace so wrapping can't break the match.
    compact = "".join(_APP_SOURCE.split())
    assert (
        "cached_document_chunks(uploaded.getvalue(),st.session_state.doc_source_lang)"
        in compact
    )


def test_document_tab_loads_model_into_its_own_status_slot() -> None:
    # AppTest can't drive st.file_uploader, so guard this at the source level
    # too. Three ways to get it wrong, none of which any test would catch:
    # passing warning_slot (the Text tab's, also in scope here) would print the
    # load error in the wrong tab; inverting the guard to `is None` would
    # unpack None and raise TypeError; and doc_warning_slot -- the previous
    # value -- put the tab's longest-running activity indicator in the slot
    # reserved for warnings and errors.
    compact = "".join(_APP_SOURCE.split())
    assert "elif(loaded:=ensure_model(doc_status_slot))isnotNone:" in compact


def test_spinner_slots_are_empty_elements_not_containers() -> None:
    # Every slot a spinner lands in must be an st.empty(), never an
    # st.container(): a transient spinner's clear message leaves a container
    # with a phantom child that holds the column's 16px gap for the run --
    # mechanism and measurements at warning_slot in streamlit_app.py. AppTest
    # renders no layout, so only the source can hold this.
    #
    # The names come from the call sites, not a hand-written list: the
    # argument of every ensure_model(...) call, plus every `with x.spinner(`
    # receiver other than ensure_model's own parameter (which those arguments
    # feed). A new slot handed a spinner is caught automatically, and a
    # renamed slot cannot slip past a stale list. Anchored at line start
    # rather than matched in the compact form, because the compact
    # `doc_warning_slot=st.container()` -- a container, and rightly: it only
    # ever receives real elements -- contains `warning_slot=st.container()` as
    # a substring. Any DG's .empty() passes; exactly one assignment each.
    src = _APP_SOURCE
    param_match = re.search(r"^def ensure_model\((\w+)", src, re.M)
    assert param_match is not None
    receivers = set(re.findall(r"(?<!def )ensure_model\((\w+)\)", src))
    receivers |= set(re.findall(r"^\s*with (\w+)\.spinner\(", src, re.M))
    receivers -= {param_match.group(1)}
    assert receivers == {"warning_slot", "doc_status_slot"}
    for name in receivers:
        callees = re.findall(rf"^\s*{name}\s*=\s*(\w+\.\w+)\(\s*\)", src, re.M)
        assert len(callees) == 1 and callees[0].endswith(".empty"), (name, callees)


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


# -- render_output -------------------------------------------------------------


def test_render_output_renders_code_with_panel_height() -> None:
    # render_output is the single sink for streamed output across both tabs; it
    # must use st.code (non-widget, replaceable mid-script without a widget-id
    # collision) with wrap_lines and the shared PANEL_HEIGHT, not st.text_area.
    placeholder = MagicMock()

    render_output(placeholder, "Bonjour")

    placeholder.code.assert_called_once_with(
        "Bonjour",
        language=None,
        wrap_lines=True,
        height=streamlit_app.PANEL_HEIGHT,
    )


def test_render_output_is_single_code_sink() -> None:
    # render_output is the single st.code sink; guard against re-inlining
    # `.code(...)` at a call site, which passes the helper unit test and the
    # height-literal guard yet defeats the dedup the refactor exists for.
    assert _APP_SOURCE.count(".code(") == 1


# -- document provenance -------------------------------------------------------


def test_document_meta_line_names_file_and_direction() -> None:
    # The filename is escaped because st.caption renders Markdown; the escapes
    # are not displayed. Language names are plain letters and pass through.
    assert (
        document_meta_line("report.pdf", "English", "French")
        == "report\\.pdf \u00b7 English \u2192 French"
    )


def test_document_meta_line_escapes_markdown_in_the_filename() -> None:
    # Unescaped, *draft*.pdf renders as italic "draft.pdf" and a bracket-paren
    # name renders as a live link -- in the caption whose job is to state the
    # filename accurately.
    assert document_meta_line("*draft*.pdf", "English", "French").startswith(
        "\\*draft\\*\\.pdf"
    )
    assert "](" not in document_meta_line(
        "[r](https://example.com).pdf", "English", "French"
    )


def test_document_download_name_strips_header_breaking_characters() -> None:
    # Streamlit builds Content-Disposition as f'filename="{filename}"' with no
    # quoting, so a double quote in an upload name truncates the saved file.
    assert document_download_name('re"port.pdf', "French") == "re_port-French.md"
    assert document_download_name("a/b.pdf", "French") == "a_b-French.md"
    assert document_download_name("x\nnew.pdf", "French") == "x_new-French.md"


def test_document_download_name_keeps_non_ascii() -> None:
    # Nothing to strip either way, but the two names take DIFFERENT branches:
    # Streamlit selects on filename.encode("latin1"), not on ASCII. Bokmal
    # encodes to latin1 and goes down the quoted branch as a raw byte; the CJK
    # name raises UnicodeEncodeError and takes filename*=utf-8''.
    assert document_download_name("rapport.pdf", "Bokm\u00e5l") == (
        "rapport-Bokm\u00e5l.md"
    )
    assert document_download_name("\u5831\u544a.pdf", "French") == (
        "\u5831\u544a-French.md"
    )


def test_document_download_name_uses_stem_and_target() -> None:
    # Every output used to be translation.md, so a run of documents produced
    # translation.md, translation (1).md ... with nothing to tell them apart.
    assert document_download_name("report.pdf", "French") == "report-French.md"


def test_document_download_name_splits_on_the_last_dot() -> None:
    assert document_download_name("scan.tar.gz", "German") == "scan.tar-German.md"


def test_document_download_name_handles_names_without_a_usable_stem() -> None:
    # A dotless upload keeps its whole name; a dot-leading one is not treated
    # as all-extension; an empty name still yields a valid download.
    assert document_download_name("noext", "Spanish") == "noext-Spanish.md"
    assert document_download_name(".hidden", "Spanish") == ".hidden-Spanish.md"
    assert document_download_name("", "French") == "translation-French.md"


def test_text_download_button_uses_the_recorded_name() -> None:
    # AppTest cannot see a download button's file_name (the proto carries only
    # a media-manager url), so guard the wiring here: the Text tab must read
    # the captured name, never re-inline the old constant.
    assert "file_name=st.session_state.download_name," in _APP_SOURCE
    # Each default name is written exactly once, at its constant. Three reset
    # paths used to inline "translation.txt" and two inlined "translation.md",
    # so changing one left the others behind with nothing failing.
    assert _APP_SOURCE.count('"translation.txt"') == 1
    assert _APP_SOURCE.count('"translation.md"') == 1
    assert "DEFAULT_TEXT_DOWNLOAD" in _APP_SOURCE
    assert "DEFAULT_DOC_DOWNLOAD" in _APP_SOURCE


def test_escape_markdown_leaves_plain_text_alone() -> None:
    assert escape_markdown("report pdf") == "report pdf"


def test_escape_markdown_neutralises_streamlit_markup() -> None:
    # Asserted by OUTCOME, not by re-listing the regex's own character class --
    # the previous version looped the identical literals the pattern is built
    # from, so it could not fail while advertising "every metacharacter".
    # st.caption renders more than CommonMark: LaTeX, emoji shortcodes and HTML
    # entities all need neutralising too.
    for raw, must_not_contain in [
        ("*draft*.pdf", "*draft*"),
        ("[r](https://example.com).pdf", "]("),
        ("budget $100 vs $200.pdf", "$100 vs $200"),
        ("photo:sunglasses:.pdf", ":sunglasses:"),
        ("AT&amp;T.pdf", "&amp;"),
    ]:
        assert must_not_contain not in escape_markdown(raw), raw
    # A leading "#" would render as a heading; it must carry its backslash.
    assert escape_markdown("# heading.pdf").startswith("\\#")


def test_escape_markdown_covers_all_ascii_punctuation() -> None:
    # CommonMark permits a backslash before any ASCII punctuation, and renders
    # it as the bare character -- so over-escaping is free and under-escaping
    # is the only failure mode.
    import string

    for ch in string.punctuation:
        assert escape_markdown(ch) == "\\" + ch
