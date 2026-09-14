# SPDX-License-Identifier: Apache-2.0
import ast
import os
import re
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from streamlit.commands.page_config import _get_favicon_string

import streamlit_app
from streamlit_app import (
    LANGUAGES,
    build_translation_prompt,
    clean_model_output,
    clip_to_input_cap,
    render_output,
    settled_download_name,
    stream_translate,
    tokenize_prompt,
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
    # The two full-width controls (translate, download) set width="stretch".
    # The swap button is deliberately NOT among them: stretched inside a 1-unit
    # column it became an invisible full-row tap target below the 640px
    # breakpoint, so it is pinned to a fixed width and centred instead.
    assert _APP_SOURCE.count('width="stretch"') == 2
    assert _APP_SOURCE.count("width=40,") == 1


# -- streamlit_app.py shared UI constants --------------------------------------


def test_panel_height_not_hardcoded() -> None:
    # The panel height lives in PANEL_HEIGHT; guard against re-inlining a
    # number at a call site (the text_area panels, the skeleton or
    # render_output). A digit rather than the current value, so the guard
    # cannot go stale when the constant is retuned -- it pinned "height=450"
    # until the 2026-09-13 fit, which would have waved a re-inlined 438
    # straight through. The swap button's width=40 is the app's only literal
    # dimension, and it is a width.
    assert not re.search(r"height=\d", _APP_SOURCE)


def test_panel_height_fits_the_main_display_fold() -> None:
    # The resting page is designed to fit the 1920x839 viewport the main
    # display gives (1080 tall, the Dock showing, Chrome at 960) with nothing
    # to scroll, because a page that scrolls on macOS with a mouse attached
    # also grows a classic 11px scrollbar that takes layout width. Ledger
    # measured 2026-09-13 with getBoundingClientRect, Streamlit 1.63.0, the
    # controls docked in st.bottom: 6rem top padding, the st.title element
    # container, the root block's two 16px gaps around the bordered language
    # row, the main block's 1rem bottom padding (st.bottom's doing -- it is
    # 10rem without one), and the bar itself, 1rem + the 40px buttons +
    # 3.5rem. Static arithmetic: it fails a bump of the constant, catches a
    # row added above the bar only if its height is added here, and cannot
    # see Streamlit's chrome move on an upgrade -- re-measure section.stMain's
    # scrollHeight against clientHeight then, rather than trust it. `<=`, not
    # `==`, so deliberate slack still passes.
    chrome = 96 + 72.8 + 16 + 72 + 16 + 16 + (16 + 40 + 56)
    assert chrome + streamlit_app.PANEL_HEIGHT <= 839


def test_controls_row_is_docked_and_never_stacks() -> None:
    # The docking itself is pinned at runtime in test_streamlit_ui.py
    # (test_controls_row_lives_in_the_bottom_bar); AppTest renders no layout,
    # so the columns' wrap flag is pinned here. Parsed rather than counted,
    # so the comment above the row can name wrap=False in prose. Exactly one
    # st.bottom block anywhere in the module (walked, not just the top level,
    # so a second bar nested in the translate block cannot hide), at the top
    # level, holding exactly one st.columns call, and that call opts out of
    # stacking: a stacked pair turns the sticky bar into a 168px permanent
    # overlay below 640px. The flag is only worth pinning if the buttons are
    # actually in those columns, so the last block ties them: the call's two
    # targets are the only `with` receivers the buttons sit under, one each,
    # and no button in the bar sits outside them -- a Translate emitted
    # directly under st.bottom would pass the wrap check and still stack.
    tree = ast.parse(_APP_SOURCE)
    bottoms = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(ast.unparse(item.context_expr) == "st.bottom" for item in node.items)
    ]
    assert len(bottoms) == 1
    assert bottoms[0] in tree.body
    columns = [
        node
        for node in ast.walk(bottoms[0])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "st.columns"
    ]
    assert len(columns) == 1
    call = columns[0].value
    assert isinstance(call, ast.Call)
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords.get("wrap") == "False", keywords
    (target,) = columns[0].targets
    assert isinstance(target, ast.Tuple)
    slots = [ast.unparse(name) for name in target.elts]
    assert len(slots) == 2

    def buttons_in(node: ast.AST) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) in {"st.button", "st.download_button"}
        ]

    per_slot = {
        ast.unparse(w.items[0].context_expr): len(buttons_in(w))
        for w in ast.walk(bottoms[0])
        if isinstance(w, ast.With) and ast.unparse(w.items[0].context_expr) in slots
    }
    assert per_slot == dict.fromkeys(slots, 1), per_slot
    assert len(buttons_in(bottoms[0])) == 2


def test_warning_strings_defined_once() -> None:
    # Each warning lives in exactly one place — its module-level constant. The
    # call sites reference SAME_LANGUAGE_WARNING / NO_OUTPUT_WARNING, so neither
    # raw literal should be duplicated.
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


def test_primary_on_page_does_not_regress() -> None:
    # primaryColor against backgroundColor. The floor dates from the page's tab
    # strip: Streamlit paints the selected tab's label in primaryColor as 14px
    # text on backgroundColor, no single primary can pass 4.5:1 both as that
    # label and under a white button label on any ordinary dark page, and the
    # theme took the button side — dark sat at 3.99:1 (light 5.33) as the
    # theme's one knowing AA text failure. The tabs are gone, and with them
    # that label; the 3.5 floor stays as a regression guard, not a standard
    # being met — it holds the line against an accent that fades into the page.
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
    # grayTextColor), so any caption is this composite. The app paints none
    # today; the guard keeps the ink caption-safe for a future one.
    # Stock light sat at 3.69:1 — a documented, accepted gap; the ink is now
    # chosen so both modes clear AA (4.96 light / 5.78 dark).
    theme = _load_theme_config()["theme"]
    for mode in ("light", "dark"):
        text, bg = theme[mode]["textColor"], theme[mode]["backgroundColor"]
        caption = _composite(text, bg, 0.6)
        ratio = _contrast_ratio(caption, bg)
        assert ratio >= 4.5, f"{mode} caption {caption} on {bg} contrast {ratio:.2f}"


def test_code_background_matches_secondary_background() -> None:
    # The input text_area (secondaryBackgroundColor) and the output st.code
    # (codeBackgroundColor) sit side by side in one row; unpinned,
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


# -- spinner slots -------------------------------------------------------------


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
    # rather than matched in the compact form, so a substring negative check
    # cannot be fooled by a longer name that contains the slot's name. Any
    # DG's .empty() passes; exactly one assignment each.
    src = _APP_SOURCE
    param_match = re.search(r"^def ensure_model\((\w+)", src, re.M)
    assert param_match is not None
    receivers = set(re.findall(r"(?<!def )ensure_model\((\w+)\)", src))
    receivers |= set(re.findall(r"^\s*with (\w+)\.spinner\(", src, re.M))
    receivers -= {param_match.group(1)}
    assert receivers == {"warning_slot"}
    for name in receivers:
        callees = re.findall(rf"^\s*{name}\s*=\s*(\w+\.\w+)\(\s*\)", src, re.M)
        assert len(callees) == 1 and callees[0].endswith(".empty"), (name, callees)


# -- in-place rerun ------------------------------------------------------------


def test_translate_block_is_the_last_top_level_statement() -> None:
    # Both exits of the translate block call st.rerun() in place, which is
    # safe only because nothing after the block registers a keyed widget: a
    # RerunException aborts the run where it is raised, Streamlit's
    # stale-widget purge then drops every keyed widget that did not get
    # registered, and the setdefault block re-seeds it silently on the next
    # run. That is exactly what happened to the Document tab's pickers while
    # `with doc_tab:` ran after this block, and why a deferred _rerun_pending
    # flag existed until 2026-09-13. The UI test that caught it went with the
    # tab, so the precondition is pinned here instead: the
    # `if st.session_state._do_translate:` block must be the module's last
    # statement. Parsed rather than grepped, so a trailing comment or a
    # re-wrapped condition cannot fail it and a widget appended below it
    # cannot pass.
    last = ast.parse(_APP_SOURCE).body[-1]
    assert isinstance(last, ast.If), ast.dump(last)[:120]
    assert ast.unparse(last.test) == "st.session_state._do_translate"


# -- render_output -------------------------------------------------------------


def test_render_output_renders_code_with_panel_height() -> None:
    # render_output is the single sink for streamed and settled output; it
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


# -- download name -------------------------------------------------------------


def test_text_download_button_uses_the_recorded_name() -> None:
    # AppTest cannot see a download button's file_name (the proto carries only
    # a media-manager url), so guard the wiring here: the button must read the
    # captured name, never re-inline the old constant.
    assert "file_name=st.session_state.download_name," in _APP_SOURCE
    # Each name is written exactly once: the default at its constant, the
    # settled form in settled_download_name. Three reset paths used to inline
    # "translation.txt" and two settle paths the f-string, so changing one
    # left the others behind with nothing failing.
    assert _APP_SOURCE.count('"translation.txt"') == 1
    assert _APP_SOURCE.count("translation-") == 1
    assert "DEFAULT_DOWNLOAD_NAME" in _APP_SOURCE


def test_settled_download_name_uses_the_target_language() -> None:
    assert settled_download_name("French") == "translation-French.txt"
    # Bokmål is the one LANGUAGES entry that is not ASCII; it is still a
    # single word with no path separators, so nothing needs sanitising.
    assert settled_download_name("Bokmål") == "translation-Bokmål.txt"


# -- clip_to_input_cap ---------------------------------------------------------


def _utf16_units(text: str) -> int:
    # What the browser's maxlength and Streamlit's frontend guard count.
    return len(text.encode("utf-16-le")) // 2


def test_clip_to_input_cap_counts_utf16_units_like_maxlength() -> None:
    # An astral character is two UTF-16 units, so 20,000 of them are 40,000
    # units against a 30,000 cap: the cut lands at 15,000 characters, where
    # text[:30_000] would have kept all 20,000 and left the input over the
    # cap the browser enforces.
    clipped = clip_to_input_cap("\U0001f600" * 20_000, 30_000)
    assert len(clipped) == 15_000
    assert _utf16_units(clipped) == 30_000


def test_clip_to_input_cap_never_splits_a_character() -> None:
    # With one unit of room left and an astral character next, the cut
    # stops before it rather than emitting half of it.
    assert clip_to_input_cap("ab\U0001f600", 3) == "ab"
    assert clip_to_input_cap("ab\U0001f600", 4) == "ab\U0001f600"


def test_clip_to_input_cap_leaves_text_under_the_cap_alone() -> None:
    assert clip_to_input_cap("short", 30_000) == "short"
    assert clip_to_input_cap("x" * 30_000, 30_000) == "x" * 30_000
    assert clip_to_input_cap("x" * 30_001, 30_000) == "x" * 30_000
