# SPDX-License-Identifier: Apache-2.0
import os
import re
from collections.abc import Iterator
from typing import Any

# Mute transformers alias-warning spam triggered by Streamlit's module watcher.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# -- Config ------------------------------------------------------------------

MODEL_ID: str = "mlx-community/tiny-aya-global-8bit-mlx"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 8192
MAX_INPUT_TOKENS: int = 8192

# Chunk budget for document translation: well below MAX_INPUT_TOKENS to leave
# room for the per-chunk prompt wrapper (instruction + chat template).
MAX_CHUNK_TOKENS: int = 7000
# LiteParse handles PDFs and images natively. Office formats additionally need
# LibreOffice on PATH, so they are left out rather than failing at parse time.
# Both TIFF spellings are listed: this list reaches the browser as a file-type
# filter, scanners emit .tif as often as .tiff, and the parser reads either --
# omitting one greyed the file out in the picker with nothing to explain it.
DOCUMENT_TYPES: list[str] = ["pdf", "png", "jpg", "jpeg", "tiff", "tif", "webp"]
# Per-upload size cap, well under Streamlit's 200 MB server default. Sized to
# accept real documents and high-resolution scans while keeping a stray
# drag-and-drop from handing hundreds of megabytes to an in-process parser.
MAX_UPLOAD_MB: int = 50

# Shared UI constants reused across the Text and Document tabs.
PANEL_HEIGHT: int = 450
# Text-tab input cap. Reaches the browser as HTML maxlength, so it truncates
# silently -- the placeholder names it because nothing else can.
MAX_INPUT_CHARS: int = 30000
# Download names before anything has been translated. Each reset path uses
# these rather than its own literal, so the three of them cannot drift.
DEFAULT_TEXT_DOWNLOAD: str = "translation.txt"
DEFAULT_DOC_DOWNLOAD: str = "translation.md"
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


# Tesseract language codes, keyed by the LANGUAGES entry the user picks as the
# document's source. OCR defaults to English, so a Cyrillic, CJK or Arabic scan
# came back as plausible-looking Latin garbage with words silently dropped --
# non-empty, so it sailed past the "no translatable text" guard.
#
# Only languages Tesseract publishes traineddata for are listed. Hausa, Igbo,
# Malagasy, Shona, Wolof, Xhosa and Zulu have none, so they fall back to
# English rather than asking for a file that does not exist. Each language's
# data is a separate ~12-15 MB download on first use.
OCR_LANGUAGES: dict[str, str] = {
    "English": "eng",
    "Dutch": "nld",
    "French": "fra",
    "Italian": "ita",
    "Portuguese": "por",
    "Romanian": "ron",
    "Spanish": "spa",
    "Czech": "ces",
    "Polish": "pol",
    "Ukrainian": "ukr",
    "Russian": "rus",
    "Greek": "ell",
    "German": "deu",
    "Danish": "dan",
    "Swedish": "swe",
    "Bokmål": "nor",
    "Catalan": "cat",
    "Galician": "glg",
    "Welsh": "cym",
    "Irish": "gle",
    "Basque": "eus",
    "Croatian": "hrv",
    "Latvian": "lav",
    "Lithuanian": "lit",
    "Slovak": "slk",
    "Slovenian": "slv",
    "Estonian": "est",
    "Finnish": "fin",
    "Hungarian": "hun",
    "Serbian": "srp",
    "Bulgarian": "bul",
    "Arabic": "ara",
    "Persian": "fas",
    "Turkish": "tur",
    "Maltese": "mlt",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Marathi": "mar",
    "Bengali": "ben",
    "Gujarati": "guj",
    "Punjabi": "pan",
    "Tamil": "tam",
    "Telugu": "tel",
    "Nepali": "nep",
    "Urdu": "urd",
    "Tagalog": "fil",
    "Malay": "msa",
    "Indonesian": "ind",
    "Vietnamese": "vie",
    "Javanese": "jav",
    "Khmer": "khm",
    "Thai": "tha",
    "Lao": "lao",
    "Chinese": "chi_sim",
    "Burmese": "mya",
    "Japanese": "jpn",
    "Korean": "kor",
    "Amharic": "amh",
    "Yoruba": "yor",
    "Swahili": "swa",
}


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


# -- Document functions -------------------------------------------------------
# LiteParse parses to markdown but ships no chunker, so chunk_text below is the
# token-aware packer that splits a document into translatable pieces.

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
# Every ASCII punctuation character -- exactly the set CommonMark permits to be
# backslash-escaped, and deliberately wider than "the metacharacters one thinks
# of". st.caption renders more than CommonMark: Streamlit also handles LaTeX
# ($...$), emoji shortcodes (:tada:) and HTML entities (&amp;), so a filename
# like `budget $100 vs $200.pdf` or `photo:sunglasses:.pdf` still renders as
# markup under a CommonMark-only class. A backslash before ASCII punctuation is
# never displayed, so over-escaping costs nothing.
_MD_PUNCTUATION = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
_MD_SPECIAL_RE = re.compile(f"([{re.escape(_MD_PUNCTUATION)}])")
# Characters that would corrupt Streamlit's unquoted Content-Disposition
# header, or escape the download directory: quotes, backslashes, forward
# slashes and C0/DEL control characters.
_UNSAFE_FILENAME_RE = re.compile(r'[\x00-\x1f\x7f"\\/]')
# Sentence boundaries for Latin punctuation plus CJK full stops, since the app
# translates across all 67 languages. The two need different whitespace rules:
# Latin prose separates sentences with a space (and requiring one is what keeps
# "3.14" intact), while CJK writes 。！？ flush against the next sentence, so
# demanding whitespace there means CJK never splits on a sentence at all.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
# Fence markers, stripped when deciding whether a parse found any real text.
_FENCE_RE = re.compile(r"```[a-zA-Z]*")
# Most tokens a single character can cost under byte fallback: UTF-8 runs to
# four bytes, and a byte-fallback tokenizer spends one token per byte.
_MAX_CHAR_TOKENS = 4


def is_blank_markdown(text: str) -> bool:
    """True when a parse produced no readable content.

    A file LiteParse could not read comes back as an empty ```` ```text``` ````
    fence. That is not blank, so it would sail past the Document tab's "no
    translatable text" guard and be handed to the model as a prompt.
    """
    return not _FENCE_RE.sub("", text).strip()


def load_document_markdown(file_bytes: bytes, source_lang: str = "English") -> str:
    """Parse uploaded file bytes into a single markdown string via LiteParse.

    ``source_lang`` selects the OCR language for image uploads; it is ignored
    for PDFs, which never run OCR. Returns ``""`` when the file yielded no
    readable text, and lets LiteParse's ``ParseError`` propagate when an
    image's OCR cannot run at all (its training data is neither cached nor
    fetchable).
    """
    from liteparse import LiteParse

    # OCR is enabled only where it is the sole extraction path. An image has no
    # text layer, so reading it *requires* OCR -- forcing it off returned an
    # empty '```text```' fence for every image type in DOCUMENT_TYPES. A PDF
    # does carry a text layer, and LiteParse's auto mode would still reach for
    # OCR on sparse pages, which downloads ~15 MB of Tesseract training data
    # from GitHub on first use. This app's whole premise is that nothing leaves
    # the machine, so PDFs -- the common case -- stay strictly offline.
    # ocr_failure_fatal is left at LiteParse's default (True). Only images run
    # OCR here, and an image has no text layer to fall back to, so False could
    # only turn "could not fetch eng.traineddata" into an empty fence that the
    # tab reports as "No translatable text found"; raising lets the tab show
    # the real error instead.
    # quiet=True keeps LiteParse's timing lines out of the server's stderr.
    parser = LiteParse(
        output_format="markdown",
        quiet=True,
        ocr_enabled=not file_bytes.lstrip()[:4].startswith(b"%PDF"),
        ocr_language=OCR_LANGUAGES.get(source_lang, "eng"),
    )
    text = parser.parse(file_bytes).text
    return "" if is_blank_markdown(text) else text


def heading_line(block: str) -> str:
    """Return ``block``'s first line.

    Blocks are split on blank lines, so a markdown heading followed immediately
    by its opening body line arrives as one block. The heading trail keeps this
    line alone: storing the whole block would prepend that body text to every
    later chunk and charge its tokens against the budget the trail reserves.
    """
    return block.lstrip().split("\n", 1)[0].rstrip()


def heading_level(block: str) -> int:
    """Return the ATX markdown heading level of ``block``, or 0 if not a heading."""
    marker = heading_line(block).split(" ", 1)[0]
    return len(marker) if marker and set(marker) == {"#"} and len(marker) <= 6 else 0


def token_overhead(tokenizer: Any) -> int:
    """Count the special tokens ``encode`` prepends to every string.

    The model's tokenizer adds a BOS token to each ``encode`` call, so summing
    per-block counts would charge one BOS per block for a chunk that only ever
    carries one. Measuring the constant lets the packer work in net tokens.
    """
    return len(tokenizer.encode(""))


def count_tokens(text: str, tokenizer: Any) -> int:
    """Return the token length of ``text``, excluding the special-token prefix."""
    return len(tokenizer.encode(text)) - token_overhead(tokenizer)


def sentence_units(text: str) -> list[str]:
    """Split ``text`` after sentence enders, keeping each unit's trailing gap.

    Units carry the whitespace that followed them, so concatenating them back
    reproduces the source exactly -- CJK included, where the gap is empty and
    rejoining on a space would insert one the original never had.
    """
    units: list[str] = []
    start = 0
    for match in _SENTENCE_RE.finditer(text):
        if match.end() > start:
            units.append(text[start : match.end()])
            start = match.end()
    if start < len(text):
        units.append(text[start:])
    return units


def hard_windows(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Cut ``text`` into token windows -- the last resort when nothing else fits.

    Strips the BOS prefix before slicing, or ``decode`` would write the special
    token back into the chunk as visible text.

    Boundaries are pulled back until the window decodes cleanly. A byte-fallback
    tokenizer spends several tokens on one character in scripts outside its
    vocabulary -- Thai, Lao, Khmer, Burmese, Amharic and rarer CJK, all of them
    in LANGUAGES -- so a fixed stride can cut a character in half. The halves
    decode to U+FFFD and the character is lost outright, and the corrupted text
    is what would then be sent to the model as source.
    """
    # A window of zero tokens would never advance past its own start.
    max_tokens = max(max_tokens, 1)
    ids = tokenizer.encode(text)[token_overhead(tokenizer) :]
    windows: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        piece = tokenizer.decode(ids[start:end])
        # Give back at most a character's worth of tokens: past that the
        # replacement character is in the source itself, not an artefact of
        # the cut, and shrinking further would only strand tokens.
        floor = max(start + 1, end - _MAX_CHAR_TOKENS)
        while end > floor and "�" in piece:
            end -= 1
            piece = tokenizer.decode(ids[start:end])
        windows.append(piece)
        start = end
    return windows


def pack_by_estimate(
    units: list[str], tokenizer: Any, budget: int, join_cost: int = 0
) -> list[list[str]]:
    """Group ``units`` greedily using one token count per unit.

    Measuring every growing candidate instead is quadratic: on a single
    chunk-sized block it re-encodes the whole accumulating piece per unit and
    costs tens of seconds. Callers verify each finished group once, which keeps
    the guarantee while staying linear in the document length.

    ``join_cost`` is charged for every join after the first. Callers that glue
    units together with a separator must pass it: a thousand blocks joined by a
    blank line cost a thousand tokens the per-unit counts cannot see, and
    ignoring that overshot the budget by a quarter.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    running = 0
    for unit in units:
        size = count_tokens(unit, tokenizer)
        if current and running + size + join_cost > budget:
            groups.append(current)
            current, running = [], 0
        elif current:
            size += join_cost
        current.append(unit)
        running += size
    if current:
        groups.append(current)
    return groups


def split_oversized(block: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Split a single over-budget block into pieces that each fit ``max_tokens``.

    Sentences are re-packed up to the budget rather than emitted one per piece.
    Without that, a chunk overflowing by a single token exploded into one piece
    per sentence -- hundreds of separate translation calls, each stripped of the
    neighbours that gave it context.
    """
    units: list[str] = []
    for unit in sentence_units(block):
        if not unit.strip():
            continue
        if count_tokens(unit, tokenizer) <= max_tokens:
            units.append(unit)
            continue
        # A single sentence over budget: an unpunctuated wall of text, or a
        # script whose sentence enders this pattern does not know.
        windows = hard_windows(unit, tokenizer, max_tokens)
        # Windows are cut mid-text, so they carry no trailing gap of their own.
        units.extend(w + " " for w in windows[:-1])
        units.extend(windows[-1:])

    pieces: list[str] = []
    for group in pack_by_estimate(units, tokenizer, max_tokens):
        text = "".join(group).strip()
        if not text:
            continue
        measured = count_tokens(text, tokenizer)
        if measured <= max_tokens:
            pieces.append(text)
            continue
        # The estimate drifted (a tokenizer merges across unit boundaries).
        # Re-pack this group alone against a budget cut by the overshoot.
        for sub in pack_by_estimate(
            group, tokenizer, max(2 * max_tokens - measured, 1)
        ):
            sub_text = "".join(sub).strip()
            if count_tokens(sub_text, tokenizer) <= max_tokens:
                pieces.append(sub_text)
            else:
                pieces.extend(
                    w.strip()
                    for w in hard_windows(sub_text, tokenizer, max_tokens)
                    if w.strip()
                )
    return pieces


def leading_headings(blocks: list[str]) -> list[str]:
    """Return the headings ``blocks`` opens with, before its first body block.

    Only these make prepended context redundant. A heading further in is the
    *next* section starting inside the chunk, so the body ahead of it still
    needs its own section's heading prepended.
    """
    found: list[str] = []
    for block in blocks:
        if not heading_level(block):
            break
        found.append(heading_line(block))
    return found


def repack_blocks(
    headings: list[str], blocks: list[str], tokenizer: Any, max_tokens: int
) -> list[str]:
    """Re-pack ``blocks`` into chunks that each carry their own heading context.

    The greedy packer estimates a chunk by summing its blocks, but a BPE
    tokenizer merges across block boundaries, so the finished chunk can measure
    a token or two over. Re-packing whole blocks keeps the blank lines between
    them -- splitting the assembled body on sentences instead would dissolve
    every paragraph, table and list in it.

    ``headings`` is the full trail, not the caller's already-deduplicated
    context: a piece cut from the middle of a chunk contains none of the
    headings the first piece opened with, so it has to have them prepended or
    it reaches the model as an unlabelled fragment.
    """

    # A chunk that opens with its own headings carries them as blocks, not in
    # the trail -- the trail was still empty when it opened. Pieces cut from
    # its middle contain neither, so those headings have to join the context or
    # every piece after the first reaches the model unlabelled.
    context = [*headings, *(h for h in leading_headings(blocks) if h not in headings)]

    def assemble(body: list[str]) -> str:
        own = leading_headings(body)
        return "\n\n".join([*[h for h in context if h not in own], *body])

    separator = count_tokens("\n\n", tokenizer)
    overhead = sum(count_tokens(h, tokenizer) + separator for h in context)
    # Each join costs the blank line plus about a token of merge drift the
    # per-block counts cannot see.
    join_cost = separator + 1
    budget = max(max_tokens - overhead, 1)

    packed: list[str] = []
    for group in pack_by_estimate(blocks, tokenizer, budget, join_cost):
        text = assemble(group)
        measured = count_tokens(text, tokenizer)
        if measured <= max_tokens:
            packed.append(text)
            continue
        # The estimate still drifted over. Re-pack this group alone against a
        # budget cut by the overshoot, assembling each sub-group so it keeps
        # its heading context -- falling straight through to sentence
        # splitting here is what stripped the context off middle pieces.
        tighter = max(budget - (measured - max_tokens), 1)
        for sub in pack_by_estimate(group, tokenizer, tighter, join_cost):
            sub_text = assemble(sub)
            if count_tokens(sub_text, tokenizer) <= max_tokens:
                packed.append(sub_text)
            else:
                packed.extend(split_oversized(sub_text, tokenizer, max_tokens))
    return packed


def chunk_text(
    text: str, tokenizer: Any, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[str]:
    """Pack markdown paragraphs into chunks that stay under ``max_tokens``.

    LiteParse returns one markdown string and no chunker, so this is where a
    document becomes translatable pieces. Blocks are packed greedily so each
    chunk carries as much as it can hold, paragraphs are never split unless one
    alone exceeds the budget, and the enclosing markdown headings are prepended
    to chunks that do not already open with them -- context that matters because
    every chunk is translated as an independent prompt with no memory of its
    neighbours.
    """
    # A non-positive budget would divide the document by zero-width windows.
    max_tokens = max(max_tokens, 1)
    blocks = [b.strip() for b in _BLANK_LINE_RE.split(text) if b.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    # Blocks are joined with a blank line, so every join after the first costs
    # a separator. Charging it keeps the running total honest instead of
    # drifting over budget by one token per block.
    separator = count_tokens("\n\n", tokenizer)
    # ``trail`` advances as blocks are read; ``pending`` is the snapshot taken
    # when the open chunk started. They differ once a heading is read into a
    # chunk that is still filling, and prepending ``trail`` at that point would
    # label a chunk with the heading of the section that comes *after* it.
    trail: dict[int, str] = {}
    trail_tokens = 0
    pending: dict[int, str] = {}
    pending_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens, pending, pending_tokens
        if not current:
            return
        # Skip only the headings the chunk *opens* with -- those alone make the
        # prepended context redundant. A heading deeper in the chunk starts the
        # next section, and the body ahead of it still needs its own label.
        headings = [pending[lv] for lv in sorted(pending)]
        opening = leading_headings(current)
        context = [h for h in headings if h not in opening]
        body = "\n\n".join([*context, *current])
        # Packing sums per-block counts, but a tokenizer merges across block
        # boundaries, so the assembled chunk can run a little over. Re-pack the
        # blocks against measured sizes rather than splitting the body, which
        # would shatter a full chunk into one prompt per sentence.
        if count_tokens(body, tokenizer) > max_tokens:
            chunks.extend(repack_blocks(headings, current, tokenizer, max_tokens))
        else:
            chunks.append(body)
        current = []
        current_tokens = 0
        pending = {}
        pending_tokens = 0

    in_fence = False
    for block in blocks:
        # Reserve room for whichever context is larger: the one already
        # committed to the open chunk, or the one a chunk opening now would
        # take. Either is possible depending on where this block lands.
        room = max_tokens - max(trail_tokens, pending_tokens)
        if room < max_tokens // 4:
            # The heading trail is eating the budget. Clamping to a token
            # instead shreds the document: a trail of 7,002 tokens against a
            # 7,000 budget turned 13,121 tokens of markdown into 12,082 chunks
            # and 292 seconds of packing. A chunk carrying less context still
            # translates; a chunk of one token does not.
            trail, trail_tokens = {}, 0
            pending, pending_tokens = {}, 0
            room = max_tokens
        budget = max(room, 1)
        n = count_tokens(block, tokenizer)
        pieces = [block] if n <= budget else split_oversized(block, tokenizer, budget)
        for piece in pieces:
            size = n if piece is block else count_tokens(piece, tokenizer)
            if current:
                size += separator
            if current and current_tokens + size > budget:
                flush()
                size -= separator
            if not current:
                pending, pending_tokens = dict(trail), trail_tokens
            current.append(piece)
            current_tokens += size

        # A '#' line inside a fenced code block is a shell comment, not a
        # heading. Blocks split on blank lines, so a fence containing one
        # arrives as several blocks and the marker count has to be carried
        # across them -- otherwise the comment evicts the real heading and
        # every chunk below it is labelled with a line of shell.
        if not in_fence and (level := heading_level(block)):
            # A deeper heading replaces its siblings; shallower ones survive.
            trail = {lv: h for lv, h in trail.items() if lv < level}
            trail[level] = heading_line(block)
            # Each context heading is followed by a separator when rendered.
            trail_tokens = sum(
                count_tokens(h, tokenizer) + separator for h in trail.values()
            )
        in_fence ^= block.count("```") % 2 == 1

    flush()
    return chunks


def translate_document(
    chunks: list[str],
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
) -> Iterator[tuple[int, str]]:
    """Translate each chunk; yield ``(index, cumulative_text)`` per token."""
    done: list[str] = []
    for i, chunk in enumerate(chunks):
        prompt_ids = tokenize_prompt(chunk, source_lang, target_lang, tokenizer)
        if len(prompt_ids) > MAX_INPUT_TOKENS:
            # Skip rather than abort the whole document on one oversized chunk.
            done.append("[Section skipped: too long to translate.]")
            yield i, "\n\n".join(p for p in done if p)
            continue
        partial = ""
        for partial in stream_translate(prompt_ids, model, tokenizer):
            yield i, "\n\n".join(p for p in [*done, partial] if p)
        done.append(partial)


def escape_markdown(text: str) -> str:
    """Backslash-escape GFM metacharacters so ``text`` renders literally.

    `st.caption` and `st.markdown` parse GitHub-flavored Markdown, so an
    upload named ``*draft*.pdf`` renders as italic ``draft.pdf`` and
    ``[report](https://example.com).pdf`` renders as a live link -- in the
    caption whose entire job is to state the filename accurately. CommonMark
    lets any ASCII punctuation be backslash-escaped, and the escape itself is
    not shown.
    """
    return _MD_SPECIAL_RE.sub(r"\\\1", text)


def safe_download_stem(file_name: str) -> str:
    """Strip characters that would corrupt a Content-Disposition header.

    Streamlit builds the header as ``f'filename="{filename}"'`` with no
    quoting (``starlette_routes.py``), so an upload named ``re"port.pdf``
    closes the quoted string early and the browser saves ``re``. Path
    separators and control characters are replaced -- with ``_``, not removed,
    which is why a non-empty stem can never sanitise away.

    Non-ASCII is deliberately kept. Note the branch it actually takes:
    Streamlit selects on ``filename.encode("latin1")``, **not** on ASCII, so
    ``Bokmål`` goes down the *quoted* branch as a raw 0xE5 byte (RFC 6266 reads
    an unencoded filename as ISO-8859-1) and only genuinely non-latin1 names
    like 報告 reach the percent-encoded ``filename*=utf-8''`` branch. Both are
    handled; neither needs stripping.
    """
    return _UNSAFE_FILENAME_RE.sub("_", file_name)


def document_meta_line(file_name: str, source_lang: str, target_lang: str) -> str:
    """One-line provenance for a settled document translation.

    The filename is escaped because the caption renders Markdown; the
    language names are not, since all 67 are plain letters.
    """
    return f"{escape_markdown(file_name)} · {source_lang} → {target_lang}"


def document_download_name(file_name: str, target_lang: str) -> str:
    """Name the download after the source file and the target language.

    Every output was previously ``translation.md``, so translating several
    documents left ``translation.md``, ``translation (1).md`` ... with nothing
    to tell them apart. All 67 language names are space- and separator-free,
    and a dotless or dot-leading upload still yields a usable stem. The stem
    comes from the *upload*, i.e. from the browser, so it is sanitised --
    see ``safe_download_stem``.
    """
    stem = file_name.rsplit(".", 1)[0] if "." in file_name[1:] else file_name
    return f"{safe_download_stem(stem) or 'translation'}-{target_lang}.md"


import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Tiny Aya Translate",
    page_icon=":material/translate:",
    layout="wide",
)


@st.cache_resource
def load_model() -> tuple[Any, Any]:
    """Load model and tokenizer once, cached for the session lifetime."""
    from mlx_lm import load

    loaded = load(MODEL_ID)
    return loaded[0], loaded[1]


@st.cache_data(max_entries=8, show_spinner=False)
def cached_document_chunks(
    file_bytes: bytes,
    source_lang: str = "English",
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """Parse + chunk an uploaded document, cached by file bytes and source
    language.

    Re-translating the same upload into another *target* language then skips
    the parse + chunk. The source language is part of the key because it picks
    the OCR language, so it changes the parsed text. LiteParse sniffs the
    format itself, so the filename never reaches the parser. The tokenizer is
    fetched from the cached ``load_model()`` rather than taken as an argument,
    since it is unhashable and would defeat ``@st.cache_data``'s argument
    hashing.
    """
    _, tokenizer = load_model()
    markdown = load_document_markdown(file_bytes, source_lang)
    return chunk_text(markdown, tokenizer, max_tokens)


def render_output(placeholder: Any, text: str) -> None:
    """Render streamed translation output into ``placeholder``.

    Uses ``st.code`` rather than ``st.text_area`` so the placeholder can be
    replaced mid-script without colliding with a widget's auto-generated
    element id.
    """
    placeholder.code(text, language=None, wrap_lines=True, height=PANEL_HEIGHT)


def record_document_provenance(
    slot: Any, file_name: str, source: str, target: str
) -> None:
    """Stamp the settled document output with what produced it.

    Captured at translation time, not read from the widgets at render time:
    the pickers keep moving after a translation settles, so a caption built
    later would describe the controls rather than the text under it. Writes
    the caption into ``slot`` as well as session state, because the Document
    tab has no ``st.rerun()`` -- the slot above the output has already been
    passed by the time a translation finishes.

    All 67 language names are ASCII letters only (bar ``Bokmål``), with no
    spaces or path separators, so the download stem needs no sanitising.
    """
    st.session_state.doc_meta = document_meta_line(file_name, source, target)
    st.session_state.doc_download_name = document_download_name(file_name, target)
    slot.caption(st.session_state.doc_meta)


def ensure_model(warning_container: Any) -> tuple[Any, Any] | None:
    """Load the model on demand, reporting into ``warning_container``.

    Called from the translate handlers rather than at page load. The weights are
    ~3.6 GB, so loading them before the tabs render left the user watching a
    spinner with nothing to type into or pick languages from.
    ``@st.cache_resource`` is process-global -- shared across every session and
    user, not per-session -- so the load happens once per server process, at the
    first translation rather than at startup.

    The spinner is rendered into ``warning_container`` rather than at the script
    cursor. Inside a translate handler that cursor sits below the panels and the
    controls row, so a bare ``st.spinner`` put the only sign of activity below
    the fold while the error landed at the top of the tab.

    Returns ``None`` when the load fails, having already written the error into
    that same slot. There is no cheap way to know the model loads without
    loading it, so the translate buttons are no longer pre-emptively disabled; a
    broken install surfaces on click, next to the action that triggered it.
    ``@st.cache_resource`` does not memoize exceptions, so a later click retries
    rather than being dead for the rest of the session.
    """
    try:
        # The first load on a fresh machine pulls ~3.6 GB, which is minutes, not
        # seconds; an unqualified "Loading model..." with no elapsed time is
        # indistinguishable from a hang. show_time and the size both come from
        # the README's own promise about that first click.
        with warning_container.spinner(
            "Loading model... the first run downloads ~3.6 GB from Hugging Face",
            show_time=True,
        ):
            return load_model()
    except Exception as e:
        # Keep the "Failed to load model: " prefix -- a UI test matches on it.
        warning_container.error(
            f"Failed to load model: {e}\n\nThe first run downloads ~3.6 GB from "
            "Hugging Face. Check your connection and disk space, then click "
            "Translate again."
        )
        return None


# -- Main page ----------------------------------------------------------------

st.title("Tiny Aya Translate")
# "sent to a server", not "sent anywhere": the model-load spinner and the
# Document tab's OCR notice both announce downloads, so an absolute claim
# would read as a contradiction two clicks later. Outbound is the property
# a user is actually deciding about when they paste text in.
st.caption("67 languages, translated on your Mac — nothing is sent to a server.")

# -- Session state defaults ---------------------------------------------------

st.session_state.setdefault("source_lang", "English")
st.session_state.setdefault("target_lang", "French")
st.session_state.setdefault("translate_input", "")
st.session_state.setdefault("translate_output", "")
st.session_state.setdefault("download_name", DEFAULT_TEXT_DOWNLOAD)
# ("warning"|"error", text), or "" -- a message raised by the translate
# block, which runs *below* the panels and then reruns. Without this the
# rerun that repaints the output would also discard the explanation of it.
st.session_state.setdefault("translate_notice", "")
# Set by the Text translate block, actioned at the very bottom of the
# script -- see the deferred-rerun block there for why it cannot rerun
# in place.
st.session_state.setdefault("_rerun_pending", False)
st.session_state.setdefault("_do_translate", False)
st.session_state.setdefault("doc_source_lang", "English")
st.session_state.setdefault("doc_target_lang", "French")
st.session_state.setdefault("doc_output", "")
# Provenance for the document output, captured at translation time rather than
# read from the widgets at render time: the pickers keep moving after a
# translation settles, so reading them later would describe the controls
# instead of the text on screen.
st.session_state.setdefault("doc_meta", "")
st.session_state.setdefault("doc_download_name", DEFAULT_DOC_DOWNLOAD)


def request_translate() -> None:
    """Flag that a translation was requested (processed after controls row)."""
    st.session_state._do_translate = True


def clear_doc_output() -> None:
    """Drop the previous file's translation when the upload changes.

    ``st.session_state.doc_output`` outlives the upload it came from, so
    without this a newly uploaded file renders -- and offers for download as
    ``translation.md`` -- the *previous* file's translation, with nothing on
    screen tying the output to the filename above it. It also defeated the
    "No translatable text found" guard: a blank upload showed that warning
    directly above the last document's intact output. Also fires when the
    file is removed, which is the same invalidation.
    """
    st.session_state.doc_output = ""
    st.session_state.doc_meta = ""
    st.session_state.doc_download_name = DEFAULT_DOC_DOWNLOAD


def clear_translate_output() -> None:
    """Drop a settled translation when either language picker moves.

    The settled output is an unlabelled ``st.code`` block -- no label, no
    caption, no placeholder -- so nothing on screen names the pair that
    produced it. Change To from French to German and the card above asserts
    English -> German while the panel below still holds French, with Download
    still offering it.

    The Document tab answers the same question the other way: it keeps a
    settled output across a language change and discloses the real pair in
    ``doc_meta``. That asymmetry is deliberate -- the Text tab has no
    provenance surface to disclose into, and adding one would push the right
    panel out of line with the left.

    Only the output goes; ``translate_input`` is left alone so re-pressing
    Translate is one click. ``swap_languages`` needs no hook of its own -- it
    already clears the output, and Streamlit does not fire ``on_change`` for a
    programmatic session-state write.
    """
    st.session_state.translate_output = ""
    st.session_state.download_name = DEFAULT_TEXT_DOWNLOAD


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


def swap_doc_languages() -> None:
    """Swap the Document tab's source and target languages.

    Languages only, unlike ``swap_languages``: the Text tab's swap also moves
    the output into the input, but a document's source is an uploaded file,
    so there is nothing to move it into. ``doc_output`` is deliberately left
    alone -- the upload has not changed, and ``clear_doc_output`` owns that
    invalidation.
    """
    st.session_state.doc_source_lang, st.session_state.doc_target_lang = (
        st.session_state.doc_target_lang,
        st.session_state.doc_source_lang,
    )


text_tab, doc_tab = st.tabs(
    [":material/text_fields: Text", ":material/description: Document"]
)

with text_tab:
    # -- Language bar ---------------------------------------------------------

    with st.container(border=True):
        col_from, col_swap, col_to = st.columns(
            [10, 1, 10], vertical_alignment="center"
        )
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

    # -- Warning slot (above panels) ------------------------------------------

    warning_slot = st.container()
    # Drain any notice left by the previous run's translate block. Read once
    # and cleared, so it survives exactly the one rerun it was raised for.
    if st.session_state.translate_notice:
        _level, _text = st.session_state.translate_notice
        st.session_state.translate_notice = ""
        if _level == "error":
            warning_slot.error(_text)
        else:
            warning_slot.warning(_text)

    # -- Side-by-side text panels ---------------------------------------------

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
            placeholder=(
                f"Enter text to translate (up to {MAX_INPUT_CHARS:,} characters)"
            ),
            key="translate_input",
            label_visibility="collapsed",
        )
    with col_output:
        output_placeholder = st.empty()
        # Settled output goes through render_output -- the same st.code sink the
        # streaming path already uses, and the one the Document tab uses for
        # both states. Previously the panel swapped to a disabled text_area the
        # instant streaming ended, changing font, weight and colour in one
        # frame: Streamlit paints disabled content at fadedText40, measured
        # 2.17:1 light and 3.53:1 dark, so the finished translation was *less*
        # legible than the placeholder that preceded it, and a disabled
        # textarea is unselectable, so it could not even be copied.
        # The text_area survives as the empty state only, where it supplies the
        # placeholder and balances the input panel opposite it.
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

    # -- Controls row ---------------------------------------------------------

    # st.columns, not st.container(horizontal=True): the horizontal container is
    # a flex row whose children are `flex: 1 1 fit-content`, so a stretched button
    # grows from its *intrinsic* width rather than splitting the row evenly.
    # Columns are a proportional grid, which is what mirroring the panels needs.
    # The reasoning is structural and still holds; the numbers that once backed
    # it do NOT. They were measured under the old theme's baseFontSize = 14
    # (Translate 460.6px vs Download 465.4px against 461/461 panels, a 14px gap
    # against the panels' 16px), and the theme now inherits Streamlit's 16px
    # base, which moves every intrinsic width and gap. Remeasure before quoting.
    sub_translate, sub_download = st.columns(
        2, vertical_alignment="center", gap="small"
    )
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
            # "rerun" -- which re-executes both tab bodies for nothing.
            on_click="ignore",
            disabled=not st.session_state.translate_output.strip(),
            type="secondary",
            width="stretch",
        )

    # -- Process translation request (below controls) -------------------------

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
                    # the fold -- the trap ensure_model and doc_status_slot
                    # already document. warning_slot is above the fold but
                    # sits above the panels, so it would push the whole
                    # side-by-side row down for the duration and snap it back.
                    # A skeleton at PANEL_HEIGHT reflows nothing: it occupies
                    # the box render_output is about to. It also clears the
                    # *previous* translation, which used to sit there looking
                    # current until the first new token overwrote it.
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
                        # instead of the bytes. Same reasoning as
                        # record_document_provenance.
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
                        st.session_state.download_name = DEFAULT_TEXT_DOWNLOAD
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
                    # again. Deferred, not immediate -- see the bottom of the
                    # script.
                    st.session_state._rerun_pending = True
            except Exception as e:
                # Keep whatever streamed before the failure, as the Document tab
                # already does; `partial` is "" if it raised before the first
                # token or before the stream even started.
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
                    st.session_state.download_name = DEFAULT_TEXT_DOWNLOAD
                    st.session_state.translate_notice = (
                        "error",
                        f"Translation failed: {e}",
                    )
                st.session_state._rerun_pending = True

with doc_tab:
    # -- Source: languages + upload -------------------------------------------

    with st.container(border=True):
        # Same [10, 1, 10] language row as the Text tab -- that row is the
        # shared component, not the whole card. This card carries a second row
        # because a document's source is a file, not a panel: describing the
        # source is one operation, so the pickers and the dropzone are grouped
        # rather than left as separate full-width siblings. Reversing a pair
        # here previously meant retyping both languages into 67-item
        # dropdowns, while the identical operation one tab over was a click.
        doc_col_from, doc_col_swap, doc_col_to = st.columns(
            [10, 1, 10], vertical_alignment="center"
        )
        with doc_col_from:
            st.selectbox(
                "From",
                LANGUAGES,
                key="doc_source_lang",
                label_visibility="collapsed",
            )
        # Bounded and centred for the reason spelled out at the Text tab's
        # swap. Same treatment because the two rows are one component, though
        # a stray tap is recoverable here -- swap_doc_languages only flips the
        # pickers.
        with doc_col_swap.container(horizontal=True, horizontal_alignment="center"):
            st.button(
                "",
                key="swap_doc",
                icon=":material/swap_horiz:",
                on_click=swap_doc_languages,
                width=40,
                type="tertiary",
                help="Swap languages",
            )
        with doc_col_to:
            st.selectbox(
                "To",
                LANGUAGES,
                key="doc_target_lang",
                label_visibility="collapsed",
            )

        # max_upload_size caps this widget alone; Streamlit's server default is
        # 200 MB, and an upload that size goes straight into in-process PDFium
        # / Tesseract while the ~3.6 GB of weights are already resident.
        #
        # It is a FRONTEND hint, not a server guarantee: the reject in
        # starlette_routes.py still reads server.maxUploadSize, so a POST
        # straight to /_stcore/upload_file is unaffected. That is acceptable
        # here -- this app binds to localhost for one user -- but it is a
        # browser-path mitigation, not a limit.
        uploaded = st.file_uploader(
            "Upload a document",
            type=DOCUMENT_TYPES,
            key="doc_upload",
            on_change=clear_doc_output,
            max_upload_size=MAX_UPLOAD_MB,
            label_visibility="collapsed",
        )
        # Scans have no text layer, so the From picker doubles as the OCR
        # language (see OCR_LANGUAGES). Getting it wrong does not fail loudly
        # -- Tesseract returns confident garbage that sails past the "no
        # translatable text" guard -- so the coupling has to be stated.
        #
        # st.info, not st.caption, and the reason is contrast, not emphasis.
        # st.caption renders as opacity 0.6 on inherited body text -- `color:
        # inherit` plus `opacity`, never reading the grayTextColor key, which
        # exists and holds the same value -- landing textColor #31333f on
        # #ffffff at 3.69:1 in light, under AA's 4.5:1 for 14px. Nothing in
        # config.toml reaches it: a markdown color directive is multiplied by
        # the same opacity and comes out worse. st.info paints blueTextColor
        # on blueBackgroundColor, 6.68:1 light and 4.90:1 dark, and design.md
        # files instructions under info and metadata under caption. This is an
        # instruction with a silent-failure mode and a network side effect.
        st.info(
            "Scans are read with OCR in the source language — the first scan "
            "in a new language downloads ~15 MB of OCR data.",
            icon=":material/language:",
        )

    # -- Controls -------------------------------------------------------------

    # Same 50/50 grid as the Text tab's controls row. The download used to sit
    # alone below the output panel purely because the script cursor was there
    # when its value became known; a reserved container decouples the two, so
    # the produce and save actions read as one pair.
    doc_translate_col, doc_download_col = st.columns(
        2, vertical_alignment="center", gap="small"
    )
    with doc_translate_col:
        translate_doc_clicked = st.button(
            "Translate document",
            key="translate_doc",
            icon=":material/translate:",
            disabled=uploaded is None,
            type="primary",
            width="stretch",
        )
    doc_download_slot = doc_download_col.container()

    # -- Warning slot + streamed output ---------------------------------------

    doc_warning_slot = st.container()
    # Activity slot, separate from the warning slot and declared above the
    # output for the same reason ensure_model's spinner is: once the first
    # chunk lands, the output panel is PANEL_HEIGHT tall, so anything created
    # at the script cursor below it spends the whole run off the fold.
    doc_status_slot = st.container()
    doc_meta_slot = st.empty()
    doc_output_placeholder = st.empty()
    if st.session_state.doc_output:
        # Which file, and which direction. The source is an off-screen filename
        # chip, so unlike the Text tab's side-by-side panels there is nothing
        # on screen that would make a stale output obvious.
        if st.session_state.doc_meta:
            doc_meta_slot.caption(st.session_state.doc_meta)
        render_output(doc_output_placeholder, st.session_state.doc_output)
    else:
        # Empty state, not dead space. Now that Download sits beside Translate
        # rather than below the output, at rest the tab ended in two disabled
        # buttons with nothing after them -- readable as a converter that only
        # ever hands back a file. This panel says the translation appears here
        # too, and reserves the exact footprint render_output will fill, so the
        # first chunk lands in place instead of shoving PANEL_HEIGHT of page
        # into existence. Mirrors the Text tab's empty state; the distinct
        # label keeps the two auto-generated element ids apart.
        doc_output_placeholder.text_area(
            "Translated document",
            height=PANEL_HEIGHT,
            placeholder="Translated document appears here",
            disabled=True,
            value="",
            label_visibility="collapsed",
        )

    def _restore_doc_meta() -> None:
        """Put the provenance caption back when a run changed nothing.

        The caption is cleared the moment Translate is committed, so every
        path that leaves the *previous* output on screen has to restore the
        line that describes it.
        """
        if st.session_state.doc_meta:
            doc_meta_slot.caption(st.session_state.doc_meta)

    # -- Process document translation -----------------------------------------

    if translate_doc_clicked and uploaded is not None:
        # Committed to a translation, so the caption above the panel stops
        # describing the old one now rather than when the new one lands. The
        # model load and parse can take minutes, and the caption exists to say
        # which file and pair produced the text below it -- left up, it spent
        # that whole window asserting a pair the controls no longer matched
        # while new text streamed in underneath. Restored from state on the
        # failure path below, where nothing actually changed.
        doc_meta_slot.empty()
        if st.session_state.doc_source_lang == st.session_state.doc_target_lang:
            doc_warning_slot.warning(SAME_LANGUAGE_WARNING)
            _restore_doc_meta()
        # ensure_model's spinner is activity, so it goes to doc_status_slot --
        # the same contract the "Reading document..." spinner below follows. It
        # is the longest-lived indicator on the tab: minutes on a cold cache.
        # A failed load has already reported itself into that slot.
        elif (loaded := ensure_model(doc_status_slot)) is not None:
            model, tokenizer = loaded
            result = ""
            try:
                with doc_status_slot.spinner("Reading document..."):
                    chunks = cached_document_chunks(
                        uploaded.getvalue(), st.session_state.doc_source_lang
                    )
                if not chunks:
                    doc_warning_slot.warning(
                        "No translatable text found in the document."
                    )
                    _restore_doc_meta()
                else:
                    last_rendered = -1
                    # A short PDF is one chunk, which is the common case, so
                    # the plural cannot be hardcoded into the status label.
                    n_sections = len(chunks)
                    sections = "section" if n_sections == 1 else "sections"
                    # st.status rather than a progress bar plus a caption: it
                    # reports "running" with a spinner instead of a fraction,
                    # and the fraction was the wrong shape here. Chunks
                    # complete, so the only honest value is idx/len(chunks) --
                    # which sits at 0% for the whole of the first chunk, and
                    # therefore for the entire run of a single-chunk document.
                    # Its context manager also settles the state itself:
                    # "complete" on a clean exit, "error" when an exception
                    # propagates. The pair it replaces did neither on the
                    # failure path, stranding a half-filled bar and a
                    # "Translating section k of n" caption under the error.
                    # type="compact" because the body is empty: render_output
                    # writes into doc_output_placeholder, an st.empty() created
                    # above this block, so nothing nests inside the status. The
                    # default type would be a bordered expander over nothing.
                    with doc_status_slot.status(
                        f"Translating {n_sections} {sections}", type="compact"
                    ) as doc_status:
                        for idx, cumulative in translate_document(
                            chunks,
                            st.session_state.doc_source_lang,
                            st.session_state.doc_target_lang,
                            model,
                            tokenizer,
                        ):
                            result = cumulative
                            # Everything in here is chunk-scoped, but
                            # translate_document yields once per *token*. Both
                            # the label and the rendered output change only
                            # when idx does, so re-emitting them per token
                            # bought two identical deltas a token: measured at
                            # 1.15 s of server work per 15k tokens against
                            # 1.5 ms guarded. Re-sending the whole growing
                            # document every token is O(n²) besides.
                            if idx != last_rendered:
                                doc_status.update(
                                    label=f"Translating section {idx + 1} "
                                    f"of {n_sections}"
                                )
                                render_output(doc_output_placeholder, result)
                                last_rendered = idx
                        render_output(doc_output_placeholder, result)
                        doc_status.update(label=f"Translated {n_sections} {sections}")
                    if result.strip():
                        st.session_state.doc_output = result
                        record_document_provenance(
                            doc_meta_slot,
                            uploaded.name,
                            st.session_state.doc_source_lang,
                            st.session_state.doc_target_lang,
                        )
                    else:
                        doc_warning_slot.warning(NO_OUTPUT_WARNING)
                        _restore_doc_meta()
            except Exception as e:
                if result.strip():
                    st.session_state.doc_output = result
                    # A partial result is still downloadable, so it still needs
                    # to say which file and pair it came from.
                    record_document_provenance(
                        doc_meta_slot,
                        uploaded.name,
                        st.session_state.doc_source_lang,
                        st.session_state.doc_target_lang,
                    )
                    doc_warning_slot.error(
                        f"Translation failed after partial output: {e}"
                    )
                else:
                    doc_warning_slot.error(f"Translation failed: {e}")
                    _restore_doc_meta()
        else:
            # ensure_model returned None and has already reported it. Nothing
            # changed, so the caption cleared above the chain has to come back
            # -- this was the one exit that fell through without restoring it,
            # leaving the previous translation on screen unlabelled.
            _restore_doc_meta()

    # -- Download -------------------------------------------------------------

    # Emitted last, as before, so it picks up a doc_output set by the translate
    # block above -- but into the slot reserved beside Translate, so it renders
    # up in the controls row rather than below the output panel.
    doc_download_slot.download_button(
        "Download",
        key="download_doc",
        icon=":material/download:",
        data=st.session_state.doc_output,
        file_name=st.session_state.doc_download_name,
        mime="text/markdown",
        on_click="ignore",
        disabled=not st.session_state.doc_output.strip(),
        type="secondary",
        width="stretch",
    )


# -- Deferred rerun ------------------------------------------------------------

# The Text tab's translate block needs a rerun to repaint its panel and refresh
# the download button, but it must not take it in place. `with doc_tab:` runs
# after `with text_tab:`, so a rerun raised inside the Text tab aborts the
# script before the Document tab registers any widget -- and Streamlit still
# runs its stale-widget purge, because RerunException leaves premature_stop
# False. The Document tab's keyed widgets are then dropped and the setdefault
# block above silently re-seeds doc_source_lang / doc_target_lang to
# English/French, so the next document translation runs the wrong pair with
# nothing on screen to say so. A pending upload goes the same way.
#
# This is the risk CLAUDE.md already records against `st.tabs(on_change="rerun")`
# -- it just also applied to the rerun that was already here.
if st.session_state._rerun_pending:
    st.session_state._rerun_pending = False
    st.rerun()
