from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

CANONICALIZATION_VERSION = "1.0"

FIELD_BORROWER_NAME = "borrower_name"
FIELD_PROPERTY_ADDRESS = "property_address"
FIELD_LOAN_NUMBER = "loan_number"
FIELD_PAGE_NUMBER = "page_number"

SUPPORTED_FIELDS: tuple[str, ...] = (
    FIELD_BORROWER_NAME,
    FIELD_PROPERTY_ADDRESS,
    FIELD_LOAN_NUMBER,
    FIELD_PAGE_NUMBER,
)

FIELD_SIMILARITY_THRESHOLDS: Mapping[str, float] = {
    FIELD_BORROWER_NAME: 0.85,
    FIELD_PROPERTY_ADDRESS: 0.90,
    FIELD_LOAN_NUMBER: 1.0,
    FIELD_PAGE_NUMBER: 1.0,
}

CLUSTER_SIMILARITY_THRESHOLDS: Mapping[str, float] = {
    FIELD_BORROWER_NAME: 1.0,
    FIELD_PROPERTY_ADDRESS: 0.90,
    FIELD_LOAN_NUMBER: 1.0,
    FIELD_PAGE_NUMBER: 1.0,
}

USPS_DESIGNATORS: Mapping[str, str] = {
    "street": "st",
    "avenue": "ave",
    "drive": "dr",
    "road": "rd",
    "boulevard": "blvd",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "place": "pl",
    "terrace": "ter",
    "parkway": "pkwy",
    "highway": "hwy",
    "trail": "trl",
    "plaza": "plz",
    "square": "sq",
    "expressway": "expy",
    "freeway": "fwy",
    "turnpike": "tpke",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "floor": "fl",
    "room": "rm",
    "department": "dept",
}

ADDRESS_LABEL_TOKENS: tuple[str, ...] = ("property", "address")
ADDRESS_LABEL_TAIL = "address"
BRACKET_PATTERN = re.compile(r"\[[^\]]*\]")
UNIT_MARKER_PATTERN = re.compile(r"#\s*(?=\d)")


def require_non_empty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    characters: list[str] = []
    for character in without_marks.casefold():
        if unicodedata.category(character) == "Pd":
            characters.append(" ")
        elif character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def _collapse_adjacent_duplicates(tokens: Sequence[str]) -> list[str]:
    collapsed: list[str] = []
    for token in tokens:
        if collapsed and collapsed[-1] == token:
            continue
        collapsed.append(token)
    return collapsed


def _strip_label_tokens(tokens: list[str]) -> list[str]:
    while len(tokens) >= len(ADDRESS_LABEL_TOKENS) and (
        tuple(tokens[: len(ADDRESS_LABEL_TOKENS)]) == ADDRESS_LABEL_TOKENS
    ):
        del tokens[: len(ADDRESS_LABEL_TOKENS)]
    while len(tokens) >= len(ADDRESS_LABEL_TOKENS) and (
        tuple(tokens[-len(ADDRESS_LABEL_TOKENS) :]) == ADDRESS_LABEL_TOKENS
    ):
        del tokens[-len(ADDRESS_LABEL_TOKENS) :]
    while tokens and tokens[0] == ADDRESS_LABEL_TAIL:
        del tokens[0]
    while tokens and tokens[-1] == ADDRESS_LABEL_TAIL:
        del tokens[-1]
    return tokens


def canonicalize_address(value: str) -> str:
    require_non_empty(value, "Address")
    without_brackets = BRACKET_PATTERN.sub(" ", value)
    with_unit_markers = UNIT_MARKER_PATTERN.sub(" unit ", without_brackets)
    tokens = [
        USPS_DESIGNATORS.get(token, token)
        for token in normalize_text(with_unit_markers).split()
    ]
    tokens = _strip_label_tokens(tokens)
    return " ".join(_collapse_adjacent_duplicates(tokens)).strip()


def canonicalize_person_name(value: str) -> str:
    require_non_empty(value, "Person name")
    return " ".join(sorted(normalize_text(value).split()))


def canonicalize_loan_number(value: str) -> str:
    require_non_empty(value, "Loan number")
    return "".join(
        character for character in value.casefold() if character.isalnum()
    )


def canonicalize_page_number(value: str) -> str:
    require_non_empty(value, "Page number")
    return normalize_text(value)


_CANONICALIZERS: Mapping[str, Callable[[str], str]] = {
    FIELD_PROPERTY_ADDRESS: canonicalize_address,
    FIELD_BORROWER_NAME: canonicalize_person_name,
    FIELD_LOAN_NUMBER: canonicalize_loan_number,
    FIELD_PAGE_NUMBER: canonicalize_page_number,
}


def _canonicalizer(field: str) -> Callable[[str], str]:
    canonicalizer = _CANONICALIZERS.get(field)
    if canonicalizer is None:
        raise ValueError(
            f"Unsupported field {field!r}; expected one of {sorted(_CANONICALIZERS)}"
        )
    return canonicalizer


def canonical_key(field: str, value: str) -> str:
    return _canonicalizer(field)(value)


def cluster_keys(field: str, value: str) -> tuple[str, ...]:
    primary = canonical_key(field, value)
    if field != FIELD_PROPERTY_ADDRESS:
        return (primary,)
    compact = primary.replace(" ", "")
    if not compact or compact == primary:
        return (primary,)
    return (primary, compact)


def cluster_similarity_threshold(field: str) -> float:
    _canonicalizer(field)
    return CLUSTER_SIMILARITY_THRESHOLDS[field]


def similarity(field: str, left: str, right: str) -> float:
    left_key = canonical_key(field, left)
    right_key = canonical_key(field, right)
    if not left_key or not right_key:
        return 0.0
    return difflib.SequenceMatcher(None, left_key, right_key).ratio()


def values_match(field: str, left: str, right: str) -> bool:
    left_key = canonical_key(field, left)
    right_key = canonical_key(field, right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if set(cluster_keys(field, left)) & set(cluster_keys(field, right)):
        return True
    return similarity(field, left, right) >= cluster_similarity_threshold(field)


def _page_tuple(pages: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for page in pages:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"Variant pages must be positive integers, got: {page!r}")
        result.append(page)
    if not result:
        raise ValueError("Each variant must reference at least one page")
    return tuple(result)


def _non_ascii_count(value: str) -> int:
    return sum(1 for character in value if ord(character) > 127)


def _annotation_count(value: str) -> int:
    return value.count("[") + value.count("]")


def representative_value(
    field: str, variants: Sequence[tuple[str, Sequence[int]]]
) -> str:
    if not variants:
        raise ValueError("variants must not be empty")
    parsed: list[tuple[str, tuple[int, ...]]] = []
    support: Counter[str] = Counter()
    for surface, pages in variants:
        require_non_empty(surface, "Variant value")
        page_tuple = _page_tuple(pages)
        parsed.append((surface, page_tuple))
        support[canonical_key(field, surface)] += len(page_tuple)

    def rank(item: tuple[str, tuple[int, ...]]) -> tuple[int, int, int, int, int, str]:
        surface, page_tuple = item
        return (
            -support[canonical_key(field, surface)],
            _non_ascii_count(surface),
            _annotation_count(surface),
            -len(page_tuple),
            min(page_tuple),
            surface,
        )

    return min(parsed, key=rank)[0]
