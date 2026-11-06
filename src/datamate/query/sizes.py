"""Size labels a shop writes in its product names.

Two ways are seen: a suffix "(M)" ("Trà vải (L)") and a word "size M" ("Tã dán ... size
XL 40 miếng"). Labels follow the clothing size notation (S, M, L, XL, XXL, XS, NB), so
other bracketed codes such as "(COD)" are not sizes.
"""

from __future__ import annotations

import re

_LABEL = r"(?:X{0,3}S|M|X{0,3}L|NB)"
_SUFFIX = re.compile(rf"\(\s*({_LABEL})\s*\)\s*$", re.IGNORECASE)
_WORD = re.compile(rf"(?<![^\W\d_])size\s+({_LABEL})(?![^\W_])", re.IGNORECASE)
_LABEL_ONLY = re.compile(rf"^{_LABEL}$")
_ORDER = ("NB", "XS", "S", "M", "L", "XL", "XXL", "XXXL")


def size_label(value: object) -> str | None:
    """`value` upper-cased when it is a size label, else None."""
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text if _LABEL_ONLY.match(text) else None


def shop_sizes(product_names: list[str]) -> dict[str, str]:
    """Size label -> "suffix" or "word", for labels found in the shop's product names.

    A label written both ways counts as "suffix" (the older cafe convention).
    """
    found: dict[str, str] = {}
    for name in product_names:
        match = _SUFFIX.search(name)
        if match:
            found[match[1].upper()] = "suffix"
        for word in _WORD.finditer(name):
            found.setdefault(word[1].upper(), "word")
    return {label: found[label] for label in sorted(found, key=_rank)}


def name_has_size(name: str, label: str) -> bool:
    """True when the product name carries the size label in either written form."""
    wanted = label.upper()
    match = _SUFFIX.search(name)
    if match and match[1].upper() == wanted:
        return True
    return any(word[1].upper() == wanted for word in _WORD.finditer(name))


def _rank(label: str) -> int:
    return _ORDER.index(label) if label in _ORDER else len(_ORDER)
