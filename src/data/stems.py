"""Stem layouts (MUSDB 4-stem and MoisesDB 4/5/6-stem groupings)."""

from __future__ import annotations

# MUSDB18 / default inference
STEM_NAMES_4 = ("vocals", "drums", "bass", "other")

# MoisesDB paper groupings (see moises-ai/moises-db moisesdb/defaults.py)
STEM_NAMES_5 = ("vocals", "bass", "guitar", "drums", "other")
STEM_NAMES_6 = ("vocals", "bass", "guitar", "piano", "drums", "other")

STEM_NAMES = STEM_NAMES_4  # backward compatible alias

_MOISES_ALL = [
    "vocals",
    "bass",
    "drums",
    "other",
    "guitar",
    "other_plucked",
    "percussion",
    "piano",
    "other_keys",
    "bowed_strings",
    "wind",
]

MOISES_MIX_4 = {
    "vocals": ["vocals"],
    "bass": ["bass"],
    "drums": ["drums"],
    "other": [s for s in _MOISES_ALL if s not in ("vocals", "bass", "drums")],
}

MOISES_MIX_5 = {
    "vocals": ["vocals"],
    "bass": ["bass"],
    "guitar": ["guitar"],
    "drums": ["drums"],
    "other": [s for s in _MOISES_ALL if s not in ("vocals", "bass", "drums", "guitar")],
}

MOISES_MIX_6 = {
    "vocals": ["vocals"],
    "bass": ["bass"],
    "guitar": ["guitar"],
    "piano": ["piano"],
    "drums": ["drums"],
    "other": [
        s
        for s in _MOISES_ALL
        if s not in ("vocals", "bass", "drums", "guitar", "piano")
    ],
}


def stem_names_for(num_stems: int) -> tuple[str, ...]:
    if num_stems == 1:
        return ("vocals",)
    if num_stems == 4:
        return STEM_NAMES_4
    if num_stems == 5:
        return STEM_NAMES_5
    if num_stems == 6:
        return STEM_NAMES_6
    raise ValueError(f"Unsupported num_stems={num_stems} (use 1, 4, 5, or 6)")


def moises_mix_map(num_stems: int) -> dict[str, list[str]]:
    if num_stems == 4:
        return MOISES_MIX_4
    if num_stems == 5:
        return MOISES_MIX_5
    if num_stems == 6:
        return MOISES_MIX_6
    raise ValueError("Moises mix maps are defined for 4, 5, or 6 stems")
