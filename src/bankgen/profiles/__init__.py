"""Bank profiles. `get_profile(key)` is the only entry point the generator uses."""

from __future__ import annotations

from bankgen.profiles.base import Profile
from bankgen.profiles.cauchy import CAUCHY

PROFILES: dict[str, Profile] = {CAUCHY.key: CAUCHY}


def get_profile(key: str) -> Profile:
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise KeyError(f"unknown profile {key!r}; known: {sorted(PROFILES)}") from exc
