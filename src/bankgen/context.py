"""The generation context: profile, seed, scale and the tables built so far."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
from faker import Faker

from bankgen.profiles.base import Profile
from bankgen.rng import make_rng


@dataclass
class Context:
    profile: Profile
    seed: int
    n_customers: int
    out_dir: Path
    tables: dict[str, pa.Table] = field(default_factory=dict)

    def rng(self, *stream: str | int):  # type: ignore[no-untyped-def]
        return make_rng(self.seed, self.profile.rng_key, *stream)

    def faker(self, *stream: str) -> Faker:
        fake = Faker(self.profile.faker_locale)
        fake.seed_instance(int(self.rng(*stream).integers(0, 2**31 - 1)))
        return fake

    def table(self, name: str) -> pa.Table:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise KeyError(f"table {name!r} has not been built yet") from exc
