"""
The data dictionary: every table, view and column the generator writes is described, and nothing
is described that the generator does not write; the descriptions name no bank (its words come from the profile);
the code columns hold only the values the dictionary lists; the keys it declares hold in the data; and `load`
writes every description into the DuckDB file.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pytest

from bankgen.context import Context
from bankgen.dictionary import (
    CATALOG,
    RELATIONS,
    SCHEMA,
    Described,
    comment_statements,
    describe,
    problems,
    sql_string,
    values_of,
    vocabulary,
)
from bankgen.profiles import PROFILES

pytestmark = pytest.mark.unit

OUT = Path(__file__).resolve().parents[2] / "data" / "out"
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def schema_of(db: Path) -> dict[str, list[str]]:
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = current_schema() "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
    out: dict[str, list[str]] = {}
    for table, column in rows:
        out.setdefault(str(table), []).append(str(column))
    return out


def raw_texts() -> list[str]:
    """Every text the dictionary holds before rendering, column keys and fixed values included."""
    texts = [CATALOG, SCHEMA]
    for relation in RELATIONS.values():
        texts.append(relation.text)
        for key, column in relation.columns.items():
            texts += [key, column.text]
            if isinstance(column.values, tuple):
                texts += list(column.values)
    return texts


def test_every_table_view_and_column_is_described(generated: tuple[str, Context, Path]) -> None:
    key, _, db = generated
    actual = schema_of(db)
    assert len(actual) == 16 and sum(len(c) for c in actual.values()) == 188
    assert problems(describe(key), actual) == []


def test_a_column_without_a_description_and_a_description_without_a_column_are_reported() -> None:
    """The failing fixture of AC1: an undescribed column added to a table, and a described column dropped."""
    described = describe("cauchy")
    actual = {t: list(cols) for t, cols in described.columns.items()}
    actual["loans"].append("added_without_description")
    actual["regions"].remove("subdivision")
    actual["new_table"] = ["id"]
    assert sorted(problems(described, actual)) == [
        "loans.added_without_description: no description",
        "new_table: no description",
        "regions.subdivision: described but not in the table",
    ]
    empty = Described(described.catalog, described.schema, described.tables, {**described.columns, "regions": {}})
    assert "regions.region_id: no description" in problems(empty, {"regions": ["region_id"]})


def test_the_text_names_no_bank_and_every_placeholder_comes_from_the_profile() -> None:
    texts = raw_texts()
    for key, p in PROFILES.items():
        words = vocabulary(p)
        named = {
            p.bank_name,
            p.currency,
            p.national_id_label,
            *(a.bank for a in p.acquisitions),
            *(r.name for r in p.regions),
        }
        for text in texts:
            assert not any(word in text for word in named), f"{text[:60]!r} names {key}'s words"
            assert not re.search(rf"\b{key}\b", text), f"{text[:60]!r} names the bank key {key}"
            assert set(PLACEHOLDER.findall(text)) <= set(words), f"{text[:60]!r} has an unknown placeholder"
            assert "\\" not in text  # a backslash would read differently in a Databricks SQL literal
        rendered = describe(key)
        everything = [rendered.catalog, rendered.schema, *rendered.tables.values()]
        everything += [c for cols in rendered.columns.values() for c in [*cols, *cols.values()]]
        assert not any("{" in t or "}" in t for t in everything)
        assert p.bank_name in rendered.catalog and p.currency in rendered.schema
        assert p.national_id_label in rendered.columns["customers"]


def test_the_code_columns_hold_only_the_listed_values(generated: tuple[str, Context, Path]) -> None:
    key, _, db = generated
    p = PROFILES[key]
    with duckdb.connect(str(db), read_only=True) as con:
        for table, relation in RELATIONS.items():
            for column, spec in relation.columns.items():
                listed = values_of(spec, p)
                if listed is None:
                    continue
                found = {str(r[0]) for r in con.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()}
                assert found <= set(listed), f"{key} {table}.{column}: {sorted(found - set(listed))} not listed"


@pytest.mark.skipif(not (OUT / "cauchy.duckdb").is_file(), reason="no generated data")
@pytest.mark.parametrize("key", sorted(PROFILES))
def test_the_full_books_match_the_dictionary(key: str) -> None:
    """The same checks on the DuckDB files `bankgen all` wrote, when they are there (not on a fresh clone)."""
    db = OUT / f"{key}.duckdb"
    assert problems(describe(key), schema_of(db)) == []
    p = PROFILES[key]
    with duckdb.connect(str(db), read_only=True) as con:
        for table, relation in RELATIONS.items():
            for column, spec in relation.columns.items():
                listed = values_of(spec, p)
                if listed is not None:
                    found = {str(r[0]) for r in con.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()}
                    assert found <= set(listed), f"{key} {table}.{column}: {sorted(found - set(listed))}"


def test_the_keys_hold_in_the_data(generated: tuple[str, Context, Path]) -> None:
    """Every primary key is unique and never null; every foreign key value is a parent's key, except the empty
    string that marks an absent id (Unity Catalog's informational keys are not enforced, so the data must hold)."""
    key, _, db = generated
    with duckdb.connect(str(db), read_only=True) as con:
        for table, relation in RELATIONS.items():
            if relation.primary_key:
                cols = ", ".join(relation.primary_key)
                nulls = " OR ".join(f"{c} IS NULL" for c in relation.primary_key)
                total, distinct, null = con.execute(
                    f"SELECT COUNT(*), COUNT(DISTINCT ({cols})), COUNT(*) FILTER (WHERE {nulls}) FROM {table}"
                ).fetchone() or (0, 0, 0)
                assert total == distinct and null == 0, f"{key} {table} ({cols})"
            for fk in relation.foreign_keys:
                parent = RELATIONS[fk.parent]
                assert parent.primary_key == (fk.parent_column,), f"{table}.{fk.column} refers to a non-key"
                (orphans,) = con.execute(
                    f"SELECT COUNT(*) FROM {table} c WHERE CAST(c.{fk.column} AS VARCHAR) <> '' AND NOT EXISTS "
                    f"(SELECT 1 FROM {fk.parent} p WHERE p.{fk.parent_column} = c.{fk.column})"
                ).fetchone() or (1,)
                assert orphans == 0, f"{key} {table}.{fk.column} -> {fk.parent}: {orphans} orphans"


def test_load_writes_every_description_into_the_duckdb_file(generated: tuple[str, Context, Path]) -> None:
    key, _, db = generated
    described = describe(key)
    with duckdb.connect(str(db), read_only=True) as con:
        tables = dict(
            con.execute(
                "SELECT table_name, table_comment FROM information_schema.tables WHERE table_schema = current_schema()"
            ).fetchall()
        )
        columns = {
            (t, c): text
            for t, c, text in con.execute(
                "SELECT table_name, column_name, column_comment FROM information_schema.columns "
                "WHERE table_schema = current_schema()"
            ).fetchall()
        }
    assert tables == described.tables
    assert columns == {(t, c): text for t, cols in described.columns.items() for c, text in cols.items()}


def test_comment_statements_quote_and_qualify() -> None:
    described = describe("cauchy")
    statements = comment_statements(described, "regions", lambda t: f"demo.data.{t}")
    assert statements[0].startswith("COMMENT ON TABLE demo.data.regions IS 'One row per region")
    assert len(statements) == 1 + len(described.columns["regions"])
    assert statements[1].startswith("COMMENT ON COLUMN demo.data.regions.region_id IS ")
    assert sql_string("the bank's own") == "'the bank''s own'"
    with duckdb.connect() as con:
        con.execute("CREATE TABLE t (a INTEGER)")
        con.execute(f"COMMENT ON COLUMN t.a IS {sql_string(described.columns['loans']['underwritten_by'])}")
        (text,) = con.execute("SELECT column_comment FROM information_schema.columns").fetchone() or ("",)
        assert text == described.columns["loans"]["underwritten_by"]
