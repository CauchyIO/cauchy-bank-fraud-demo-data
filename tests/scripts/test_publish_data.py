"""Unit tests of ``scripts/publish_data.py``.

The client is a ``MagicMock(spec=WorkspaceClient)`` whose services are autospecced from the SDK
classes and read and write one in-memory ``World``: the volume's files, the tables with their row
counts, the views with their bodies, the comments and keys, and every statement run, in order. The local side is
a real
``bankgen`` cut of 60 customers generated once into ``tmp_path`` (deterministic: seed 3), so the
counts the script reads from DuckDB are the ones the generator wrote. No network, no clock.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, create_autospec

import duckdb
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service import catalog, files, iam, sql

import publish_data as publish
from bankgen.dictionary import RELATIONS, describe
from bankgen.generate import TABLE_ORDER, generate
from bankgen.load import VIEWS as BANKGEN_VIEWS
from bankgen.load import load

pytestmark = pytest.mark.unit

CALLER = "analyst@example.com"
BANK = "cauchy"
CATALOG, SCHEMA = "demo", "fraud"
WAREHOUSE = ("13ee27ac339ceda7", "Shared Serverless")
ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
TABLES = [*TABLE_ORDER, "transactions"]
VIEWS = list(BANKGEN_VIEWS)


@pytest.fixture(scope="module")
def out_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One small generated dataset for the module: Parquet, the manifest and the DuckDB file."""
    root = tmp_path_factory.mktemp("out")
    generate(BANK, root, seed=3, n_customers=60, log=False)
    load(BANK, root)
    return root


@dataclass
class World:
    """What the workspace holds, and every write made to it, in order."""

    files: dict[str, bytes] = field(default_factory=dict)
    tables: dict[str, int] = field(default_factory=dict)  # full name -> row count
    views: dict[str, str] = field(default_factory=dict)  # full name -> body
    statements: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    local_db: Path | None = None  # where a CTAS reads its row count from, and every table its columns
    count_override: dict[str, int] = field(default_factory=dict)
    catalog_comment: str | None = "Someone else's catalog"
    schema_comment: str | None = None
    catalog_exists: bool = True
    schema_exists: bool = True
    volume_exists: bool = True
    comments: dict[str, str] = field(default_factory=dict)  # full name, or full name.column -> comment
    not_null: set[str] = field(default_factory=set)  # full name.column
    constraints: dict[str, dict[str, catalog.TableConstraint]] = field(default_factory=dict)  # full name -> by name


LITERAL = r"'((?:[^']|'')*)'"
CTAS = re.compile(rf"CREATE OR REPLACE TABLE (\S+) COMMENT {LITERAL} AS\s+SELECT \* FROM read_files\('([^']+)'", re.S)
COMMENT = re.compile(rf"COMMENT ON (SCHEMA|TABLE|COLUMN) (\S+) IS {LITERAL}$", re.S)
NOT_NULL = re.compile(r"ALTER TABLE (\S+) ALTER COLUMN (\w+) SET NOT NULL$")
PRIMARY = re.compile(r"ALTER TABLE (\S+) ADD CONSTRAINT (\w+) PRIMARY KEY \(([\w, ]+)\)$")
FOREIGN = re.compile(r"ALTER TABLE (\S+) ADD CONSTRAINT (\w+) FOREIGN KEY \((\w+)\) REFERENCES (\S+) \((\w+)\)$")
QUALIFIED = f"{CATALOG}.{SCHEMA}."
VIEW = re.compile(r"CREATE OR REPLACE VIEW (\S+) AS (.+)", re.S)
COUNT = re.compile(r"SELECT COUNT\(\*\) FROM (\S+)")
COUNT_COLUMN = sql.ColumnInfo(name="n", type_text="BIGINT")


def local_count(db: Path | None, name: str) -> int:
    if db is None:
        return 0
    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def local_columns(db: Path | None, name: str) -> list[str]:
    if db is None:
        return []
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows = con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [name]
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


def unquote(text: str) -> str:
    return text.replace("''", "'")


def replace_table(world: World, name: str) -> None:
    """What a CREATE OR REPLACE TABLE ... AS SELECT does on Databricks: the column comments stay, every
    column turns nullable, and the table's constraints go, with every foreign key that refers to it."""
    world.not_null = {c for c in world.not_null if not c.startswith(name + ".")}
    world.constraints.pop(name, None)
    for held in world.constraints.values():
        for key, c in list(held.items()):
            if c.foreign_key_constraint and c.foreign_key_constraint.parent_table == name:
                del held[key]


def client(world: World) -> MagicMock:
    w = MagicMock(spec=WorkspaceClient)
    w.config.host = "https://adb-1.azuredatabricks.net"
    w.current_user = create_autospec(iam.CurrentUserAPI, instance=True)
    w.current_user.me.return_value = iam.User(user_name=CALLER)

    w.warehouses = create_autospec(sql.WarehousesAPI, instance=True)
    w.warehouses.list.return_value = [
        sql.EndpointInfo(id=WAREHOUSE[0], name=WAREHOUSE[1], enable_serverless_compute=True),
        sql.EndpointInfo(id="other1", name="Serverless Starter Warehouse", enable_serverless_compute=True),
        sql.EndpointInfo(id="classic1", name="classic", enable_serverless_compute=False),
    ]

    w.files = create_autospec(files.FilesAPI, instance=True)

    def list_directory(directory_path: str, **_: Any) -> Iterator[files.DirectoryEntry]:
        prefix = directory_path.rstrip("/") + "/"
        names = {p[len(prefix) :].split("/", 1)[0] for p in world.files if p.startswith(prefix)}
        if not names:
            raise NotFound(f"No such directory: {directory_path}")
        for name in sorted(names):
            path = prefix + name
            is_dir = path not in world.files
            yield files.DirectoryEntry(
                name=name, path=path, is_directory=is_dir, file_size=None if is_dir else len(world.files[path])
            )

    def upload(file_path: str, contents: io.BufferedReader, overwrite: bool | None = None, **_: Any) -> None:
        assert overwrite is True
        world.files[file_path] = contents.read()
        world.log.append(f"upload {file_path}")

    def delete(file_path: str) -> None:
        del world.files[file_path]
        world.log.append(f"delete {file_path}")

    w.files.list_directory_contents.side_effect = list_directory
    w.files.upload.side_effect = upload
    w.files.delete.side_effect = delete

    w.tables = create_autospec(catalog.TablesAPI, instance=True)

    def exists(full_name: str) -> catalog.TableExistsResponse:
        return catalog.TableExistsResponse(table_exists=full_name in world.tables or full_name in world.views)

    def get(full_name: str, **_: Any) -> catalog.TableInfo:
        if full_name not in world.views and full_name not in world.tables:
            raise NotFound(f"Table '{full_name}' does not exist.")
        columns = [
            catalog.ColumnInfo(
                name=c,
                comment=world.comments.get(f"{full_name}.{c}"),
                nullable=f"{full_name}.{c}" not in world.not_null,
            )
            for c in local_columns(world.local_db, full_name.rsplit(".", 1)[1])
        ]
        return catalog.TableInfo(
            full_name=full_name,
            view_definition=world.views.get(full_name),
            comment=world.comments.get(full_name),
            columns=columns,
            table_constraints=list(world.constraints.get(full_name, {}).values()),
        )

    w.tables.exists.side_effect = exists
    w.tables.get.side_effect = get
    w.catalogs = create_autospec(catalog.CatalogsAPI, instance=True)

    def get_catalog(name: str, **_: Any) -> catalog.CatalogInfo:
        if not world.catalog_exists:
            raise NotFound(f"Catalog '{name}' does not exist.")
        return catalog.CatalogInfo(name=name, comment=world.catalog_comment)

    w.catalogs.get.side_effect = get_catalog
    w.schemas = create_autospec(catalog.SchemasAPI, instance=True)

    def get_schema(full_name: str, **_: Any) -> catalog.SchemaInfo:
        if not world.schema_exists:
            raise NotFound(f"Schema '{full_name}' does not exist.")
        return catalog.SchemaInfo(full_name=full_name, comment=world.schema_comment)

    def create_schema(name: str, catalog_name: str, **_: Any) -> catalog.SchemaInfo:
        world.schema_exists = True
        world.log.append(f"create schema {catalog_name}.{name}")
        return catalog.SchemaInfo(full_name=f"{catalog_name}.{name}")

    w.schemas.get.side_effect = get_schema
    w.schemas.create.side_effect = create_schema
    w.volumes = create_autospec(catalog.VolumesAPI, instance=True)

    def read_volume(name: str, **_: Any) -> catalog.VolumeInfo:
        if not world.volume_exists:
            raise NotFound(f"Volume '{name}' does not exist.")
        return catalog.VolumeInfo(full_name=name)

    def create_volume(
        catalog_name: str, schema_name: str, name: str, volume_type: catalog.VolumeType, **_: Any
    ) -> catalog.VolumeInfo:
        assert world.schema_exists and volume_type == catalog.VolumeType.MANAGED
        world.volume_exists = True
        world.log.append(f"create volume {catalog_name}.{schema_name}.{name}")
        return catalog.VolumeInfo(full_name=f"{catalog_name}.{schema_name}.{name}")

    w.volumes.read.side_effect = read_volume
    w.volumes.create.side_effect = create_volume

    w.statement_execution = create_autospec(sql.StatementExecutionAPI, instance=True)

    def execute(statement: str, warehouse_id: str, **kwargs: Any) -> sql.StatementResponse:
        assert warehouse_id == WAREHOUSE[0]
        assert kwargs.get("catalog") == CATALOG and kwargs.get("schema") == SCHEMA
        assert world.schema_exists, "a statement ran before the schema existed"
        world.statements.append(statement)
        rows: list[list[str | None]] = []
        if m := CTAS.search(statement):
            name, comment, directory = m.group(1), unquote(m.group(2)), m.group(3)
            table = name.rsplit(".", 1)[1]
            assert directory == f"{ROOT}/{table}/" and "schemaEvolutionMode => 'none'" in statement
            assert any(p.startswith(directory) for p in world.files), f"no files under {directory}"
            world.tables[name] = world.count_override.get(table, local_count(world.local_db, table))
            world.comments[name] = comment
            replace_table(world, name)
            world.log.append(f"ctas {name}")
        elif m := VIEW.search(statement):
            world.views[m.group(1)] = m.group(2)
            world.comments = {k: v for k, v in world.comments.items() if k.split(".")[:3] != m.group(1).split(".")}
            world.log.append(f"view {m.group(1)}")
        elif m := COMMENT.match(statement):
            kind, target, text = m.group(1), m.group(2), unquote(m.group(3))
            if kind == "SCHEMA":
                world.schema_comment = text
            else:
                relation = target if kind == "TABLE" else target.rsplit(".", 1)[0]
                assert relation in world.tables or relation in world.views, f"no relation {relation}"
                world.comments[target] = text
            world.log.append(f"comment {kind.lower()} {target}")
        elif m := NOT_NULL.match(statement):
            world.not_null.add(f"{m.group(1)}.{m.group(2)}")
            world.log.append(f"not null {m.group(1)}.{m.group(2)}")
        elif m := PRIMARY.match(statement):
            name, constraint, cols = m.group(1), m.group(2), [c.strip() for c in m.group(3).split(",")]
            if any(f"{name}.{c}" not in world.not_null for c in cols):
                return failed("a primary key column must be NOT NULL")
            world.constraints.setdefault(name, {})[constraint] = catalog.TableConstraint(
                primary_key_constraint=catalog.PrimaryKeyConstraint(name=constraint, child_columns=cols)
            )
            world.log.append(f"pk {constraint}")
        elif m := FOREIGN.match(statement):
            name, constraint, column, parent, parent_column = m.groups()
            keys = [c.primary_key_constraint for c in world.constraints.get(parent, {}).values()]
            if not any(k and k.child_columns == [parent_column] for k in keys):
                return failed(f"{parent} has no primary key ({parent_column})")
            world.constraints.setdefault(name, {})[constraint] = catalog.TableConstraint(
                foreign_key_constraint=catalog.ForeignKeyConstraint(
                    name=constraint, child_columns=[column], parent_table=parent, parent_columns=[parent_column]
                )
            )
            world.log.append(f"fk {constraint}")
        elif m := COUNT.search(statement):
            name = m.group(1)
            short = name.rsplit(".", 1)[1]
            if name in world.tables:
                rows = [[str(world.tables[name])]]
            elif name in world.views:
                rows = [[str(world.count_override.get(short, local_count(world.local_db, short)))]]
            else:
                return failed(f"TABLE_OR_VIEW_NOT_FOUND: {name}")
        else:
            raise AssertionError(f"unexpected statement: {statement}")
        return sql.StatementResponse(
            statement_id="s1",
            status=sql.StatementStatus(state=sql.StatementState.SUCCEEDED),
            manifest=sql.ResultManifest(
                schema=sql.ResultSchema(columns=[sql.ColumnInfo(name="n", type_text="BIGINT")])
            ),
            result=sql.ResultData(
                data_array=cast("list[list[str]]", rows)
            ),  # the API returns nulls; the SDK's annotation does not
        )

    def failed(message: str) -> sql.StatementResponse:
        return sql.StatementResponse(
            statement_id="s1",
            status=sql.StatementStatus(state=sql.StatementState.FAILED, error=sql.ServiceError(message=message)),
        )

    w.statement_execution.execute_statement.side_effect = execute
    return w


def publisher(
    world: World, out_root: Path, *, apply: bool, force: bool = False, show_sql: bool = False
) -> tuple[publish.Publish, list[str]]:
    printed: list[str] = []
    p = publish.Publish(
        client(world),
        catalog=CATALOG,
        schema=SCHEMA,
        warehouse_id=WAREHOUSE[0],
        apply=apply,
        out_root=out_root,
        force=force,
        sleep=lambda _: None,
        out=printed.append,
        show_sql=show_sql,
    )
    return p, printed


def states(p: publish.Publish, part: str) -> list[str]:
    return [line.state for line in p.lines if line.part == part]


DESCRIBED = describe(BANK)
KEYED = [t for t, r in RELATIONS.items() if r.primary_key]
FOREIGN_KEYS = [(t, k) for t, r in RELATIONS.items() if not r.view for k in r.foreign_keys]
KEY_COLUMNS = sum(len(r.primary_key) for r in RELATIONS.values())
COMMENT_STATEMENTS = 1 + len(DESCRIBED.tables) + sum(len(c) for c in DESCRIBED.columns.values())


def assert_described_and_keyed(world: World) -> None:
    """Everything the dictionary says is in the workspace: every comment, every key with its NOT NULL columns; the
    catalog's own comment is left as it was."""
    assert world.catalog_comment == "Someone else's catalog" and world.schema_comment == DESCRIBED.schema
    for name, text in DESCRIBED.tables.items():
        assert world.comments[QUALIFIED + name] == text
        for column, ctext in DESCRIBED.columns[name].items():
            assert world.comments[f"{QUALIFIED}{name}.{column}"] == ctext
    held = {n: c for cs in world.constraints.values() for n, c in cs.items()}
    assert sorted(held) == sorted([f"{t}_pk" for t in KEYED] + [f"{t}_{k.column}_fk" for t, k in FOREIGN_KEYS])
    assert len(world.not_null) == KEY_COLUMNS


# --- the parts ------------------------------------------------------------------------------------


def test_a_dry_run_lists_everything_and_writes_nothing(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    p, printed = publisher(world, out_root, apply=False)
    assert publish.run(p) is True
    expected_files = len(publish.local_files(p))
    assert states(p, "local") == ["exists"]
    assert states(p, "schema") == ["exists", "exists"]
    assert states(p, "files") == ["would"] * expected_files
    assert states(p, "tables") == ["would"] * len(TABLES)
    assert states(p, "views") == ["would"] * len(VIEWS)
    assert states(p, "comments") == ["would"] * COMMENT_STATEMENTS
    assert states(p, "keys") == ["would"] * (KEY_COLUMNS + len(KEYED) + len(FOREIGN_KEYS))
    assert world.files == {} and world.tables == {} and world.statements == [] and world.log == []
    assert any("after apply" in line and "transactions" in line for line in printed)


def test_an_apply_uploads_creates_and_counts(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    p, printed = publisher(world, out_root, apply=True)
    assert publish.run(p) is True
    local = publish.local_files(p)
    assert sorted(world.files) == sorted(f.remote for f in local)
    assert all(len(world.files[f.remote]) == f.size for f in local)
    assert [s for s in world.log if s.startswith("ctas")] == [f"ctas {QUALIFIED}{t}" for t in TABLES]
    assert [s for s in world.log if s.startswith("view")] == [f"view {QUALIFIED}{v}" for v in VIEWS]
    uploads = [s for s in world.log if s.startswith("upload")]
    assert world.log.index(uploads[-1]) < world.log.index(f"ctas {QUALIFIED}regions")  # files before tables
    assert world.views[f"{QUALIFIED}v_monthly_loans"] == publish.view_sql(BANKGEN_VIEWS["v_monthly_loans"])
    assert states(p, "mismatch") == [] and p.count("mismatch") == 0
    assert any(line.state == "info" and "counts match" in line.what for line in p.lines)
    assert sum(1 for s in world.statements if s.startswith("SELECT COUNT(*)")) == len(TABLES) + len(VIEWS)
    assert any(re.search(r"transactions\s+[\d,]+\s+[\d,]+\s+ok", line) for line in printed)
    assert_described_and_keyed(world)
    # the CTAS carries the table's comment, so only the schema, the views and the columns are commented
    assert sum(1 for s in world.log if s.startswith("comment")) == COMMENT_STATEMENTS - len(TABLES)
    last_view = world.log.index(f"view {QUALIFIED}{VIEWS[-1]}")
    first_comment = next(i for i, s in enumerate(world.log) if s.startswith("comment"))
    first_fk = next(i for i, s in enumerate(world.log) if s.startswith("fk"))
    last_pk = max(i for i, s in enumerate(world.log) if s.startswith("pk"))
    assert last_view < first_comment < last_pk < first_fk  # tables and views, then comments, then keys


def test_a_second_apply_finds_the_files_and_tables_in_place(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    first, _ = publisher(world, out_root, apply=True)
    publish.run(first)
    world.log.clear()
    world.statements.clear()
    second, printed = publisher(world, out_root, apply=True)
    assert publish.run(second) is True
    assert set(states(second, "files")) == {"exists"}
    assert set(states(second, "tables")) == {"exists"}
    assert set(states(second, "views")) == {"exists"}
    assert world.log == []
    assert all(s.startswith("SELECT COUNT(*)") for s in world.statements)
    assert printed.count("  nothing to do") == 7  # local, schema, files, tables, views, comments, keys
    assert set(states(second, "comments")) == {"exists"} and set(states(second, "keys")) == {"exists"}


def test_force_replaces_everything(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    first, _ = publisher(world, out_root, apply=True)
    publish.run(first)
    world.log.clear()
    forced, _ = publisher(world, out_root, apply=True, force=True)
    assert publish.run(forced) is True
    assert sum(1 for s in world.log if s.startswith("upload")) == len(publish.local_files(forced))
    assert sum(1 for s in world.log if s.startswith("ctas")) == len(TABLES)
    assert sum(1 for s in world.log if s.startswith("view")) == len(VIEWS)
    # the replaced tables keep their column comments and lose their keys; the replaced views lose their comments
    assert sum(1 for s in world.log if s.startswith("pk")) == len(KEYED)
    assert sum(1 for s in world.log if s.startswith("fk")) == len(FOREIGN_KEYS)
    view_columns = sum(len(DESCRIBED.columns[v]) for v in VIEWS)
    assert sum(1 for s in world.log if s.startswith("comment")) == len(VIEWS) + view_columns
    assert_described_and_keyed(world)


def test_a_changed_file_replaces_only_its_table_and_a_stale_file_is_deleted(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    first, _ = publisher(world, out_root, apply=True)
    publish.run(first)
    world.files[f"{ROOT}/regions/regions.parquet"] = b"old"  # a different size: re-uploaded
    world.files[f"{ROOT}/transactions/part-2019-01.parquet"] = b"stale"  # not the generator's: deleted
    world.files[f"{ROOT}/somebody-elses.txt"] = b"x"  # at the root: left alone
    world.log.clear()
    second, _ = publisher(world, out_root, apply=True)
    assert publish.run(second) is True
    assert world.log == [
        f"upload {ROOT}/regions/regions.parquet",
        f"delete {ROOT}/transactions/part-2019-01.parquet",
        f"ctas {QUALIFIED}regions",
        f"ctas {QUALIFIED}transactions",
        # a replaced table loses its key and the foreign keys that refer to it; the keys part puts them back
        f"not null {QUALIFIED}regions.region_id",
        "pk regions_pk",
        f"not null {QUALIFIED}transactions.txn_id",
        "pk transactions_pk",
        "fk partners_region_id_fk",
        "fk merchants_region_id_fk",
        "fk customers_region_id_fk",
        "fk fraud_cases_first_txn_id_fk",
        "fk transactions_account_id_fk",
        "fk transactions_customer_id_fk",
        "fk transactions_merchant_id_fk",
        "fk transactions_fraud_case_id_fk",
    ]
    assert_described_and_keyed(world)
    assert f"{ROOT}/somebody-elses.txt" in world.files
    assert any(line.state == "info" and "left alone" in line.what for line in second.lines)


def test_a_changed_view_definition_is_replaced(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    first, _ = publisher(world, out_root, apply=True)
    publish.run(first)
    world.views[f"{QUALIFIED}v_customer_products"] = "SELECT 1"
    world.log.clear()
    second, _ = publisher(world, out_root, apply=True)
    publish.run(second)
    assert world.log[0] == f"view {QUALIFIED}v_customer_products"
    assert world.log[1:] == [f"comment table {QUALIFIED}v_customer_products"] + [
        f"comment column {QUALIFIED}v_customer_products.{c}" for c in DESCRIBED.columns["v_customer_products"]
    ]


def test_a_changed_description_is_rewritten_and_nothing_else(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb")
    first, _ = publisher(world, out_root, apply=True)
    publish.run(first)
    world.comments[f"{QUALIFIED}loans.status_at_window_end"] = "an older text"
    world.schema_comment = None
    world.log.clear()
    second, _ = publisher(world, out_root, apply=True)
    assert publish.run(second) is True
    assert world.log == [
        f"comment schema {CATALOG}.{SCHEMA}",
        f"comment column {QUALIFIED}loans.status_at_window_end",
    ]
    assert_described_and_keyed(world)


def test_the_dry_run_prints_the_statements_and_duckdb_runs_them(out_root: Path, tmp_path: Path) -> None:
    """``--show-sql`` prints every write in full; the comment, NOT NULL and primary-key statements run unchanged on
    a copy of the generated DuckDB file once unqualified (DuckDB enforces the keys, so they hold in the data); the
    foreign keys, which DuckDB cannot add by ALTER TABLE, are the dictionary's."""
    world = World(local_db=out_root / f"{BANK}.duckdb")
    p, printed = publisher(world, out_root, apply=False, show_sql=True)
    publish.run(p, only=["comments", "keys"])
    statements = [line.strip() for line in printed if line.startswith("      ")]
    assert len(statements) == COMMENT_STATEMENTS + KEY_COLUMNS + len(KEYED) + len(FOREIGN_KEYS)
    copy = tmp_path / "copy.duckdb"
    copy.write_bytes((out_root / f"{BANK}.duckdb").read_bytes())
    con = duckdb.connect(str(copy))
    try:
        ran = 0
        for statement in statements:
            if statement.startswith("COMMENT ON SCHEMA") or "FOREIGN KEY" in statement:
                continue
            con.execute(statement.replace(QUALIFIED, ""))
            ran += 1
        assert ran == COMMENT_STATEMENTS - 1 + KEY_COLUMNS + len(KEYED)
        keys = con.execute(
            "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_type = 'PRIMARY KEY'"
        ).fetchone()
        assert keys == (len(KEYED),)
    finally:
        con.close()
    foreign = [s for s in statements if "FOREIGN KEY" in s]
    assert foreign == [
        f"ALTER TABLE {QUALIFIED}{t} ADD CONSTRAINT {t}_{k.column}_fk FOREIGN KEY ({k.column}) "
        f"REFERENCES {QUALIFIED}{k.parent} ({k.parent_column})"
        for t, k in FOREIGN_KEYS
    ]
    assert world.statements == [] and world.log == []


def test_a_count_mismatch_fails_the_run(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb", count_override={"loans": 1})
    p, printed = publisher(world, out_root, apply=True)
    assert publish.run(p) is False
    assert [line.what for line in p.lines if line.state == "mismatch"] == [
        f"{QUALIFIED}loans: 1 rows in the workspace, {local_count(world.local_db, 'loans'):,} in DuckDB"
    ]
    assert any("MISMATCH" in line and line.strip().startswith("loans") for line in printed)


def test_missing_local_data_is_generated_on_apply_and_only_announced_in_a_dry_run(tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_generator(bank: str, root: Path) -> None:
        calls.append((bank, root))
        # the minimum the script checks for: every Parquet, a ledger part, the manifest, the DuckDB file
        (root / bank / "transactions").mkdir(parents=True)
        for t in TABLE_ORDER:
            (root / bank / f"{t}.parquet").write_bytes(b"")
        (root / bank / "transactions" / "part-2024-10.parquet").write_bytes(b"")
        (root / bank / "manifest.json").write_text('{"seed": 1, "n_customers": 2, "generator_version": "t"}')
        (root / f"{bank}.duckdb").write_bytes(b"")

    world = World()
    dry = publish.Publish(
        client(world), catalog=CATALOG, schema=SCHEMA, warehouse_id=WAREHOUSE[0], apply=False, out_root=tmp_path
    )
    dry.generator = fake_generator
    printed: list[str] = []
    dry.out = printed.append
    assert publish.run(dry) is True
    assert calls == [] and states(dry, "local") == ["would"] and dry.part == "local"
    assert printed[-1] == "  the other parts wait on the generated data"

    live = publish.Publish(
        client(world), catalog=CATALOG, schema=SCHEMA, warehouse_id=WAREHOUSE[0], apply=True, out_root=tmp_path
    )
    live.generator = fake_generator
    live.out = lambda _: None
    assert publish.ensure_local(live) is True
    assert calls == [(BANK, tmp_path)] and [line.state for line in live.lines] == ["done"]


def test_a_missing_catalog_stops_the_run(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb", catalog_exists=False)
    p, _ = publisher(world, out_root, apply=True)
    with pytest.raises(publish.PublishError, match=f"catalog {CATALOG} does not exist"):
        publish.run(p)
    assert world.log == [] and world.statements == []


def test_a_missing_schema_stops_a_dry_run_and_is_created_by_an_apply(out_root: Path) -> None:
    world = World(local_db=out_root / f"{BANK}.duckdb", schema_exists=False, volume_exists=False)
    dry, printed = publisher(world, out_root, apply=False)
    assert publish.run(dry) is True
    assert states(dry, "schema") == ["would", "would"] and dry.part == "schema"
    assert printed[-1] == "  the other parts wait on the schema"
    assert world.log == [] and world.statements == []

    live, _ = publisher(world, out_root, apply=True)
    assert publish.run(live) is True
    assert world.log[:2] == [f"create schema {CATALOG}.{SCHEMA}", f"create volume {CATALOG}.{SCHEMA}.raw"]
    assert_described_and_keyed(world)


# --- the warehouse and the statement API ------------------------------------------------------------


def test_the_warehouse_is_resolved_by_id_name_variable_or_as_the_only_serverless_one() -> None:
    w = client(World())
    assert publish.resolve_warehouse(w, WAREHOUSE[0], {}) == WAREHOUSE
    assert publish.resolve_warehouse(w, WAREHOUSE[1], {}) == WAREHOUSE
    assert publish.resolve_warehouse(w, None, {publish.WAREHOUSE_VARIABLE: "other1"}) == (
        "other1",
        "Serverless Starter Warehouse",
    )
    with pytest.raises(publish.PublishError, match="pass --warehouse"):
        publish.resolve_warehouse(w, None, {})
    with pytest.raises(publish.PublishError, match="no warehouse with id or name 'nope'"):
        publish.resolve_warehouse(w, "nope", {})
    w.warehouses.list.return_value = [sql.EndpointInfo(id="only", name="the one", enable_serverless_compute=True)]
    assert publish.resolve_warehouse(w, None, {}) == ("only", "the one")


def test_run_sql_polls_and_follows_the_chunks() -> None:
    w = MagicMock(spec=WorkspaceClient)
    w.statement_execution = create_autospec(sql.StatementExecutionAPI, instance=True)
    pending = sql.StatementResponse(statement_id="s9", status=sql.StatementStatus(state=sql.StatementState.PENDING))
    done = sql.StatementResponse(
        statement_id="s9",
        status=sql.StatementStatus(state=sql.StatementState.SUCCEEDED),
        manifest=sql.ResultManifest(
            schema=sql.ResultSchema(
                columns=[
                    sql.ColumnInfo(name="a", type_text="BIGINT"),
                    sql.ColumnInfo(name="t", type_text="TIMESTAMP_NTZ"),
                ]
            )
        ),
        result=sql.ResultData(data_array=[["1", "x"]], next_chunk_index=1),
    )
    w.statement_execution.execute_statement.return_value = pending
    w.statement_execution.get_statement.side_effect = [pending, done]
    w.statement_execution.get_statement_result_chunk_n.return_value = sql.ResultData(
        data_array=cast("list[list[str]]", [["2", None]])
    )
    slept: list[float] = []
    result = publish.run_sql(w, "wh", "SELECT 1", catalog="c", schema="s", sleep=slept.append)
    assert result.columns == [("a", "BIGINT"), ("t", "TIMESTAMP_NTZ")]
    assert result.rows == [["1", "x"], ["2", None]]
    assert result.scalar() == "1"
    assert len(slept) == 2
    kwargs = w.statement_execution.execute_statement.call_args.kwargs
    assert kwargs["catalog"] == "c" and kwargs["schema"] == "s" and kwargs["wait_timeout"] == "50s"
    assert kwargs["on_wait_timeout"] == sql.ExecuteStatementRequestOnWaitTimeout.CONTINUE


def test_run_sql_raises_on_a_failed_statement() -> None:
    w = MagicMock(spec=WorkspaceClient)
    w.statement_execution = create_autospec(sql.StatementExecutionAPI, instance=True)
    w.statement_execution.execute_statement.return_value = sql.StatementResponse(
        statement_id="s1",
        status=sql.StatementStatus(state=sql.StatementState.FAILED, error=sql.ServiceError(message="boom")),
    )
    with pytest.raises(publish.PublishError, match=r"statement failed \(FAILED\): boom"):
        publish.run_sql(w, "wh", "SELECT 1", catalog="c", schema="s", sleep=lambda _: None)
