# Cauchy Bank fraud-ring demo data

Synthetic data for a knowledge-graph demo: the banking book of Cauchy Bank, a fictitious Dutch retail bank, and a
fraud ring hidden in it. Two scripts publish everything to one schema of your Databricks workspace, where you can
build the graph yourself, for example with OntoBricks.

Every customer, account and amount is synthetic. The mule accounts, the ring and the unreported payments are
planted by a seeded script, not inferred.

## What you get

All tables land in the schema you name.

| Tables | Made by |
|---|---|
| `customers`, `accounts`, `transactions`, `fraud_cases`, `loans`, `insurance_policies`, `partners`, `regions`, `merchants`, `interest_rates`, `account_balances_monthly`, `loan_balances_monthly`, `story_events`, and three views | `scripts/publish_data.py` (the generator in `src/bankgen`) |
| `mule_accounts`, `fraud_payment_beneficiaries`, `unreported_ring_payments` | `scripts/build_overlay.py` |

The book holds 10,000 customers from October 2024 to September 2026. In May and June 2026 a phishing and
bank-helpdesk wave hits the payment accounts. The overlay adds where that money went: four mule accounts of ring
`RING-01`, a lone mule for every other case, and a few customers who paid the ring without reporting it.

Every table and column carries a description, and the book's tables carry informational primary and foreign keys.

## Run it

You need [uv](https://docs.astral.sh/uv/), a Databricks CLI profile, an existing catalog you can create a schema
in, and a SQL warehouse.

```sh
uv sync
uv run scripts/publish_data.py  --profile <PROFILE> --catalog <CATALOG> --schema <SCHEMA>
uv run scripts/build_overlay.py --profile <PROFILE> --catalog <CATALOG> --schema <SCHEMA>
```

Each script is a dry run until you add `--apply`, so run both again with it:

1. `publish_data.py --apply` generates the book locally into `data/out/` (a few seconds), creates the schema and a
   volume `raw` if missing, uploads the Parquet files, and creates the tables, views, descriptions and keys. A
   second run finds everything in place.
2. `build_overlay.py --apply` reads the book, builds the overlay, checks its rules and writes the three tables.

Both scripts use `--warehouse <id or name>`, else `DATABRICKS_WAREHOUSE_ID`, else your only serverless warehouse.

The generator and the overlay are seeded: the same code gives the same rows every time, in any workspace.

## Develop

```sh
uv run pytest                            # unit tests, no workspace needed
uv run bankgen all --profile cauchy      # generate, load into DuckDB and run the data checks
```
