"""Build the small training-serving parity fixture from the real feature parquet.

Writes ``tests/fixtures/serving_parity_rows.parquet`` (about 300 rows, no body)
with columns uid, from, subject, date, priority and every stored feature column.

Selection rules (deterministic: seed 42, sorted by uid before sampling, and each
row is taken by the first rule that claims it, so no row is chosen twice):

1. leadership senders (``sender_is_leadership == 1``), up to 40 rows
2. null or empty subjects, up to 20 rows
3. subjects with a leading RE:/FW:/FWD:/REW: marker, up to 40 rows
4. subjects containing a question mark, up to 30 rows
5. null or empty sender addresses (after ``normalize_sender``), up to 20 rows
6. null or unparseable dates (``hour_of_day == -1``), up to 20 rows
7. a random remainder, up to the 300-row total

Run from the repo root::

    python -m scripts.build_parity_fixture
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from src.features.features import normalize_sender
from src.serving import config

logger = logging.getLogger(__name__)

SEED: int = 42
TOTAL_ROWS: int = 300
MAX_FIXTURE_BYTES: int = 100_000
DEFAULT_OUTPUT: Path = (
    config.REPO_ROOT / "tests" / "fixtures" / "serving_parity_rows.parquet"
)
FIXTURE_COLUMNS: tuple[str, ...] = ("uid", "from", "subject", "date", "priority")
_THREAD_PREFIX = r"^\s*(?:FWD|FW|REW|RE)(?:\[\d+\])?\s*:"


def _rules(df: pd.DataFrame) -> list[tuple[str, pd.Series, int]]:
    """Return (name, boolean mask, cap) for each edge-case selection rule."""
    subject = df["subject"]
    return [
        ("leadership_sender", df["sender_is_leadership"] == 1, 40),
        ("null_or_empty_subject", subject.fillna("").str.strip() == "", 20),
        (
            "thread_prefix",
            subject.fillna("").str.match(_THREAD_PREFIX, case=False),
            40,
        ),
        ("question_mark", subject.fillna("").str.contains("?", regex=False), 30),
        ("null_or_empty_sender", df["from"].map(normalize_sender) == "", 20),
        ("null_or_bad_date", df["hour_of_day"] == -1, 20),
    ]


def select_rows(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Pick the fixture rows and return them with per-rule selection counts."""
    df = df.sort_values("uid").reset_index(drop=True)
    chosen: list[int] = []
    counts: dict[str, int] = {}
    taken = pd.Series(False, index=df.index)
    for name, mask, cap in _rules(df):
        pool = df.index[mask & ~taken]
        picked = pd.Series(pool).sample(min(cap, len(pool)), random_state=seed)
        chosen.extend(int(i) for i in picked)
        taken.loc[picked.to_numpy()] = True
        counts[name] = len(picked)
    remainder = max(TOTAL_ROWS - len(chosen), 0)
    pool = df.index[~taken]
    picked = pd.Series(pool).sample(min(remainder, len(pool)), random_state=seed)
    chosen.extend(int(i) for i in picked)
    counts["random_remainder"] = len(picked)
    return df.loc[sorted(chosen)].reset_index(drop=True), counts


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=config.FEATURES_PATH,
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
)
def main(input_path: Path, output_path: Path) -> None:
    """Write the parity fixture and log how many rows each rule contributed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = pd.read_parquet(input_path)
    if "body" in df.columns:
        df = df.drop(columns=["body"])
    missing = [c for c in FIXTURE_COLUMNS if c not in df.columns]
    if missing:
        raise click.ClickException(f"input is missing columns: {missing}")
    fixture, counts = select_rows(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_parquet(output_path, index=False)
    for name, count in counts.items():
        logger.info("selected %-24s %d", name, count)
    size = output_path.stat().st_size
    logger.info("wrote %d rows, %d bytes to %s", len(fixture), size, output_path)
    if size > MAX_FIXTURE_BYTES:
        logger.warning("fixture is %d bytes, above the %d cap", size, MAX_FIXTURE_BYTES)


if __name__ == "__main__":
    main()
