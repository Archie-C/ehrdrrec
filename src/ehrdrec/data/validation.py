from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ehrdrec.data.canonical import (
    ADMIT_TIME,
    CANONICAL_SCHEMA,
    DIAGNOSES,
    HADM_ID,
    MEDICATIONS,
    PROCEDURES,
    REQUIRED_COLUMNS,
    SUBJECT_ID,
    VISIT_INDEX,
)
from ehrdrec.exceptions import SchemaValidationError


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            joined = "\n - ".join(self.errors)
            raise SchemaValidationError(
                f"DataFrame failed canonical schema validation:\n - {joined}"
            )

def _check_required_columns(df: pl.DataFrame, result: ValidationResult) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        result.errors.append(f"missing required columns: {sorted(missing)}")

def _check_column_types(df: pl.DataFrame, result: ValidationResult) -> None:
    for col, expected_dtype in CANONICAL_SCHEMA.items():
        if col not in df.columns:
            continue
        actual = df[col].dtype
        if actual != expected_dtype:
            result.errors.append(
                f"column '{col}': expected {expected_dtype}, got {actual}"
            )

def _check_no_null_required(df: pl.DataFrame, result: ValidationResult) -> None:
    never_null = [
        SUBJECT_ID,
        HADM_ID,
        VISIT_INDEX,
        ADMIT_TIME,
        DIAGNOSES,
        PROCEDURES,
        MEDICATIONS
    ]
    for col in never_null:
        if col not in df.columns:
            continue
        n_null = df[col].null_count()
        if n_null > 0:
            result.errors.append(
                f"column '{col}': contains {n_null} null values (not permitted)"
            )

def _check_visit_index(df: pl.DataFrame, result: ValidationResult) -> None:
    if VISIT_INDEX not in df.columns or SUBJECT_ID not in df.columns:
        return

    invalid = (
        df.sort([SUBJECT_ID, VISIT_INDEX])
        .with_columns(
            pl.int_range(pl.len()).over(SUBJECT_ID).alias("_expected_idx")
        )
        .filter(pl.col(VISIT_INDEX) != pl.col("_expected_idx"))
        .select(SUBJECT_ID)
        .unique()
    )

    if len(invalid) > 0:
        bad_patients = invalid[SUBJECT_ID].to_list()[:5]
        result.errors.append(
            f"visit_index is not 0-based and contiguous for "
            f"{len(invalid)} patient(s), e.g.: {bad_patients}"
        )

def _check_non_empty_medications(df: pl.DataFrame, result: ValidationResult) -> None:
    if MEDICATIONS not in df.columns:
        return

    n_empty = df.filter(pl.col(MEDICATIONS).list.len() == 0).height
    if n_empty > 0:
        result.errors.append(
            f"column 'medications': {n_empty} visit(s) have empty medication "
            f"lists - these rows are not valid training samples"
        )


# TODO: Check if this is the ideal way to load the data?
# Is it better to load it into days?
# Or even keep it split for the preprocessor to decide about?
def _check_no_duplicate_hadm(df: pl.DataFrame, result: ValidationResult) -> None:
    if HADM_ID not in df.columns:
        return
    n_dupes = df.height - df[HADM_ID].n_unique()
    if n_dupes > 0:
        result.errors.append(
            f"column 'hadm_id': {n_dupes} duplicate admission IDs found - "
            f"each visit must appear exactly once"
        )

def _check_min_visits(
    df: pl.DataFrame,
    result: ValidationResult,
    min_visits: int = 2
) -> None:
    """Warn (not error) about patients with fewer than min_visits visits."""
    if SUBJECT_ID not in df.columns:
        return
    visit_counts = df.group_by(SUBJECT_ID).len()
    n_short = visit_counts.filter(pl.col("len") < min_visits).height
    if n_short > 0:
        result.warnings.append(
            f"{n_short} patient(s) have fewer than {min_visits} visits -"
            f"they will be dropped by the preprocessor"
        )

# ---------------------
# PUBLIC API
# ---------------------

def validate(
    df: pl.DataFrame,
    *,
    min_visits: int = 2,
    raise_on_error: bool = True,
) -> ValidationResult:
    """
    Validate a DataFrame against the canonical schema.

    Parameters
    ----------
    df:
        DataFrame to validate.
    min_visits:
        Minimum visits per patient — violations produce a warning, not an error.
    raise_on_error:
        If True (default), raises SchemaValidationError on the first call to
        result.raise_if_invalid(). Set to False to inspect errors manually.

    Returns
    -------
    ValidationResult
        Contains .errors (blocking) and .warnings (non-blocking).
    """

    result = ValidationResult()

    _check_required_columns(df, result)

    # if required columns are missing, type checks will produce noise - break early
    if not result.is_valid:
        if raise_on_error:
            result.raise_if_invalid()
        return result

    _check_column_types(df, result)

    if not result.is_valid:
        if raise_on_error:
            result.raise_if_invalid()
        return result
    _check_no_null_required(df, result)
    _check_no_duplicate_hadm(df, result)
    _check_visit_index(df, result)
    _check_non_empty_medications(df, result)
    _check_min_visits(df, result, min_visits=min_visits)

    if raise_on_error:
        result.raise_if_invalid()

    return result
