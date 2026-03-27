# tests/unit/data/test_validation.py
from __future__ import annotations

import polars as pl
import pytest

from ehrdrec.data.canonical import (
    ADMIT_TIME,
    DEATH_TIME,
    DIAGNOSES,
    DISCHARGE_TIME,
    HADM_ID,
    MEDICATIONS,
    PROCEDURES,
    SUBJECT_ID,
    VISIT_INDEX,
)
from ehrdrec.data.validation import (
    ValidationResult,
    _check_column_types,
    _check_min_visits,
    _check_no_duplicate_hadm,
    _check_no_null_required,
    _check_non_empty_medications,
    _check_visit_index,
    validate,
)
from ehrdrec.exceptions import SchemaValidationError


def test_validate_accepts_valid_dataframe(canonical_df):

    result = validate(canonical_df, raise_on_error=False)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


def test_validate_raises_on_invalid_dataframe(canonical_df):
    df = canonical_df.drop(PROCEDURES)

    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate(df, raise_on_error=True)


def test_missing_required_columns_returns_error(canonical_df):
    df = canonical_df.drop(DIAGNOSES)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any("missing required columns" in err for err in result.errors)
    assert any(DIAGNOSES in err for err in result.errors)


def test_wrong_dtype_returns_error(canonical_df):
    df = canonical_df.with_columns(
        pl.col(VISIT_INDEX).cast(pl.Utf8)
    )

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any(f"column '{VISIT_INDEX}'" in err for err in result.errors)
    assert any("expected" in err and "got" in err for err in result.errors)


def test_null_in_required_column_returns_error(canonical_df):
    df = canonical_df.with_columns(
        pl.when(pl.col(HADM_ID) == "h00001")
        .then(None)
        .otherwise(pl.col(HADM_ID))
        .alias(HADM_ID)
    )

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any(f"column '{HADM_ID}'" in err for err in result.errors)
    assert any("null values" in err for err in result.errors)


def test_duplicate_hadm_id_returns_error(canonical_df):
    df = canonical_df.with_columns(
        pl.when(pl.col(HADM_ID) == "h00001")
        .then(pl.lit("h00000"))
        .otherwise(pl.col(HADM_ID))
        .alias(HADM_ID)
    )

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any("duplicate admission IDs" in err for err in result.errors)


def test_non_contiguous_visit_index_returns_error(canonical_df):
    df = canonical_df.with_columns(
        pl.when(
            (pl.col(SUBJECT_ID) == "p000") & (pl.col(HADM_ID) == "h00001")
        )
        .then(pl.lit(2).cast(pl.Int32))
        .otherwise(pl.col(VISIT_INDEX))
        .alias(VISIT_INDEX)
    )

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any(
        "visit_index is not 0-based and contiguous" in err for err in result.errors
    )


def test_empty_medications_list_returns_error():
    rows = [
        {
            SUBJECT_ID: "p000",
            HADM_ID: "h00000",
            VISIT_INDEX: 0,
            ADMIT_TIME: "2150-01-01 00:00:00",
            DISCHARGE_TIME: "2150-02-01 00:00:00",
            DEATH_TIME: None,
            DIAGNOSES: ["401.9"],
            PROCEDURES: ["99213"],
            MEDICATIONS: [{"DRUG": "Aspirin", "NDC": "12345"}],
        },
        {
            SUBJECT_ID: "p000",
            HADM_ID: "h00001",
            VISIT_INDEX: 1,
            ADMIT_TIME: "2151-01-01 00:00:00",
            DISCHARGE_TIME: "2151-02-01 00:00:00",
            DEATH_TIME: None,
            DIAGNOSES: ["250.00"],
            PROCEDURES: ["99213"],
            MEDICATIONS: [],
        },
    ]
    df = pl.DataFrame(rows).with_columns([
        pl.col(SUBJECT_ID).cast(pl.Utf8),
        pl.col(HADM_ID).cast(pl.Utf8),
        pl.col(VISIT_INDEX).cast(pl.Int32),
        pl.col(ADMIT_TIME).str.strptime(
            pl.Datetime("us"),
            "%Y-%m-%d %H:%M:%S"
            ).dt.replace_time_zone("UTC"),
        pl.col(DISCHARGE_TIME).str.strptime(
            pl.Datetime("us"),
            "%Y-%m-%d %H:%M:%S"
        ).dt.replace_time_zone("UTC"),
        pl.col(DEATH_TIME).cast(pl.Datetime("us", "UTC")),
    ])

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any("empty medication lists" in err for err in result.errors)


def test_missing_required_columns_short_circuits_other_checks(canonical_df):
    df = canonical_df.drop([DIAGNOSES, PROCEDURES])

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]


def test_validation_result_raise_if_invalid():
    result = ValidationResult(errors=["bad thing happened"])

    with pytest.raises(SchemaValidationError, match="bad thing happened"):
        result.raise_if_invalid()


def test_validation_result_raise_if_invalid_does_nothing_when_valid():
    result = ValidationResult()

    result.raise_if_invalid()


def test_validation_result_is_valid_property():
    assert ValidationResult().is_valid is True
    assert ValidationResult(errors=["x"]).is_valid is False


def test_min_visits_produces_warning_not_error(canonical_df):
    df = canonical_df.filter(
        ~(
            (pl.col(SUBJECT_ID) == "p001") &
            (pl.col(HADM_ID) == "h00101")
        )
    )

    result = validate(df, raise_on_error=False, min_visits=2)

    assert result.is_valid
    assert result.errors == []
    assert any("fewer than 2 visits" in warning for warning in result.warnings)


def test_wrong_dtype_raises_when_raise_on_error_true(canonical_df):
    df = canonical_df.with_columns(
        pl.col(VISIT_INDEX).cast(pl.Utf8)
    )

    with pytest.raises(SchemaValidationError, match=f"column '{VISIT_INDEX}'"):
        validate(df, raise_on_error=True)


def test_check_no_duplicate_hadm_early_return_when_column_missing(canonical_df):
    df = canonical_df.drop(HADM_ID)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert HADM_ID in result.errors[0]


def test_check_visit_index_early_return_when_column_missing(canonical_df):
    df = canonical_df.drop(VISIT_INDEX)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert VISIT_INDEX in result.errors[0]


def test_check_non_empty_medications_early_return_when_column_missing(canonical_df):
    df = canonical_df.drop(MEDICATIONS)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert MEDICATIONS in result.errors[0]


def test_null_in_medications_required_column_returns_error(canonical_df):
    df = canonical_df.with_columns(
        pl.when(pl.col(HADM_ID) == "h00001")
        .then(None)
        .otherwise(pl.col(MEDICATIONS))
        .alias(MEDICATIONS)
    )

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any(f"column '{MEDICATIONS}'" in err for err in result.errors)
    assert any("null values" in err for err in result.errors)


def test_null_in_diagnoses_required_column_returns_error(canonical_df):
    df = canonical_df.with_columns(
        pl.when(pl.col(HADM_ID) == "h00001")
        .then(None)
        .otherwise(pl.col(DIAGNOSES))
        .alias(DIAGNOSES)
    )

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert any(f"column '{DIAGNOSES}'" in err for err in result.errors)
    assert any("null values" in err for err in result.errors)

def test_check_column_types_skips_missing_optional_columns(canonical_df):
    df = canonical_df.drop(DEATH_TIME)

    result = validate(df, raise_on_error=False)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


def test_check_no_null_required_skips_missing_column(canonical_df):
    df = canonical_df.drop(DIAGNOSES)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert DIAGNOSES in result.errors[0]


def test_check_visit_index_returns_early_when_subject_id_missing(canonical_df):
    df = canonical_df.drop(SUBJECT_ID)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert SUBJECT_ID in result.errors[0]


def test_check_non_empty_medications_returns_early_when_column_missing(canonical_df):
    df = canonical_df.drop(MEDICATIONS)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert MEDICATIONS in result.errors[0]


def test_check_no_duplicate_hadm_returns_early_when_column_missing(canonical_df):
    df = canonical_df.drop(HADM_ID)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert HADM_ID in result.errors[0]


def test_check_min_visits_returns_early_when_subject_id_missing(canonical_df):
    df = canonical_df.drop(SUBJECT_ID)

    result = validate(df, raise_on_error=False)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "missing required columns" in result.errors[0]
    assert SUBJECT_ID in result.errors[0]


def test_validate_with_raise_on_error_false_returns_result_with_errors(canonical_df):
    df = canonical_df.drop(PROCEDURES)

    result = validate(df, raise_on_error=False)

    assert isinstance(result, ValidationResult)
    assert result.is_valid is False
    assert any("missing required columns" in err for err in result.errors)


def test_validate_with_raise_on_error_true_raises_for_dtype_error(canonical_df):
    df = canonical_df.with_columns(
        pl.col(VISIT_INDEX).cast(pl.Utf8)
    )

    with pytest.raises(SchemaValidationError, match=f"column '{VISIT_INDEX}'"):
        validate(df, raise_on_error=True)


def test_validate_with_raise_on_error_true_and_valid_df_returns_result(canonical_df):
    result = validate(canonical_df, raise_on_error=True)

    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


def test_check_column_types_directly_skips_missing_columns(canonical_df):
    df = canonical_df.drop(DEATH_TIME)
    result = ValidationResult()

    _check_column_types(df, result)

    assert result.errors == []


def test_check_no_null_required_directly_skips_missing_column(canonical_df):
    df = canonical_df.drop(DIAGNOSES)
    result = ValidationResult()

    _check_no_null_required(df, result)

    assert result.errors == []


def test_check_visit_index_directly_returns_when_subject_id_missing(canonical_df):
    df = canonical_df.drop(SUBJECT_ID)
    result = ValidationResult()

    _check_visit_index(df, result)

    assert result.errors == []


def test_check_non_empty_medications_directly_returns_when_column_missing(canonical_df):
    df = canonical_df.drop(MEDICATIONS)
    result = ValidationResult()

    _check_non_empty_medications(df, result)

    assert result.errors == []


def test_check_no_duplicate_hadm_directly_returns_when_column_missing(canonical_df):
    df = canonical_df.drop(HADM_ID)
    result = ValidationResult()

    _check_no_duplicate_hadm(df, result)

    assert result.errors == []


def test_check_min_visits_directly_returns_when_subject_id_missing(canonical_df):
    df = canonical_df.drop(SUBJECT_ID)
    result = ValidationResult()

    _check_min_visits(df, result)

    assert result.errors == []
    assert result.warnings == []
