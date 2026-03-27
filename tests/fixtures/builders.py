from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

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


def make_canonical_df(
    n_patients: int = 2,
    visits_per_patient: int = 2
) -> pl.DataFrame:
    """
    Build a minimal valid canonical DataFrame for testing.
    All values are synthetic — no real patient data.
    """
    rows = []
    base_time = datetime(2150, 1, 1, tzinfo=UTC)

    for p in range(n_patients):
        for v in range(visits_per_patient):
            rows.append({
                SUBJECT_ID:      f"p{p:03d}",
                HADM_ID:         f"h{p:03d}{v:02d}",
                VISIT_INDEX:     v,
                ADMIT_TIME:      base_time.replace(year=2150 + v),
                DISCHARGE_TIME:  base_time.replace(year=2150 + v, month=2),
                DEATH_TIME:      None,
                DIAGNOSES:       ["401.9", "250.00"],
                PROCEDURES:      ["99213"],
                MEDICATIONS:     [{"DRUG": "Aspirin", "NDC": "12345"}],
            })

    return pl.DataFrame(rows).with_columns([
        pl.col(SUBJECT_ID).cast(pl.Utf8),
        pl.col(HADM_ID).cast(pl.Utf8),
        pl.col(VISIT_INDEX).cast(pl.Int32),
        pl.col(ADMIT_TIME).cast(pl.Datetime("us", "UTC")),
        pl.col(DISCHARGE_TIME).cast(pl.Datetime("us", "UTC")),
        pl.col(DEATH_TIME).cast(pl.Datetime("us", "UTC")),
    ])

def make_raw_mimic3_csvs(tmp_path) -> str:
    """
    Write minimal MIMIC-III-shaped CSVs to a temp directory.
    Returns the path as a string (matching what MIMIC3Loader expects).
    """
    # ADMISSIONS.csv
    pl.DataFrame({
        "SUBJECT_ID":  [1, 1, 2],
        "HADM_ID":     [100, 101, 200],
        "ADMITTIME":   [
            "2150-01-01 08:00:00",
            "2151-03-01 09:00:00",
            "2148-06-15 10:00:00"
        ],
        "DISCHTIME":   [
            "2150-01-05 12:00:00",
            "2151-03-07 14:00:00",
            "2148-06-20 11:00:00"
        ],
        "DEATHTIME":   [None, None, None],
    }).write_csv(tmp_path / "ADMISSIONS.csv")

    # DIAGNOSES_ICD.csv
    pl.DataFrame({
        "SUBJECT_ID": [1, 1, 1, 2],
        "HADM_ID":    [100, 100, 101, 200],
        "SEQ_NUM":    [1, 2, 1, 1],
        "ICD9_CODE":  ["401.9", "250.00", "401.9", "I50.9"],
    }).write_csv(tmp_path / "DIAGNOSES_ICD.csv")

    # PROCEDURES_ICD.csv
    pl.DataFrame({
        "SUBJECT_ID": [1, 2],
        "HADM_ID":    [100, 200],
        "SEQ_NUM":    [1, 1],
        "ICD9_CODE":  ["99213", "37.23"],
    }).write_csv(tmp_path / "PROCEDURES_ICD.csv")

    # PRESCRIPTIONS.csv
    pl.DataFrame({
        "SUBJECT_ID":    [1, 1, 2],
        "HADM_ID":       [100, 101, 200],
        "STARTDATE":     ["2150-01-02", "2151-03-02", "2148-06-16"],
        "DRUG":          ["Aspirin", "Metformin", "Furosemide"],
        "GSN":           ["001", "002", "003"],
        "NDC":           ["111", "222", "333"],
        "PROD_STRENGTH": ["100mg", "500mg", "40mg"],
        "DOSE_VAL_RX":   ["1", "1", "1"],
        "DOSE_UNIT_RX":  ["tab", "tab", "tab"],
    }).write_csv(tmp_path / "PRESCRIPTIONS.csv")

    return str(tmp_path) + "/"
