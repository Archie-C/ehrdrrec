# src/ehrdrec/data/canonical.py
"""
Canonical Polars schema for multi-label drug recommendation.

Every loader (MIMIC-III, MIMIC-IV, eICU, custom) must produce a DataFrame
that conforms to this schema. Validation is enforced in validation.py.

Design decisions
----------------
-   One row per visit, not per patient
-   Clinical codes are stored as List[str] columns
-   Medication codes are dataset specific
-   visit_index is 0-based
-   Timestamps are kept as Polars Datetime rather than strings so
    duration/gap features can be derived without reparsing.
"""

from __future__ import annotations

import polars as pl

# TODO: Decide if this is even the best option

# --------------------------------------------------------------
# Column name constants
# --------------------------------------------------------------

SUBJECT_ID   = "subject_id"         # str  — unique patient identifier
HADM_ID      = "hadm_id"            # str  — unique visit/admission identifier
VISIT_INDEX  = "visit_index"        # i32  — 0-based visit order per patient
ADMIT_TIME   = "admit_time"         # Datetime[us, UTC]
DISCHARGE_TIME = "discharge_time"   # Datetime[us, UTC]  (nullable)
DIAGNOSES    = "diagnoses"          # List[str]  — diagnoses ordered by seq_num
PROCEDURES   = "procedures"         # List[struct]  — procedure objects
MEDICATIONS  = "medications"        # List[struct]  — medication objects

# nullable columns
DEATH_TIME = "death_time"           # Datetime[us, UTC]  (nullable)

# ---------------------------------------------------------------
# The schema
# ---------------------------------------------------------------

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    SUBJECT_ID:      pl.Utf8,
    HADM_ID:         pl.Utf8,
    VISIT_INDEX:     pl.Int32,
    ADMIT_TIME:      pl.Datetime("us", "UTC"),
    DISCHARGE_TIME:  pl.Datetime("us", "UTC"),   # nullable
    DIAGNOSES:       pl.List(pl.Utf8),
    PROCEDURES:      pl.List(pl.Utf8),
    MEDICATIONS:     pl.List(pl.Struct),
    DEATH_TIME:      pl.Datetime("us", "UTC") # nullable
}

# Columns that every loader MUST produce (others are optional)
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    SUBJECT_ID,
    HADM_ID,
    VISIT_INDEX,
    ADMIT_TIME,
    DIAGNOSES,
    PROCEDURES,
    MEDICATIONS,
})
