# src/ehrdrec/data/loaders/mimic3.py
from __future__ import annotations

from pathlib import Path

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
from ehrdrec.data.loaders.base import BaseLoader
from ehrdrec.exceptions import DataLoaderError


class MIMIC3Loader(BaseLoader):

    def load(self) -> pl.DataFrame:
        diagnoses, procedures, prescriptions, admissions = self._read_data()
        diagnoses = self._process_diags(diagnoses)
        procedures = self._process_procs(procedures)
        prescriptions = self._process_meds(prescriptions)
        admissions = self._process_adms(admissions)
        data = self._combine_tables(diagnoses, procedures, prescriptions, admissions)
        data = self._add_visit_index(data)
        return data

    # ------------------------------------------------------------------
    # Private: reading
    # ------------------------------------------------------------------

    def _read_data(
        self
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        root = Path(self.source)
        expected = [
            "DIAGNOSES_ICD.csv",
            "PROCEDURES_ICD.csv",
            "PRESCRIPTIONS.csv",
            "ADMISSIONS.csv",
        ]
        for fname in expected:
            if not (root / fname).exists():
                raise DataLoaderError(
                    f"MIMIC-III file not found: {root / fname}"
                )

        diagnoses = pl.read_csv(root / "DIAGNOSES_ICD.csv")
        procedures = pl.read_csv(root / "PROCEDURES_ICD.csv")
        prescriptions = pl.read_csv(
            root / "PRESCRIPTIONS.csv",
            infer_schema_length=10000,
        )
        admissions = pl.read_csv(
            root / "ADMISSIONS.csv",
            infer_schema_length=10000,
        )
        return diagnoses, procedures, prescriptions, admissions

    # ------------------------------------------------------------------
    # Private: processing each table
    # ------------------------------------------------------------------

    def _process_diags(self, diags: pl.DataFrame) -> pl.DataFrame:
        return (
            diags
            .sort(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"])
            .group_by(["SUBJECT_ID", "HADM_ID"])
            .agg(pl.col("ICD9_CODE").alias(DIAGNOSES))
            .rename({"SUBJECT_ID": SUBJECT_ID, "HADM_ID": HADM_ID})
            .with_columns([
                pl.col(SUBJECT_ID).cast(pl.Utf8),
                pl.col(HADM_ID).cast(pl.Utf8),
            ])
        )

    def _process_procs(self, procs: pl.DataFrame) -> pl.DataFrame:
        return (
            procs
            .sort(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"])
            .group_by(["SUBJECT_ID", "HADM_ID"])
            .agg(pl.col("ICD9_CODE").alias(PROCEDURES))
            .rename({"SUBJECT_ID": SUBJECT_ID, "HADM_ID": HADM_ID})
            .with_columns([
                pl.col(SUBJECT_ID).cast(pl.Utf8),
                pl.col(HADM_ID).cast(pl.Utf8),
            ])
        )

    def _process_meds(self, meds: pl.DataFrame) -> pl.DataFrame:
        med_struct_fields = [
            "STARTDATE", "DRUG", "GSN", "NDC",
            "PROD_STRENGTH", "DOSE_VAL_RX", "DOSE_UNIT_RX",
        ]
        return (
            meds
            .with_columns(
                pl.struct(med_struct_fields).alias(MEDICATIONS)
            )
            .group_by(["SUBJECT_ID", "HADM_ID"])
            .agg(pl.col(MEDICATIONS))
            .rename({"SUBJECT_ID": SUBJECT_ID, "HADM_ID": HADM_ID})
            .with_columns([
                pl.col(SUBJECT_ID).cast(pl.Utf8),
                pl.col(HADM_ID).cast(pl.Utf8),
            ])
        )

    def _process_adms(self, adms: pl.DataFrame) -> pl.DataFrame:
        datetime_fmt = "%Y-%m-%d %H:%M:%S"

        return (
            adms
            .select(["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME"])
            .rename({
                "SUBJECT_ID": SUBJECT_ID,
                "HADM_ID": HADM_ID,
                "ADMITTIME": ADMIT_TIME,
                "DISCHTIME": DISCHARGE_TIME,
                "DEATHTIME": DEATH_TIME,
            })
            .with_columns([
                pl.col(SUBJECT_ID).cast(pl.Utf8),
                pl.col(HADM_ID).cast(pl.Utf8),
                pl.col(ADMIT_TIME).str.strptime(
                    pl.Datetime("us"), datetime_fmt, strict=False
                ).dt.replace_time_zone("UTC"),
                pl.col(DISCHARGE_TIME).str.strptime(
                    pl.Datetime("us"), datetime_fmt, strict=False
                ).dt.replace_time_zone("UTC"),
                pl.col(DEATH_TIME).str.strptime(
                    pl.Datetime("us"), datetime_fmt, strict=False
                ).dt.replace_time_zone("UTC"),
            ])
        )

    # ------------------------------------------------------------------
    # Private: combining and indexing
    # ------------------------------------------------------------------

    def _combine_tables(
        self,
        diags: pl.DataFrame,
        procs: pl.DataFrame,
        meds: pl.DataFrame,
        adms: pl.DataFrame,
    ) -> pl.DataFrame:
        return (
            adms
            .join(meds, on=[SUBJECT_ID, HADM_ID], how="left")
            .join(diags, on=[SUBJECT_ID, HADM_ID], how="left")
            .join(procs,  on=[SUBJECT_ID, HADM_ID], how="left")
        )

    def _add_visit_index(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .sort([SUBJECT_ID, ADMIT_TIME])
            .with_columns(
                pl.int_range(pl.len())
                .over(SUBJECT_ID)
                .cast(pl.Int32)
                .alias(VISIT_INDEX)
            )
        )
