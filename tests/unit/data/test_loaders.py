# tests/unit/data/test_loaders.py
import polars as pl
import pytest

from ehrdrec.data.canonical import (
    ADMIT_TIME,
    DIAGNOSES,
    HADM_ID,
    MEDICATIONS,
    PROCEDURES,
    SUBJECT_ID,
    VISIT_INDEX,
)
from ehrdrec.data.loaders.mimic3 import MIMIC3Loader
from ehrdrec.exceptions import DataLoaderError


def test_mimic3_loader_returns_dataframe(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    assert isinstance(df, pl.DataFrame)


def test_mimic3_loader_required_columns_present(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    required = {SUBJECT_ID, HADM_ID, VISIT_INDEX, ADMIT_TIME,
                DIAGNOSES, PROCEDURES, MEDICATIONS}
    assert required.issubset(set(df.columns))


def test_mimic3_visit_index_is_contiguous(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    for _, group in df.group_by(SUBJECT_ID):
        indices = group.sort(VISIT_INDEX)[VISIT_INDEX].to_list()
        assert indices == list(range(len(indices)))


def test_mimic3_subject_ids_are_strings(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    assert df[SUBJECT_ID].dtype == pl.Utf8


def test_mimic3_admit_time_is_datetime_utc(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    assert df[ADMIT_TIME].dtype == pl.Datetime("us", "UTC")


def test_mimic3_medications_are_structs(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    inner_type = df[MEDICATIONS].dtype.inner
    assert isinstance(inner_type, pl.Struct)


def test_mimic3_diagnoses_ordered_by_seq_num(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    # patient 1, visit 100 has SEQ_NUM 1="401.9", 2="250.00"
    row = df.filter(pl.col(HADM_ID) == "100")
    assert row[DIAGNOSES][0].to_list() == ["401.9", "250.00"]


def test_mimic3_missing_file_raises(tmp_path):
    with pytest.raises(DataLoaderError, match="not found"):
        MIMIC3Loader(str(tmp_path) + "/").load()


def test_mimic3_left_join_preserves_all_admissions(mimic3_csv_dir):
    df = MIMIC3Loader(mimic3_csv_dir).load()
    # fixture has 3 admissions — all should survive the join
    assert df.height == 3
