import pytest

from tests.fixtures.builders import (
    make_canonical_df,
    make_raw_mimic3_csvs,
)


@pytest.fixture
def mimic3_csv_dir(tmp_path):
    return make_raw_mimic3_csvs(tmp_path)

@pytest.fixture
def canonical_df():
    return make_canonical_df()
