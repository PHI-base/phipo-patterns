# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from phipo_patterns import synchronize


@pytest.fixture
def pattern_df():
    df = pd.DataFrame(
        {
            'label': pd.Series(dtype=object),
            'term': pd.Series(dtype=object),
            'modified': pd.Series(dtype=object),
            'previous label': pd.Series(dtype=object),
            'pattern': pd.Series(dtype=object),
            'variables': pd.Series(dtype=object),
            'error': pd.Series(dtype=object),
            'checked': pd.Series(dtype=object),
            'definition': pd.Series(dtype=object),
            'exact_synonym': pd.Series(dtype=object),
            'namespace': pd.Series(dtype=object),
            'subset': pd.Series(dtype=object),
            'alt_id': pd.Series(dtype=object),
            'obsolete': pd.Series(dtype=bool),
        }
    )
    return df.copy()


def test_fill_no_pattern_error(pattern_df):
    upheno_patterns = ['existing pattern']
    pattern_df.pattern = [
        np.nan,
        '?',
        '-',
        'existing pattern',
        'unknown pattern',  # not in upheno_patterns
    ]
    pattern_df.error = [
        np.nan,
        np.nan,
        'ERROR',
        np.nan,
        np.nan,
    ]
    expected = pattern_df.copy()
    expected.error = [
        'NO_PATTERN',
        'NO_PATTERN',
        'ERROR, NO_PATTERN',  # new errors should be merged with old errors
        np.nan,
        'NO_PATTERN',
    ]
    actual = synchronize.fill_no_pattern_error(pattern_df, upheno_patterns)
    assert_frame_equal(actual, expected)


def test_fill_no_variable_error(pattern_df):
    import_namespaces = ['GO']
    pattern_df.variables = [
        np.nan,
        '?',
        '-',
        'not_obo_id',  # not valid OBO ID
        'CHEBI:0000001',  # not in import_namespaces
        'GO:1234567',  # valid
    ]
    pattern_df.error = [
        np.nan,
        np.nan,
        'ERROR',
        np.nan,
        np.nan,
        np.nan,
    ]
    expected = pattern_df.copy()
    expected.error = [
        'NO_VARIABLE',
        'NO_VARIABLE',
        'ERROR, NO_VARIABLE',  # new errors should be merged with old errors
        'NO_VARIABLE',
        'NO_VARIABLE',
        np.nan,
    ]
    actual = synchronize.fill_no_variable_error(pattern_df, import_namespaces)
    assert_frame_equal(actual, expected)


def test_set_obsolete_error(pattern_df):
    pattern_df.label = [
        'obsolete term',
        'active term',
    ]
    pattern_df.error = [
        'NO_VARIABLE',
        np.nan,
    ]
    expected = pattern_df.copy()
    expected.error = [
        'OBSOLETE',  # should override existing errors
        np.nan,
    ]
    actual = synchronize.set_obsolete_error(pattern_df)
    assert_frame_equal(actual, expected)


def test_set_merged_error(pattern_df):
    alt_ids = ['PHIPO:0000001']
    pattern_df.term = [
        'PHIPO:0000001',  # merged term
        'PHIPO:0000002',  # unmerged term
    ]
    pattern_df.error = [
        'NO_VARIABLE',
        np.nan,
    ]
    expected = pattern_df.copy()
    expected.error = [
        'MERGED',  # should override existing errors
        np.nan,
    ]
    actual = synchronize.set_merged_error(pattern_df, alt_ids)
    assert_frame_equal(actual, expected)
