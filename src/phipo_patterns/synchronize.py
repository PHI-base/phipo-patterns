# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT

import itertools
import os
import importlib.resources
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from phipo_patterns import patterns
from phipo_patterns.robot import Robot

OBO_ID_PATTERN = re.compile(r'^(?P<ns>[A-Za-z]+):(?P<id>\d+)$', re.ASCII)


def iri_to_obo_id(series):
    """Convert a pandas.Series of OBOLibrary PURLs into prefixed term IDs."""
    pattern = re.compile(r'http://purl\.obolibrary\.org/obo/([A-Z]+)_(\d+)')
    replace = r'\1:\2'
    return series.str.replace(pattern, replace, regex=True)


def collapse_rows(df, grouper=None, sep=' | '):
    """Collapse rows with non-duplicate values into a single row.
    Defaults to grouping by index."""

    def aggregator(series):
        if series.isna().any():
            return np.nan
        return sep.join(series.unique().astype('str'))

    grouper = df.index if not grouper else grouper
    return df.groupby(grouper).agg(aggregator)


def is_newer(path1, path2):
    """True if path1 was modified more recently than path2, else False."""
    return os.path.getmtime(path1) > os.path.getmtime(path2)


def merge_phipo_dataframes(spreadsheet_df, phipo_df):
    """Update the spreadsheet DataFrame with data from the PHIPO ontology
    DataFrame, while preserving previous labels from the spreadsheet."""
    # Update previous label column, except where label is NaN
    previous_labels = spreadsheet_df['previous label']
    current_labels = spreadsheet_df['label']
    updated_labels = previous_labels.mask(current_labels.notna(), current_labels)
    spreadsheet_df['previous label'] = updated_labels
    # Update spreadsheet columns with new data from PHIPO
    for column in phipo_df.columns:
        if column not in spreadsheet_df.columns:
            spreadsheet_df[column] = phipo_df[column]
    merged_df = spreadsheet_df.copy()
    merged_df.update(phipo_df)
    # Set rows not in PHIPO to NaN, but preserve previous label and term ID
    preserved_columns = ['previous label', 'term']
    not_in_phipo = ~merged_df.index.isin(phipo_df.index)
    not_in_preserved_columns = ~merged_df.columns.isin(preserved_columns)
    merged_df.loc[not_in_phipo, not_in_preserved_columns] = np.nan
    # Add new terms not in spreadsheet_df; those that were missed by update()
    new_terms_df = phipo_df.loc[phipo_df.index > spreadsheet_df.index.max()]
    merged_df = pd.concat([merged_df, new_terms_df])
    merged_df = fill_missing_indexes(merged_df)
    return merged_df


def fill_missing_indexes(df):
    """Find gaps in the term index and fill them, leaving all columns NaN
    except 'term', which is generated from the index."""
    df = df.reindex(index=range(df.index.min(), df.index.max() + 1))
    missing_indexes = df[df.isna().all(axis=1)].index
    missing_ids = ['PHIPO:{:0>7}'.format(ix) for ix in missing_indexes]
    df.loc[missing_indexes, 'term'] = missing_ids
    return df


def set_modified_column(df):
    """Set 'modified' column to `True` where the previous label doesn't
    match the current label."""
    df['modified'] = False
    index = df['label'].fillna('') != df['previous label'].fillna('')
    df.loc[index, 'modified'] = True
    return df


def clear_unchanged_labels(df):
    """Clear previous term labels that are equal to current term labels."""
    df['previous label'] = df['previous label'].where(
        df['previous label'] != df['label'], ''
    )
    return df


def all_terms_have_row(df):
    """Test if every row has a one-to-one mapping to a term ID.
    Term numbers must increase monotonically, and there must be
    no gaps between numbers."""
    ids = list(df.index)
    pairs = itertools.pairwise(ids)
    return all(y == x + 1 for x, y in pairs)


def set_index_to_term_number(df):
    """Set the Index of a DataFrame to the term number of an OBO term ID,
    or IRI, in the `col` column."""
    term_column = 'term'
    term_id_pattern = r'[A-Z]+[:_](\d+)'
    term_ids = pd.to_numeric(df[term_column].str.extract(term_id_pattern, expand=False))
    term_ids.name = 'id'
    if any(term_ids.isna()):
        raise ValueError(f"Invalid term ID in column '{df[term_column]}'")
    return df.set_index(term_ids)


def mark_merged_terms(df, alt_ids):
    """Add 'MERGED' to the 'error' column if the term ID of the row exists
    in the the list of alternative IDs."""

    def marker(row: pd.Series):
        merge_label = 'MERGED'
        if row.term in alt_ids:
            if pd.isna(row.error):
                row.error = merge_label
            elif merge_label not in row.error:
                row.error = ', '.join((row.error, merge_label))
        return row

    return df.apply(marker, axis=1)


def merge_value(new_value, old_value, sep):
    """Merge a value into existing values, delimited by `sep`.
    Do not merge the value if it already exists."""
    if not old_value:
        return new_value
    values = [s.strip() for s in old_value.split(sep)]
    if new_value in values:
        return old_value
    values.append(new_value)
    return sep.join(sorted(values))


def merge_error(df, indexer, error_value):
    """Merge a new error value into the error column where
    indexer is True."""
    df2 = df.copy()
    df2.error.loc[indexer] = (
        df.error.loc[indexer]
        .fillna('')  # the merge_value function only handles strings
        .apply(lambda x: merge_value(error_value, x, sep=', '))
    )
    return df2


def is_variable_not_obo_id(variables):
    def any_not_obo_id(variable_str):
        variables = variable_str.split('; ')
        for variable in variables:
            if not OBO_ID_PATTERN.match(variable.strip()):
                return True
        return False

    return variables.copy().fillna('').apply(any_not_obo_id)


def is_variable_not_imported(variables, import_namespaces):
    def any_not_imported(variable_str):
        variables = variable_str.split('; ')
        for variable in variables:
            match = OBO_ID_PATTERN.match(variable.strip())
            if not match:
                return True
            namespace = match.group('ns')
            if namespace not in import_namespaces:
                return True
        return False

    return variables.copy().fillna('').apply(any_not_imported)


def fill_no_pattern_error(df, upheno_patterns):
    index = (
        (df.pattern.isna())
        | (df.pattern == '?')
        | (df.pattern == '-')
        | (~df.pattern.isin(upheno_patterns))
    )
    return merge_error(df, index, 'NO_PATTERN')


def fill_no_variable_error(df, import_namespaces):
    index = (
        df.variables.isna()
        | (df.variables == '?')
        | (df.variables == '-')
        | is_variable_not_obo_id(df.variables)
        | is_variable_not_imported(df.variables, import_namespaces)
    )
    return merge_error(df, index, 'NO_VARIABLE')


def set_obsolete_error(df):
    index = (
        df.obsolete.replace('True', True).replace('False', False).fillna(False)
    ) | (df.label.str.match('obsolete'))
    df2 = df.copy()
    df2.loc[index, 'error'] = 'OBSOLETE'
    return df2


def set_merged_error(df, alt_ids):
    index = df.term.isin(alt_ids)
    df2 = df.copy()
    df2.loc[index, 'error'] = 'MERGED'
    return df2


def load_upheno_pattern_file(path):
    with open(path, encoding='utf8') as txt:
        upheno_patterns = set(line.strip() for line in txt.readlines())
    return upheno_patterns


def load_pattern_mapping_table(path):
    spreadsheet_df = pd.read_csv(path)
    spreadsheet_df = set_index_to_term_number(spreadsheet_df)
    return spreadsheet_df


def export_as_csv(path, df):
    df.to_csv(path, encoding='utf-8', index=False)


def sync_term_mapping_table(
    spreadsheet_path: str,
    phipo_dir: str,
    upheno_dir: str,
    out_path: str,
) -> None:
    phipo_path = Path(phipo_dir) / 'src' / 'ontology' / 'phipo_edit.owl'
    query_path = (
        importlib.resources.files('phipo_patterns')
        / 'queries'
        / 'phipo_metadata_table.sparql'
    )
    import_namespaces = {'IDO', 'GO', 'RO', 'CHEBI', 'PATO', 'SO', 'CL'}
    upheno_pattern_definition_paths = patterns.iter_upheno_pattern_definition_paths(upheno_dir)
    upheno_patterns = list(
        patterns.iter_normalized_pattern_names(upheno_pattern_definition_paths)
    )
    spreadsheet_df = load_pattern_mapping_table(spreadsheet_path)
    robot = Robot()

    # Extract PHIPO term data as a table
    with tempfile.NamedTemporaryFile() as term_data_path:
        robot.query(
            query=query_path, input_path=phipo_path, output_path=term_data_path
        )
        phipo_df = pd.read_csv(term_data_path)

    phipo_df = set_index_to_term_number(phipo_df)
    alt_ids = set(phipo_df.alt_id.dropna())
    phipo_df = collapse_rows(phipo_df)
    phipo_df.term = iri_to_obo_id(phipo_df.term)

    merged_df = merge_phipo_dataframes(spreadsheet_df, phipo_df)
    assert all_terms_have_row(merged_df)
    merged_df = set_modified_column(merged_df)
    merged_df = mark_merged_terms(merged_df, alt_ids)
    merged_df = fill_no_pattern_error(merged_df, upheno_patterns)
    merged_df = fill_no_variable_error(merged_df, import_namespaces)
    merged_df = set_obsolete_error(merged_df)
    merged_df = set_merged_error(merged_df, alt_ids)

    # Prepare for export
    export_df = merged_df.copy()
    export_df = clear_unchanged_labels(export_df)
    export_df.modified = export_df.modified.replace(False, '')
    export_df.namespace = export_df.namespace.replace('_', ' ', regex=True)
    export_df = export_df.sort_values('term')

    no_term_ids_are_blank = ~export_df.term.isna().any()
    no_term_ids_are_duplicated = ~export_df.term.duplicated().any()
    columns_are_unchanged = export_df.columns.equals(spreadsheet_df.columns)
    rows_have_been_added = len(export_df) >= len(spreadsheet_df)

    assert no_term_ids_are_blank
    assert no_term_ids_are_duplicated
    assert all_terms_have_row(export_df.term)
    assert columns_are_unchanged
    assert rows_have_been_added

    export_as_csv(out_path, export_df)
