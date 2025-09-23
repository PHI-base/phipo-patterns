# SPDX-FileCopyrightText: 2025-present Rothamsted Research
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.resources
import itertools
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Container, Iterable, Optional, Set

import numpy as np
import pandas as pd

from phipo_patterns import patterns
from phipo_patterns.robot import Robot

OBO_ID_PATTERN = re.compile(r'^(?P<ns>[A-Za-z]+):(?P<id>\d+)$', re.ASCII)


def iri_to_obo_id(series: pd.Series) -> pd.Series:
    """
    Convert a series of OBOLibrary PURLs into OBO IDs.

    :param series: Series containing OBOLibrary PURLs (IRIs).
    :type series: pandas.Series
    :return: Series with values converted to compact OBO IDs.
    :rtype: pandas.Series
    """
    pattern = re.compile(r'http://purl\.obolibrary\.org/obo/([A-Z]+)_(\d+)')
    replace = r'\1:\2'
    return series.str.replace(pattern, replace, regex=True)


def collapse_rows(
    df: pd.DataFrame, grouper: Optional[Any] = None, sep: str = ' '
) -> pd.DataFrame:
    """
    Collapse rows with non-duplicate values into a single row. Defaults to grouping by index.

    For each group, non-null values from each column are deduplicated and then
    joined using ``sep``. If any value in a grouped column is NaN, the
    aggregated result for that column is also NaN.

    :param df: the DataFrame to collapse.
    :type df: pandas.DataFrame
    :param grouper: grouping keys or index to use. If ``None``, use the DataFrame
        index.
    :type grouper: Any, optional
    :param sep: separator used to join unique values within each group.
    :type sep: str
    :return: the collapsed DataFrame with one row per group.
    :rtype: pandas.DataFrame
    """

    def aggregator(series: pd.Series):
        if series.isna().any():
            return np.nan
        return sep.join(series.unique().astype('str'))

    grouper2 = df.index if grouper is None else grouper
    return df.groupby(grouper2).agg(aggregator)


def is_newer(path1: Path | str, path2: Path | str) -> bool:
    """
    Test if ``path1`` was modified more recently than ``path2``.

    :param path1: first filesystem path.
    :type path1: Path | str
    :param path2: second filesystem path.
    :type path2: Path | str
    :return: ``True`` if ``path1`` is newer than ``path2``; otherwise ``False``.
    :rtype: bool
    """
    return os.path.getmtime(path1) > os.path.getmtime(path2)


def merge_phipo_dataframes(
    spreadsheet_df: pd.DataFrame, phipo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Update the spreadsheet DataFrame with data from the PHIPO ontology
    DataFrame, while preserving previous labels from the spreadsheet.

    :param spreadsheet_df: the PHIPO pattern mapping table.
    :type spreadsheet_df: pandas.DataFrame
    :param phipo_df: columns queried from PHIPO to be merged into the mapping table.
    :type phipo_df: pandas.DataFrame
    :return: PHIPO pattern mapping table merged with new values from PHIPO.
    :rtype: pandas.DataFrame
    """
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


def fill_missing_indexes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find gaps in the term index and fill them, leaving all columns NaN
    except the 'term' column (which is derived from the index).

    :param df: PHIPO pattern mapping table
    :type df: pandas.DataFrame
    :return: PHIPO pattern mapping table with missing indexes inserted
    :rtype: pandas.DataFrame
    """
    index_range = range(df.index.min(), df.index.max() + 1)
    df = df.reindex(index=index_range)
    all_columns_na = df.isna().all(axis=1)
    missing_indexes = df[all_columns_na].index
    missing_ids = [f'PHIPO:{index:0>7}' for index in missing_indexes]
    df.loc[missing_indexes, 'term'] = missing_ids
    return df


def set_modified_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the 'modified' column to ``True`` where the previous label doesn't
    match the current label.

    :param df: PHIPO pattern mapping table
    :type df: pandas.DataFrame
    :return: PHIPO pattern mapping table with updated 'modified' column.
    :rtype: pandas.DataFrame
    """
    df['modified'] = False
    index = df['label'].fillna('') != df['previous label'].fillna('')
    df.loc[index, 'modified'] = True
    return df


def clear_unchanged_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clear previous term labels that are equal to current term labels.

    :param df: PHIPO pattern mapping table
    :type df: pandas.DataFrame
    :return: PHIPO pattern mapping table with unchanged labels cleared.
    :rtype: pandas.DataFrame
    """
    df['previous label'] = df['previous label'].where(
        df['previous label'] != df['label'], ''
    )
    return df


def all_terms_have_row(obj: pd.DataFrame | pd.Series) -> bool:
    """
    Test if every row has a one-to-one mapping to a term ID.

    Term numbers must increase monotonically with no gaps between numbers. The
    check is performed on the index of the provided object (DataFrame or Series).

    :param obj: DataFrame or Series whose index is expected to be a contiguous
        range of integers with step ``1``.
    :type obj: pandas.DataFrame | pandas.Series
    :return: ``True`` if the index is a contiguous range of integers with step ``1``, otherwise ``False``.
    :rtype: bool
    """
    ids = list(obj.index)
    pairs = itertools.pairwise(ids)
    return all(y == x + 1 for x, y in pairs)


def set_index_to_term_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the index of the PHIPO pattern mapping table to the term number of the OBO ID from the 'term' column.

    :param df: PHIPO pattern mapping table
    :type df: pandas.DataFrame
    :raises ValueError: if a term does not contain a valid numeric ID.
    :return: PHIPO pattern mapping table with updated index
    :rtype: pandas.DataFrame
    """
    term_column = 'term'
    term_id_pattern = r'[A-Z]+[:_](\d+)'
    term_ids = pd.to_numeric(df[term_column].str.extract(term_id_pattern, expand=False))
    term_ids.name = 'id'
    if any(term_ids.isna()):
        raise ValueError(f"Invalid term ID in column '{df[term_column]}'")
    return df.set_index(term_ids)


def mark_merged_terms(df: pd.DataFrame, alt_ids: Container[str]) -> pd.DataFrame:
    """
    Add 'MERGED' to the 'error' column if the term ID of the row
    exists in the list of alternative OBO IDs.

    :param df: PHIPO pattern mapping table.
    :type df: pandas.DataFrame
    :param alt_ids: container of alternative OBO IDs.
    :type alt_ids: Container[str]
    :return: PHIPO pattern mapping table with updated 'error' column.
    :rtype: pandas.DataFrame
    """

    def marker(row: pd.Series):
        merge_label = 'MERGED'
        if row.term in alt_ids:
            if pd.isna(row.error):
                row.error = merge_label
            elif merge_label not in row.error:
                row.error = ', '.join((row.error, merge_label))
        return row

    return df.apply(marker, axis=1)


def merge_value(new_value: str, old_value: str, sep: str) -> str:
    """
    Merge a value into a string of existing values delimited by ``sep``.

    Do not merge the ``new_value`` if it already exists in ``old_value``. The values in the string are lexicographically sorted after merging.

    :param new_value: value to merge in.
    :type new_value: str
    :param old_value: existing values, separated by ``sep``.
    :type old_value: str
    :param sep: separator used in ``old_value``.
    :type sep: str
    :return: updated delimited string with order preserved.
    :rtype: str
    """
    if not old_value:
        return new_value
    values = [s.strip() for s in old_value.split(sep)]
    if new_value in values:
        return old_value
    values.append(new_value)
    return sep.join(sorted(values))


def merge_error(df: pd.DataFrame, indexer: pd.Series, error_value: str) -> pd.DataFrame:
    """
    Merge a new error value into the 'error' column where ``indexer`` is
    ``True``.

    :param df: PHIPO pattern mapping table.
    :type df: pandas.DataFrame
    :param indexer: boolean indexer for rows to update.
    :type indexer: pandas.Series
    :param error_value: the error value to merge into the 'error' column.
    :type error_value: str
    :return: PHIPO pattern mapping table with merged error values.
    :rtype: pandas.DataFrame
    """
    df2 = df.copy()
    df2.loc[indexer, 'error'] = (
        df2.loc[indexer, 'error']
        .fillna('')  # the merge_value function only handles strings
        .apply(lambda x: merge_value(error_value, x, sep=', '))
    )
    return df2


def is_variable_not_obo_id(variables: pd.Series) -> pd.Series:
    """
    Return a boolean Series indicating variables that are not valid OBO IDs.

    :param variables: the 'variables' column from the PHIPO pattern mapping table.
    :type variables: pandas.Series
    :return: boolean Series where ``True`` indicates at least one invalid ID.
    :rtype: pandas.Series
    """

    def any_not_obo_id(variable_str: str) -> bool:
        parts = variable_str.split('; ')
        for variable in parts:
            if not OBO_ID_PATTERN.match(variable.strip()):
                return True
        return False

    return variables.copy().fillna('').apply(any_not_obo_id)


def is_variable_not_imported(
    variables: pd.Series, import_namespaces: Container[str]
) -> pd.Series:
    """
    Return a boolean Series indicating variables with IDs from
    non-imported ontologies.

    :param variables: the 'variables' column from the PHIPO pattern mapping table.
    :type variables: pandas.Series
    :param import_namespaces: OBO namespaces for each imported ontology.
    :type import_namespaces: Container[str]
    :return: boolean Series where ``True`` indicates a non-imported
        term ID.
    :rtype: pandas.Series
    """

    def any_not_imported(variable_str: str) -> bool:
        parts = variable_str.split('; ')
        for variable in parts:
            match = OBO_ID_PATTERN.match(variable.strip())
            if not match:
                return True
            namespace = match.group('ns')
            if namespace not in import_namespaces:
                return True
        return False

    return variables.copy().fillna('').apply(any_not_imported)


def fill_no_pattern_error(
    df: pd.DataFrame, upheno_patterns: Iterable[str]
) -> pd.DataFrame:
    """
    Add the 'NO_PATTERN' error for rows with missing or undefined patterns.

    :param df: PHIPO pattern mapping table.
    :type df: pandas.DataFrame
    :param upheno_patterns: iterable of pattern names from uPheno.
    :type upheno_patterns: Iterable[str]
    :return: PHIPO pattern mapping table with updated error values.
    :rtype: pandas.DataFrame
    """
    index = (
        (df.pattern.isna())
        | (df.pattern == '?')
        | (df.pattern == '-')
        | (~df.pattern.isin(upheno_patterns))
    )
    return merge_error(df, index, 'NO_PATTERN')


def fill_no_variable_error(
    df: pd.DataFrame, import_namespaces: Container[str]
) -> pd.DataFrame:
    """
    Add the 'NO_VARIABLE' error for rows with missing, invalid, or non-imported variable term IDs.

    :param df: PHIPO pattern mapping table.
    :type df: pandas.DataFrame
    :param import_namespaces: OBO namespaces for each imported ontology.
    :type import_namespaces: Container[str]
    :return: PHIPO pattern mapping table with updated error values.
    :rtype: pandas.DataFrame
    """
    index = (
        df.variables.isna()
        | (df.variables == '?')
        | (df.variables == '-')
        | is_variable_not_obo_id(df.variables)
        | is_variable_not_imported(df.variables, import_namespaces)
    )
    return merge_error(df, index, 'NO_VARIABLE')


def set_obsolete_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the 'error' column to 'OBSOLETE' where the term is obsolete.

    :param df: PHIPO pattern mapping table.
    :type df: pandas.DataFrame
    :return: PHIPO pattern mapping table with updated error values.
    :rtype: pandas.DataFrame
    """
    index = (
        df.obsolete.replace('True', True).replace('False', False).fillna(False)
    ) | (df.label.str.match('obsolete'))
    df2 = df.copy()
    df2.loc[index, 'error'] = 'OBSOLETE'
    return df2


def set_merged_error(df: pd.DataFrame, alt_ids: Iterable[str]) -> pd.DataFrame:
    """
    Set the 'error' column to 'MERGED' where the term ID is an alternative OBO ID.

    :param df: PHIPO pattern mapping table.
    :type df: pandas.DataFrame
    :param alt_ids: iterable of alternative OBO IDs.
    :type alt_ids: Iterable[str]
    :return: PHIPO pattern mapping table with updated error values.
    :rtype: pandas.DataFrame
    """
    index = df.term.isin(alt_ids)
    df2 = df.copy()
    df2.loc[index, 'error'] = 'MERGED'
    return df2


def load_upheno_pattern_file(path: Path | str) -> Set[str]:
    """
    Load the uPheno pattern names file into a set of strings.

    Each line of the file is expected to contain one pattern name.

    :param path: Path to a text file containing pattern names, one per line.
    :type path: Path | str
    :return: Set of normalized pattern names.
    :rtype: set[str]
    """
    with open(path, encoding='utf8') as txt:
        upheno_patterns = set(line.strip() for line in txt.readlines())
    return upheno_patterns


def load_pattern_mapping_table(path: Path | str) -> pd.DataFrame:
    """
    Load the PHIPO pattern mapping table from the CSV file and set its index to the term number.

    :param path: path to the PHIPO pattern mapping table in CSV format.
    :type path: Path | str
    :return: PHIPO pattern mapping table indexed by term number.
    :rtype: pandas.DataFrame
    """
    spreadsheet_df = pd.read_csv(path)
    spreadsheet_df = set_index_to_term_number(spreadsheet_df)
    return spreadsheet_df


def export_as_csv(path: Path | str, df: pd.DataFrame) -> None:
    """
    Export a DataFrame as a UTF-8 encoded CSV file, without the index.

    :param path: destination path for the CSV file.
    :type path: Path | str
    :param df: DataFrame to export.
    :type df: pandas.DataFrame
    """
    df.to_csv(path, encoding='utf-8', index=False)


def sync_term_mapping_table(
    spreadsheet_path: Path | str,
    phipo_dir: Path | str,
    upheno_dir: Path | str,
    out_path: Path | str,
) -> None:
    """
    Synchronize the term mapping table with the latest PHIPO and uPheno data.

    Load the PHIPO pattern mapping table, use the ROBOT tool to query information from the working copy of PHIPO, merge new information into the pattern mapping table, update the 'error' column, and export as CSV.

    :param spreadsheet_path: path to the PHIPO pattern mapping table in CSV format.
    :type spreadsheet_path: Path | str
    :param phipo_dir: path to the local PHIPO repository folder.
    :type phipo_dir: Path | str
    :param upheno_dir: path to the local uPheno repository folder.
    :type upheno_dir: Path | str
    :param out_path: destination path for the updated pattern mapping table.
    :type out_path: Path | str
    """
    phipo_path = Path(phipo_dir) / 'src' / 'ontology' / 'phipo-edit.owl'
    query_path = (
        importlib.resources.files('phipo_patterns')
        / 'queries'
        / 'phipo_metadata_table.sparql'
    )
    import_namespaces = {'IDO', 'GO', 'RO', 'CHEBI', 'PATO', 'SO', 'CL'}
    upheno_pattern_definition_paths = patterns.iter_upheno_pattern_definition_paths(
        upheno_dir
    )
    upheno_patterns = list(
        patterns.iter_normalized_pattern_names(upheno_pattern_definition_paths)
    )
    spreadsheet_df = load_pattern_mapping_table(spreadsheet_path)
    robot = Robot()

    # Extract PHIPO term data as a table
    with tempfile.NamedTemporaryFile() as term_data_file:
        term_data_path = term_data_file.name
        robot.query(
            query=str(query_path),
            input_path=str(phipo_path),
            output_path=str(term_data_path),
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
